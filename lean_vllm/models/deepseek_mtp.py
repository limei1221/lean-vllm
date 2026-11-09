import re

import torch
from torch import nn
from transformers import PretrainedConfig

from lean_vllm.layers.layernorm import RMSNorm
from lean_vllm.layers.linear import ReplicatedLinear
from lean_vllm.models.deepseek_v2 import DeepseekV2DecoderLayer, packed_modules_mapping

# Skip these checkpoint weights because the drafter uses the target's embedding and output layers.
SHARED_WEIGHTS = ("embed_tokens.", "shared_head.head.")
OWN_WEIGHTS = ("enorm.", "hnorm.", "eh_proj.", "shared_head.")


class SharedHead(nn.Module):
    """Load only the norm from shared_head. The proposer uses the target's lm_head for logits; in V3 the
    checkpoint's head is a copy of it."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    """Combine a token's embedding with the previous token's hidden state, then run one decoder layer."""

    def __init__(self, config: PretrainedConfig, layer_idx: int, enable_expert_parallel: bool = False) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = SharedHead(config)
        self.mtp_block = DeepseekV2DecoderLayer(config, layer_idx, enable_expert_parallel)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # As vLLM: zero the embedding at position 0.
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        hidden_states = self.eh_proj(torch.cat([self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)], dim=-1))
        hidden_states, residual = self.mtp_block(positions, hidden_states, None)
        # Add the residual and normalize the output for both logits and the next draft step.
        hidden_states, _ = self.shared_head.norm(hidden_states, residual)
        return hidden_states


class DeepSeekMTP(nn.Module):
    """DeepSeek-V3's multi-token prediction layers, stored after the target's decoder layers in the checkpoint.

    Used as the drafter for speculative decoding. The proposer uses the target's embedding and output layers,
    so this module holds neither. The first draft step takes the target's final normalized hidden states;
    later steps take the previous draft step's output.
    """

    def __init__(self, config: PretrainedConfig, enable_expert_parallel: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleList(
            [
                DeepSeekMultiTokenPredictorLayer(config, self.mtp_start_layer_idx + i, enable_expert_parallel)
                for i in range(self.num_mtp_layers)
            ]
        )
        self.packed_modules_mapping = packed_modules_mapping(config)
        layer_ids = "|".join(str(self.mtp_start_layer_idx + i) for i in range(self.num_mtp_layers))
        self._weight_name = re.compile(rf"model\.layers\.({layer_ids})\.(.+)")

    def remap_weight_name(self, name: str) -> str | None:
        """Map checkpoint names from model.layers.{num_hidden_layers + i}.X to layers.{i}.X.

        Decoder weights go under layers.{i}.mtp_block.X. Return None for names outside the MTP layers
        and for embedding and output weights supplied by the target.
        """
        match = self._weight_name.fullmatch(name)
        if match is None or match[2].startswith(SHARED_WEIGHTS):
            return None
        layer, rest = int(match[1]) - self.mtp_start_layer_idx, match[2]
        return f"layers.{layer}.{rest}" if rest.startswith(OWN_WEIGHTS) else f"layers.{layer}.mtp_block.{rest}"

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.layers[spec_step_idx % self.num_mtp_layers](inputs_embeds, positions, hidden_states)
