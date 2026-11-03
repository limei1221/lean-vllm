import torch

from lean_vllm.attention.abstract import LayerSpec
from lean_vllm.attention.flash_backend import FlashAttention3Backend
from lean_vllm.utils.context import Context

_IMPORT_ERROR: ImportError | None = None
try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
except ImportError as e:  # the flashmla extra builds it on Linux x86_64
    _IMPORT_ERROR = e


class FlashMLABackend(FlashAttention3Backend):
    """Use FlashMLA for decode over cached MLA latents and FlashAttention-3 for prefill. Hopper only."""

    supported_kinds = ("mla",)

    @staticmethod
    def get_name() -> str:
        return "flashmla"

    @staticmethod
    def is_available() -> bool:
        # FA3's availability check also enforces the Hopper (sm90) requirement.
        return _IMPORT_ERROR is None and FlashAttention3Backend.is_available()

    @classmethod
    def validate_configuration(cls, spec: LayerSpec) -> list[str]:
        reasons = super().validate_configuration(spec)  # prefill uses FA3 with expanded keys and values
        if spec.latent_dim and spec.latent_dim != 576:
            reasons.append(f"latent width {spec.latent_dim} is not the 512 + 64 FlashMLA decodes")
        return reasons

    @staticmethod
    def supports_mla_decode() -> bool:
        return True

    @staticmethod
    def supports_full_cudagraph_mla_decode() -> bool:
        # The first call's schedule kernel reads context_lens on the GPU. Capture starts with a fresh holder, so
        # each replay rebuilds the schedule from the lengths the runner copied in, and later layers reuse it.
        return True

    @staticmethod
    def mla_block_size() -> int | None:
        return 64

    def mla_decode(self, q, latent_cache, v_dim, context: Context) -> torch.Tensor:
        if context.mla_decode_metadata is None:
            # Create an empty metadata holder; the first decode call fills it, and later layers reuse it.
            context.mla_decode_metadata, _ = get_mla_metadata()
        assert context.context_lens is not None
        batch_size = context.context_lens.size(0)
        o, _ = flash_mla_with_kvcache(
            q.view(batch_size, -1, *q.shape[1:]),  # [batch, queries per sequence, heads, latent_dim]
            latent_cache.unsqueeze(-2),  # add a single KV head shared by all query heads
            context.block_tables,
            context.context_lens,
            v_dim,
            context.mla_decode_metadata,
            softmax_scale=self.scale,
            causal=True,
        )
        return o.flatten(0, 1)  # match the [batch * queries, heads, v_dim] contract
