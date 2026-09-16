import torch

from lean_vllm.attention.flash_backend import FlashAttention3Backend
from lean_vllm.utils.context import Context

_IMPORT_ERROR: ImportError | None = None
try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
except ImportError as e:    # built from source, Hopper only
    _IMPORT_ERROR = e


class FlashMLABackend(FlashAttention3Backend):
    """FlashMLA's dense decode over MLA latents, and FlashAttention-3 for everything else. Hopper only."""

    @staticmethod
    def get_name() -> str:
        return "flashmla"

    @staticmethod
    def is_available() -> bool:
        # The dense decode kernel is built for sm90 alone, as FA3 is.
        return _IMPORT_ERROR is None and FlashAttention3Backend.is_available()

    @staticmethod
    def supports_mla_decode() -> bool:
        return True

    @staticmethod
    def mla_block_size() -> int | None:
        return 64

    def mla_decode(self, q, latent_cache, v_dim, context: Context) -> torch.Tensor:
        if context.mla_decode_metadata is None:
            # Scheduled by the first layer's call; the rest reuse it, as they share the step's lengths.
            context.mla_decode_metadata, _ = get_mla_metadata()
        o, _ = flash_mla_with_kvcache(
            q.unsqueeze(1), latent_cache.unsqueeze(2), context.block_tables, context.context_lens, v_dim,
            context.mla_decode_metadata, softmax_scale=self.scale, causal=True,
        )
        return o.squeeze(1)    # match the [batch, heads, v_dim] contract
