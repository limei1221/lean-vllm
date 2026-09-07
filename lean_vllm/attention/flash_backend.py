import torch

from lean_vllm.attention.abstract import AttentionBackend
from lean_vllm.utils.context import Context

_IMPORT_ERROR: ImportError | None = None
try:
    import triton
    import triton.language as tl
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError as e:    # CUDA only
    _IMPORT_ERROR = e
else:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention kernels with a Triton KV-cache scatter. CUDA only."""

    @staticmethod
    def get_name() -> str:
        return "flash_attn"

    @staticmethod
    def is_available() -> bool:
        return _IMPORT_ERROR is None and torch.cuda.is_available()

    @staticmethod
    def supports_cuda_graph() -> bool:
        return True

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping) -> None:
        num_tokens, num_heads, head_dim = key.shape
        dim = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == dim and v_cache.stride(1) == dim
        assert slot_mapping.numel() == num_tokens
        store_kvcache_kernel[(num_tokens,)](
            key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, dim
        )

    def prefill(self, q, k, v, k_cache, v_cache, context: Context) -> torch.Tensor:
        if context.block_tables is not None:    # prefix cache
            k, v = k_cache, v_cache
        return flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=self.scale, causal=True, block_table=context.block_tables,
        )

    def decode(self, q, k_cache, v_cache, context: Context) -> torch.Tensor:
        o = flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context.context_lens, block_table=context.block_tables,
            softmax_scale=self.scale, causal=True,
        )
        return o.squeeze(1)    # match the [batch, heads, dim] contract
