import torch
from torch import nn

from lean_vllm.attention import AttentionBackend, get_attention_backend
from lean_vllm.utils.context import get_context


class Attention(nn.Module):
    """Paged causal attention for one layer, executed by the selected backend."""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        backend: type[AttentionBackend] | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        backend_cls = backend or get_attention_backend()
        self.backend = backend_cls(num_heads, head_dim, scale, num_kv_heads)
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.backend.store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            return self.backend.prefill(q, k, v, k_cache, v_cache, context)
        return self.backend.decode(q, k_cache, v_cache, context)
