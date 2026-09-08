import torch
from torch import nn

from lean_vllm.attention import AttentionBackend, get_attention_backend
from lean_vllm.utils.context import get_context

# A custom op takes tensors and primitives, not modules, so layers are addressed
# by name through this registry. One per process; each TP worker has its own.
_LAYERS: dict[str, "Attention"] = {}


def register_layers(model: nn.Module):
    """Name every attention layer, so `torch.ops.lean_vllm.attention` can find it.

    Called once the model exists and before it runs, since the op resolves the
    name on every forward.
    """
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.layer_name = name
            _LAYERS[name] = module


@torch.library.custom_op("lean_vllm::attention", mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _LAYERS.get(layer_name)
    if layer is None:
        raise KeyError(f"attention layer {layer_name!r} was never registered; call register_layers(model)")
    return layer.attend(q, k, v)


@attention.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str) -> torch.Tensor:
    return torch.empty_like(q)


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
        self.layer_name = ""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Through the op rather than straight to attend(): a custom op is opaque
        # to torch.compile, so the graph splits here. That is what piecewise CUDA
        # graph capture cuts on, and it keeps the context read and the is_prefill
        # branch out of anything traced.
        return torch.ops.lean_vllm.attention(q, k, v, self.layer_name)

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """The op's body. Writes the KV cache, which the op does not declare as a
        mutation: the cache is module state rather than an argument."""
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.backend.store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            return self.backend.prefill(q, k, v, k_cache, v_cache, context)
        return self.backend.decode(q, k_cache, v_cache, context)
