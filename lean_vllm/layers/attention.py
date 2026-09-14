import dataclasses
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from lean_vllm.attention import AttentionBackend, get_attention_backend
from lean_vllm.utils.context import Context, get_context

# A custom op cannot take modules, so layers are looked up by name. One per process.
_LAYERS: dict[str, "Attention"] = {}


def register_layers(model: nn.Module):
    """Name every attention layer, so `torch.ops.lean_vllm.attention` can find it. Call before forward."""
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.layer_name = name
            _LAYERS[name] = module


def _layer(layer_name: str) -> "Attention":
    layer = _LAYERS.get(layer_name)
    if layer is None:
        raise KeyError(f"attention layer {layer_name!r} was never registered; call register_layers(model)")
    return layer


@torch.library.custom_op("lean_vllm::attention", mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str) -> torch.Tensor:
    return _layer(layer_name).attend(q, k, v)


@attention.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op("lean_vllm::mla_attention", mutates_args=())
def mla_attention(q: torch.Tensor, latent: torch.Tensor, layer_name: str) -> torch.Tensor:
    return _layer(layer_name).attend(q, latent)


@mla_attention.register_fake
def _(q: torch.Tensor, latent: torch.Tensor, layer_name: str) -> torch.Tensor:
    return q.new_empty(q.size(0), q.size(1), _layer(layer_name).v_head_dim)


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

    def kv_cache_shape(self, num_blocks: int, block_size: int) -> tuple[int, ...]:
        """This layer's cache: a key and a value tensor, stacked, in the backend's layout."""
        return (2, *self.backend.get_kv_cache_shape(num_blocks, block_size, self.num_kv_heads, self.head_dim))

    def bind_kv_cache(self, cache: torch.Tensor):
        self.k_cache, self.v_cache = cache[0], cache[1]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Through the opaque op, so torch.compile splits the graph here for piecewise capture.
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


def key_slots(context: Context, block_size: int) -> torch.Tensor:
    """The cache slot of every key the step attends, row after row. Computed once per step."""
    if context.key_slots is None:
        cu_seqlens_k = context.cu_seqlens_k.long()
        rows = torch.repeat_interleave(cu_seqlens_k.diff(), output_size=context.num_keys)
        positions = torch.arange(context.num_keys, device=rows.device) - cu_seqlens_k[rows]
        blocks = context.block_tables[rows, positions // block_size].long()
        context.key_slots = blocks * block_size + positions % block_size
    return context.key_slots


class MLAAttention(Attention):
    """Multi-head latent attention: the cache holds one compressed latent per token.

    Each step expands every latent it reads back into keys and values, then runs the
    backend's varlen prefill over them, so decode rows are just rows one query long.
    """

    def __init__(
        self,
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        scale: float,
        latent_dim: int,
        expand: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        backend: type[AttentionBackend] | None = None,
    ):
        super().__init__(num_heads, qk_head_dim, scale, num_heads, backend)
        self.v_head_dim = v_head_dim
        self.latent_dim = latent_dim
        self.expand = expand    # a method of the owning layer, so not a registered submodule
        self.latent_cache = torch.tensor([])

    def kv_cache_shape(self, num_blocks: int, block_size: int) -> tuple[int, ...]:
        return (1, num_blocks, block_size, self.latent_dim)

    def bind_kv_cache(self, cache: torch.Tensor):
        self.latent_cache = cache[0]

    def forward(self, q: torch.Tensor, latent: torch.Tensor):
        return torch.ops.lean_vllm.mla_attention(q, latent, self.layer_name)

    def attend(self, q: torch.Tensor, latent: torch.Tensor):
        context = get_context()
        cache = self.latent_cache
        if cache.numel():
            # No -1 slots to skip: an MLA step is never padded for a graph.
            cache.view(-1, self.latent_dim).index_copy_(0, context.slot_mapping.long(), latent)
        if context.is_prefill and (context.keys_are_new or context.block_tables is None):
            keys = latent    # every key the step reads is new
        else:
            keys = cache.view(-1, self.latent_dim)[key_slots(context, cache.size(1))]
        k, v = self.expand(keys)
        # Backends take one head size, so values pad up to the keys' and are cut back after.
        v = F.pad(v, (0, self.head_dim - self.v_head_dim))
        unpaged = dataclasses.replace(context, block_tables=None, keys_are_new=True)
        o = self.backend.prefill(q, k, v, self.k_cache, self.v_cache, unpaged)
        return o[..., :self.v_head_dim].contiguous()
