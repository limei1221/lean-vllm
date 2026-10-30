import dataclasses
from collections.abc import Callable
from itertools import accumulate
from typing import cast

import torch
from torch import nn

from lean_vllm.attention import AttentionBackend, LayerSpec, get_attention_backend
from lean_vllm.attention.merge import merge_attention_
from lean_vllm.utils import device as dev
from lean_vllm.utils.context import Context, get_context, split_decodes_and_prefills

# Attention ops look up modules by name in this process-wide registry.
_LAYERS: dict[str, "Attention"] = {}


def register_layers(model: nn.Module):
    """Register each attention layer by name so attention ops can find it. Call before forward."""
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.layer_name = name
            _LAYERS[name] = module


def _layer(layer_name: str) -> "Attention":
    layer = _LAYERS.get(layer_name)
    if layer is None:
        raise KeyError(f"attention layer {layer_name!r} was never registered; call register_layers(model)")
    return layer


def _actual_rows(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Remove piecewise graph padding so only actual token rows reach attention and the cache."""
    n = get_context().num_actual_tokens
    return tensors if n is None else tuple(tensor[:n] for tensor in tensors)


# The preceding graph piece allocates out, so the ops write to the same address on every graph replay.
@torch.library.custom_op("lean_vllm::attention", mutates_args=("out",))
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, layer_name: str) -> None:
    q, k, v = _actual_rows(q, k, v)
    _layer(layer_name).attend(q, k, v, out[: q.size(0)])


@attention.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, layer_name: str) -> None:
    return None


@torch.library.custom_op("lean_vllm::mla_attention", mutates_args=("out",))
def mla_attention(q: torch.Tensor, latent: torch.Tensor, out: torch.Tensor, layer_name: str) -> None:
    q, latent = _actual_rows(q, latent)
    cast("MLAAttention", _layer(layer_name)).attend(q, latent, out[: q.size(0)])


@mla_attention.register_fake
def _(q: torch.Tensor, latent: torch.Tensor, out: torch.Tensor, layer_name: str) -> None:
    return None


class Attention(nn.Module):
    """Paged causal attention for one layer, executed by the backend chosen for it."""

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
        # The runner sets the default dtype to the model's dtype during construction.
        backend_cls = backend or get_attention_backend(
            LayerSpec(head_dim, num_heads, num_kv_heads, torch.get_default_dtype())
        )
        self.backend = backend_cls(num_heads, head_dim, scale, num_kv_heads)
        self.k_cache = self.v_cache = torch.tensor([])
        self.layer_name = ""

    def kv_cache_shape(self, num_blocks: int, block_size: int, layout: str = "NHD") -> tuple[int, ...]:
        """Return the stacked key/value cache shape, with heads before tokens for HND layout."""
        # (num_blocks, block_size, num_kv_heads, head_dim)
        shape = self.backend.get_kv_cache_shape(num_blocks, block_size, self.num_kv_heads, self.head_dim)
        if layout == "HND":
            shape = (shape[0], shape[2], shape[1], shape[3])
        return (2, *shape)

    def bind_kv_cache(self, cache: torch.Tensor, layout: str = "NHD"):
        if layout == "HND":  # expose (block, token, head, dim) as a view; strides preserve the HND storage layout
            cache = cache.transpose(2, 3)
        self.k_cache, self.v_cache = cache[0], cache[1]

    def output_shape(self, num_tokens: int) -> tuple[int, ...]:
        """Return the attention output shape used by forward to allocate out."""
        return (num_tokens, self.num_heads, self.head_dim)

    def supports_full_cudagraph(self) -> bool:
        """Return whether this layer supports decode inside a full CUDA graph."""
        return self.backend.supports_full_cudagraph()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # The custom op marks where piecewise compilation splits the model; attention runs between graph pieces.
        out = q.new_empty(self.output_shape(q.size(0)))
        torch.ops.lean_vllm.attention(q, k, v, out, self.layer_name)
        return out

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor | None = None):
        """Run attention, writing into out if provided.

        The custom op declares the out mutation, but not writes to the KV cache stored on this module.
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.backend.store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        return self.backend.forward(q, k, v, k_cache, v_cache, context, out)


@dataclasses.dataclass(slots=True)
class ContextChunk:
    """Describe a chunk of cached keys and the queries for its consecutive batch rows."""

    queries: slice  # query tokens for these batch rows in the q passed to _prefill
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    slots: torch.Tensor  # the cache slot of each key
    host_cu_seqlens: tuple[list[int], list[int]]  # host copies of cu_seqlens_q and cu_seqlens_k for backend planning


def plan_context_chunks(context_lens: list[int], budget: int) -> list[tuple[int, list[int], list[int]]]:
    """Pack cached keys into chunks of at most budget keys, splitting long rows across chunks.

    Return (first row, starts, lengths) for each chunk. Starts and lengths count keys within each row.
    """
    assert budget > 0
    chunks = []
    row = start = 0
    while row < len(context_lens):
        if not context_lens[row]:
            row += 1
            continue
        first, starts, lens, used = row, [], [], 0
        while row < len(context_lens) and context_lens[row] and used < budget:
            take = min(context_lens[row] - start, budget - used)
            starts.append(start)
            lens.append(take)
            used += take
            start += take
            if start == context_lens[row]:
                row, start = row + 1, 0
        chunks.append((first, starts, lens))
    return chunks


def context_chunks(context: Context, block_size: int, budget: int) -> list[ContextChunk]:
    """Plan cached-key chunks on the host and reuse them for subsequent calls with this context."""
    if context.context_chunks is None:
        cu_q, cu_k = context.cu_seqlens_q_host, context.cu_seqlens_k_host
        if cu_k is None:  # a draft step's key lengths are only on the device; copying them to the host synchronizes
            assert context.cu_seqlens_k is not None
            cu_k = context.cu_seqlens_k.tolist()
        assert cu_q is not None and context.block_tables is not None
        context_lens = [(cu_k[i + 1] - cu_k[i]) - (cu_q[i + 1] - cu_q[i]) for i in range(len(cu_q) - 1)]
        device = context.block_tables.device
        context.context_chunks = []
        for first, starts, lens in plan_context_chunks(context_lens, budget):
            last = first + len(lens)
            cu_seqlens_k = [0, *accumulate(lens)]
            num_keys = cu_seqlens_k[-1]
            key_rows = torch.repeat_interleave(dev.make_tensor(lens, torch.int64, device), output_size=num_keys)
            offsets = dev.make_tensor([s - c for s, c in zip(starts, cu_seqlens_k)], torch.int64, device)
            positions = torch.arange(num_keys, device=device) + offsets[key_rows]
            blocks = context.block_tables[first + key_rows, positions // block_size].long()
            cu_seqlens_q = [n - cu_q[first] for n in cu_q[first : last + 1]]
            context.context_chunks.append(
                ContextChunk(
                    queries=slice(cu_q[first], cu_q[last]),
                    cu_seqlens_q=dev.make_tensor(cu_seqlens_q, torch.int32, device),
                    cu_seqlens_k=dev.make_tensor(cu_seqlens_k, torch.int32, device),
                    max_seqlen_q=max(cu_q[i + 1] - cu_q[i] for i in range(first, last)),
                    max_seqlen_k=max(lens),
                    slots=blocks * block_size + positions % block_size,
                    host_cu_seqlens=(cu_seqlens_q, cu_seqlens_k),
                )
            )
    return context.context_chunks


class MLAAttention(Attention):
    """Multi-head latent attention: the cache holds one compressed latent per token.

    Expand cached latents in chunks of at most max_context_chunk tokens and merge attention outputs using log-sum-exp.
    Backends with MLA decode support attend directly to cached latents during decode.
    Corresponds to vLLM's MLACommonImpl; splits each step into decode and prefill.
    """

    max_context_chunk = 8192  # the runner replaces this with its per-step token budget

    def __init__(
        self,
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        scale: float,
        latent_dim: int,
        expand: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        latent_projections: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        backend: type[AttentionBackend] | None = None,
    ):
        backend = backend or get_attention_backend(
            LayerSpec(qk_head_dim, num_heads, num_heads, torch.get_default_dtype(), latent_dim)
        )
        super().__init__(num_heads, qk_head_dim, scale, num_heads, backend)
        self.v_head_dim = v_head_dim
        self.latent_dim = latent_dim
        # Values stay narrower than keys: FA3 and FlashInfer support DeepSeek's (192, 128)
        # key/value dimensions, and SDPA supports arbitrary dimensions.
        # No equivalent of vLLM's _pad_v for FlashAttention-2 is needed here.
        self.expand = expand  # the owning layer passes a bound method, not a registered submodule
        self.latent_projections = latent_projections
        self.latent_cache = torch.tensor([])

    def kv_cache_shape(self, num_blocks: int, block_size: int, layout: str = "NHD") -> tuple[int, ...]:
        return (1, num_blocks, block_size, self.latent_dim)  # no heads, so one layout

    def bind_kv_cache(self, cache: torch.Tensor, layout: str = "NHD"):
        self.latent_cache = cache[0]

    def output_shape(self, num_tokens: int) -> tuple[int, ...]:
        return (num_tokens, self.num_heads, self.v_head_dim)

    def supports_full_cudagraph(self) -> bool:
        # A full graph needs latent decode, as expansion needs the step's host plan, and mla_decode safe to replay.
        return self.backend.supports_mla_decode() and self.backend.supports_full_cudagraph_mla_decode()

    def forward(self, q: torch.Tensor, latent: torch.Tensor):  # type: ignore[override]
        out = q.new_empty(self.output_shape(q.size(0)))
        torch.ops.lean_vllm.mla_attention(q, latent, out, self.layer_name)
        return out

    def attend(self, q: torch.Tensor, latent: torch.Tensor, out: torch.Tensor | None = None):  # type: ignore[override]
        context = get_context()
        cache = self.latent_cache
        if not cache.numel():
            return self._prefill(q, latent, context, out)  # warmup: no cache yet, so every key is new
        self.backend.store_latents(latent, cache, context.slot_mapping)
        if not self.backend.supports_mla_decode():
            return self._prefill(q, latent, context, out)  # decode rows expand like prompt rows
        if not context.is_prefill:
            return self._decode_latents(q, context, out)  # pure decode
        # Draft verification gives decode rows several queries; decode_query_len is the batch's most common length
        # up to 1 + drafts, the bound vLLM's reorder_batch_threshold sets.
        n, decodes, prefills = split_decodes_and_prefills(context, context.decode_query_len)
        if decodes is None:
            assert prefills is not None
            return self._prefill(q, latent, prefills, out)
        if prefills is None:
            return self._decode_latents(q, decodes, out)
        if out is None:
            out = q.new_empty(self.output_shape(q.size(0)))
        num_decode_tokens = n * decodes.max_seqlen_q
        self._decode_latents(q[:num_decode_tokens], decodes, out[:num_decode_tokens])
        self._prefill(q[num_decode_tokens:], latent[num_decode_tokens:], prefills, out[num_decode_tokens:])
        return out

    def _prefill(
        self, q: torch.Tensor, latent: torch.Tensor, context: Context, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        cache = self.latent_cache
        k, v = self.expand(latent)
        chunks = [] if context.block_tables is None else context_chunks(context, cache.size(1), self.max_context_chunk)
        if not chunks:
            # No cached keys to chunk (vLLM's has_context is False); attend to the current tokens without a page table.
            unpaged = dataclasses.replace(context, block_tables=None)
            return self.backend.prefill(q, k, v, self.k_cache, self.v_cache, unpaged, out)
        # New tokens attend to one another causally and to cached keys without a causal mask.
        # Merge the cached-chunk outputs first, then combine with the current-token output, writing to out if given.
        cu_seqlens_q, max_seqlen_q = context.cu_seqlens_q, context.max_seqlen_q
        assert cu_seqlens_q is not None
        host = context.cu_seqlens_q_host
        # compute attention among the new tokens
        o, lse = self.backend.varlen_with_lse(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_q,
            causal=True,
            host_cu_seqlens=None if host is None else (host, host),
        )
        del k, v
        # Rows with no cached keys keep context_lse at -inf, giving cached attention zero weight in the final merge.
        # compute and accumulate attention over cached chunks
        context_o = o.new_zeros(o.size(0), self.num_heads, self.v_head_dim)
        context_lse = torch.full_like(lse, float("-inf"))
        latents = cache.view(-1, self.latent_dim)
        for chunk in chunks:
            k, v = self.expand(latents[chunk.slots])
            rows = chunk.queries
            o_chunk, lse_chunk = self.backend.varlen_with_lse(
                q[rows],
                k,
                v,
                chunk.cu_seqlens_q,
                chunk.cu_seqlens_k,
                chunk.max_seqlen_q,
                chunk.max_seqlen_k,
                causal=False,
                host_cu_seqlens=chunk.host_cu_seqlens,
            )
            del k, v
            # rows is a slice, so the merge updates views of context_o and context_lse
            merge_attention_(context_o[rows], context_lse[rows], o_chunk, lse_chunk)
        # merge the cached-prefix and new-token results
        merge_attention_(context_o, context_lse, o, lse, out)
        return context_o if out is None else out  # (num_new_tokens, num_heads, v_head_dim)

    def _decode_latents(self, q: torch.Tensor, context: Context, out: torch.Tensor | None = None) -> torch.Tensor:
        """Attend directly to cached latents without expanding their keys and values.

        q^T (W_UK c) = (W_UK^T q)^T c projects the non-positional query into latent space.
        Apply W_UV to the output afterward because attention is linear in the values.
        """
        W_UK_T, W_UV = self.latent_projections()
        # N = heads, P = qk_nope_head_dim, L = kv_lora_rank; W_UV is (N, L, V) with V = self.v_head_dim
        N, P, L = W_UK_T.shape
        q_nope, q_pe = q.split([P, self.head_dim - P], dim=-1)
        # Project into latent space: (N, B, P) x (N, P, L) -> (N, B, L), then back to (B, N, L)
        ql_nope = torch.bmm(q_nope.transpose(0, 1), W_UK_T).transpose(0, 1)
        o = self.backend.mla_decode(torch.cat([ql_nope, q_pe], dim=-1), self.latent_cache, L, context)  # (B, N, L)
        if out is None:
            out = o.new_empty(o.size(0), N, self.v_head_dim)  # (B, N, V)
        # (N, B, L) x (N, L, V) -> (N, B, V) -> (B, N, V)
        torch.bmm(o.transpose(0, 1), W_UV, out=out.transpose(0, 1))
        return out
