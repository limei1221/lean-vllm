from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None  # token rows used for sampling; None means all rows
    cu_seqlens_q_host: list[int] | None = None  # host copies, so backends plan without a device sync
    cu_seqlens_k_host: list[int] | None = None
    context_chunks: list | None = None  # filled on first use by layers.attention.context_chunks
    mla_decode_metadata: object | None = None  # FlashMLA schedule metadata, created by the first layer that needs it
    decode_splits: dict | None = None  # cached results of split_decodes_and_prefills, keyed by query length
    decode_query_len: int = 1  # queries per leading decode row; includes draft tokens during verification
    attn_metadata: dict | None = None  # backend plans reused across layers, such as FlashInfer wrappers
    piecewise_size: int | None = None  # bucket size for piecewise graph capture or replay
    full_graph_size: int | None = None  # batch size for full graph capture
    num_actual_tokens: int | None = None  # token rows excluding piecewise graph padding; None means all rows


def split_decodes_and_prefills(context: Context, query_len: int = 1) -> tuple[int, Context | None, Context | None]:
    """Split the leading rows with query_len queries each from the rest of the batch.

    Return (num_decode_rows, decode_context, prefill_context). If no rows can use decode attention, return
    (0, None, context). If all rows use decode attention, the prefill context is None.
    The runner puts rows with this query length first, so each group is a slice of the batch. The split is cached
    per query length and shared by every layer, as vLLM's split_decodes_and_prefills with require_uniform.
    A prompt chunk with query_len tokens also uses decode attention because its query shape is the same.
    """
    if context.block_tables is None or context.cu_seqlens_q_host is None:
        return 0, None, context  # no block tables or host query lengths; keep the batch in the prefill group
    if context.decode_splits is None:
        context.decode_splits = {}
    if query_len not in context.decode_splits:
        cu_q, cu_k = context.cu_seqlens_q_host, context.cu_seqlens_k_host
        assert cu_k is not None and context.cu_seqlens_q is not None and context.cu_seqlens_k is not None
        assert context.context_lens is not None
        num_rows = len(cu_q) - 1
        n = next((i for i in range(num_rows) if cu_q[i + 1] - cu_q[i] != query_len), num_rows)
        decodes = prefills = None
        if n:
            decodes = Context(
                cu_seqlens_q=context.cu_seqlens_q[: n + 1],
                cu_seqlens_k=context.cu_seqlens_k[: n + 1],
                cu_seqlens_q_host=cu_q[: n + 1],
                cu_seqlens_k_host=cu_k[: n + 1],
                max_seqlen_q=query_len,
                max_seqlen_k=max(cu_k[i + 1] - cu_k[i] for i in range(n)),
                context_lens=context.context_lens[:n],
                block_tables=context.block_tables[:n],
            )
        if not n:
            prefills = context
        elif n < num_rows:
            sub_q, sub_k = [x - cu_q[n] for x in cu_q[n:]], [x - cu_k[n] for x in cu_k[n:]]
            prefills = Context(
                is_prefill=True,
                cu_seqlens_q=context.cu_seqlens_q[n:] - cu_q[n],
                # Device offset: async rejection moves decode rows back on the device; host lengths stay upper bounds.
                cu_seqlens_k=context.cu_seqlens_k[n:] - context.cu_seqlens_k[n],
                cu_seqlens_q_host=sub_q,
                cu_seqlens_k_host=sub_k,
                max_seqlen_q=max(b - a for a, b in zip(sub_q, sub_q[1:])),
                max_seqlen_k=max(b - a for a, b in zip(sub_k, sub_k[1:])),
                context_lens=context.context_lens[n:],
                block_tables=context.block_tables[n:],
            )
        context.decode_splits[query_len] = (n, decodes, prefills)
    return context.decode_splits[query_len]


_CONTEXT = Context()


def get_context():
    return _CONTEXT


@contextmanager
def set_context(is_prefill: bool, **kwargs):
    """Create a context for one forward pass and restore the previous context on exit, even on error."""
    with use_context(Context(is_prefill, **kwargs)) as context:
        yield context


@contextmanager
def use_context(context: Context):
    """Run a pass in an existing context, e.g. the one a graph was captured in; restore the previous on exit."""
    global _CONTEXT
    previous, _CONTEXT = _CONTEXT, context
    try:
        yield context
    finally:
        _CONTEXT = previous


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
