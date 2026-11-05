"""Shared CUDA graph helpers for the model runner and MTP drafter.

Select graph sizes, prepare fixed buffers, capture graphs, and replay steps.
"""

from bisect import bisect_left
from collections.abc import Callable

import torch

from lean_vllm.utils.context import Context


def smallest_fitting(sizes: list[int], n: int) -> int | None:
    """Return the smallest size >= n from sizes sorted in ascending order, or None if none fits."""
    i = bisect_left(sizes, n)
    return sizes[i] if i < len(sizes) else None


def capture_graph(run: Callable[[], None], pool) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool):
        run()
    torch.cuda.synchronize()
    return graph


def full_graph_buffers(
    max_bs: int,
    query_len: int,
    max_num_blocks: int,
    max_model_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate input and output buffers for up to max_bs sequence rows of query_len tokens each.

    Context lengths start at max_model_len, but nothing captured is sized by them: FlashMLA rebuilds its schedule
    inside the graph from the lengths each replay copies in.
    """
    num_tokens = max_bs * query_len
    return dict(
        input_ids=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        positions=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        context_lens=torch.full((max_bs,), max_model_len, dtype=torch.int32, device=device),
        block_tables=torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=device),
        outputs=torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device),
    )


def replay_full_graph(
    graph,
    buffers: dict[str, torch.Tensor],
    context: Context,
    batch_size: int,
    backends: list,
    **inputs: torch.Tensor,
) -> torch.Tensor:
    """Replay a full decode graph captured for batch_size sequence rows and return this step's token outputs.

    Copy inputs into buffers with matching names. Set padding slots to -1 to skip KV-cache writes and padding
    context lengths to zero so they attend to no tokens. Call each attention backend's hook to refresh any
    metadata it needs before replay. The returned tensor is a view of the reusable output buffer.
    """
    assert context.slot_mapping is not None and context.context_lens is not None
    assert context.block_tables is not None
    num_tokens, num_rows = inputs["input_ids"].size(0), context.context_lens.size(0)
    for name, value in inputs.items():
        buffers[name][:num_tokens] = value
    buffers["slot_mapping"].fill_(-1)
    buffers["slot_mapping"][:num_tokens] = context.slot_mapping
    buffers["context_lens"].zero_()
    buffers["context_lens"][:num_rows] = context.context_lens
    # Columns beyond the graph's block-table capacity hold lookahead blocks that this step does not read.
    width = min(context.block_tables.size(1), buffers["block_tables"].size(1))
    buffers["block_tables"][:num_rows, :width] = context.block_tables[:, :width]
    for backend in backends:  # e.g. FlashInfer updates its decode plan for this step
        backend.before_full_graph_replay(context, batch_size)
    graph.replay()
    return buffers["outputs"][:num_tokens]


def pad_to_bucket(
    buffers: dict[str, torch.Tensor], bucket: int, context: Context, **inputs: torch.Tensor
) -> list[torch.Tensor]:
    """Copy inputs into the start of the buffers and return views of the first bucket rows, in input order.

    Record the bucket size and actual token count in context so attention skips the padding rows.
    """
    num_tokens = next(iter(inputs.values())).size(0)
    for name, value in inputs.items():
        buffers[name][:num_tokens] = value
    context.piecewise_size, context.num_actual_tokens = bucket, num_tokens
    return [buffers[name][:bucket] for name in inputs]
