"""What the runner and the MTP drafter share to run a step from CUDA graphs: pick a captured size, copy the step into
the graph's fixed buffers, capture and replay."""

from bisect import bisect_left
from collections.abc import Callable

import torch

from lean_vllm.utils.context import Context


def smallest_fitting(sizes: list[int], n: int) -> int | None:
    """The smallest of the ascending sizes that holds n, or None if none does."""
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
    """A full decode graph's inputs and output, for up to max_bs rows of query_len tokens each.

    Key lengths start at max_model_len, so FlashMLA sizes the tile schedule and split-KV workspace it captures for the
    longest row."""
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
    """Replay the full graph captured at batch_size for a pure-decode step, and return the step's rows of its output.

    inputs go into the buffers of the same names. Padding rows write no slot and attend nothing. backends are the
    attention backends whose hooks refresh, before the replay, what the graph reads but cannot compute itself."""
    assert context.slot_mapping is not None and context.context_lens is not None
    assert context.block_tables is not None
    num_tokens, num_rows = inputs["input_ids"].size(0), context.context_lens.size(0)
    for name, value in inputs.items():
        buffers[name][:num_tokens] = value
    buffers["slot_mapping"].fill_(-1)
    buffers["slot_mapping"][:num_tokens] = context.slot_mapping
    buffers["context_lens"].zero_()
    buffers["context_lens"][:num_rows] = context.context_lens
    # Positions stop short of the buffer's width, so the table's blocks past it are lookahead no pass reads.
    width = min(context.block_tables.size(1), buffers["block_tables"].size(1))
    buffers["block_tables"][:num_rows, :width] = context.block_tables[:, :width]
    for backend in backends:  # e.g. FlashInfer re-plans the graph's decode
        backend.before_full_graph_replay(context, batch_size)
    graph.replay()
    return buffers["outputs"][:num_tokens]


def pad_to_bucket(
    buffers: dict[str, torch.Tensor], bucket: int, context: Context, **inputs: torch.Tensor
) -> list[torch.Tensor]:
    """A step's inputs copied into the head of a piecewise bucket's buffers, as the bucket's slices of them. The
    context records the step's own token count, so attention leaves the padding out."""
    num_tokens = next(iter(inputs.values())).size(0)
    for name, value in inputs.items():
        buffers[name][:num_tokens] = value
    context.piecewise_size, context.num_actual_tokens = bucket, num_tokens
    return [buffers[name][:bucket] for name in inputs]
