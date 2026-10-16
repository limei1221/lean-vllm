"""Split the traced model at attention ops and compile the remaining pieces with Inductor.

Each compiled piece captures one CUDA graph per token-count bucket. Attention runs eagerly between pieces.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import fx, nn
from torch._dispatch.python import enable_python_dispatcher
from torch._guards import detect_fake_mode
from torch._inductor.compile_fx import compile_fx
from torch.fx.passes.split_module import split_module

from lean_vllm.utils.context import get_context

# Custom ops defined in layers/attention.py. Match these names after removing any .default suffix.
SPLITTING_OPS = ("lean_vllm.attention", "lean_vllm.mla_attention")


def split_at_attention(gm: fx.GraphModule) -> tuple[fx.GraphModule, set[str]]:
    """Put each attention op in its own submodule and group intervening nodes into separate pieces."""
    part, attention, index = {}, set(), 0
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op == "call_function" and str(node.target).removesuffix(".default") in SPLITTING_OPS:
            part[node] = index + 1
            attention.add(f"submod_{index + 1}")
            index += 2
        else:
            part[node] = index
    return split_module(gm, None, lambda node: part[node], keep_original_order=True), attention  # type: ignore[arg-type]


def weak_ref(value: Any) -> Any:
    """Return tensor views that share memory without keeping it alive; recurse through lists and tuples."""
    if isinstance(value, (list, tuple)):
        return type(value)(weak_ref(item) for item in value)
    if not isinstance(value, torch.Tensor):
        return value
    storage = value.untyped_storage()
    unowned = torch._C._construct_storage_from_data_pointer(storage.data_ptr(), value.device, storage.nbytes())
    return value.new_empty(0).set_(unowned, value.storage_offset(), value.shape, value.stride())


class Piece:
    """One compiled piece. When piecewise_size is set, capture or replay the graph for that size bucket."""

    def __init__(self, compiled: Callable, pool):
        self.compiled, self.pool = compiled, pool
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, Any]] = {}

    def __call__(self, *args):
        size = get_context().piecewise_size
        if size is None:
            return self.compiled(*args)
        if size in self.graphs:
            # Replay uses the same input addresses as capture: runner buffers, weights, or an earlier piece's output.
            graph, output = self.graphs[size]
            graph.replay()
            return output
        self.compiled(*args)  # warm up before capture, since first launches may compile or allocate
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.pool):
            output = self.compiled(*args)
        # Cache views without keeping output memory alive, so the pool can reuse it once later pieces read it.
        self.graphs[size] = graph, weak_ref(output)
        return output


class _CompilePieces(fx.Interpreter):
    """Run the split graph on fake inputs and compile each non-attention piece for its symbolic input shapes."""

    def __init__(self, split: fx.GraphModule, attention: set[str], pool):
        super().__init__(split)
        self.attention, self.pool = attention, pool
        self.pieces: list[Piece] = []

    def call_module(self, target, args, kwargs):
        output = super().call_module(target, args, kwargs)
        if target not in self.attention:
            piece = Piece(compile_fx(self.fetch_attr(target), list(args)), self.pool)
            self.pieces.append(piece)
            self.module.__dict__[target] = piece  # bypass nn.Module's setattr, which requires a module here
        return output


class PiecewiseBackend:
    """The torch.compile backend: split and compile the traced graph, keeping pieces to clear on exit."""

    def __init__(self, pool=None):
        self.pool = pool
        self.pieces: list[Piece] = []

    def __call__(self, gm: fx.GraphModule, example_inputs: list) -> Callable:
        split, attention = split_at_attention(gm)
        fake_mode = detect_fake_mode(example_inputs)
        assert fake_mode is not None
        # Reuse Dynamo's fake tensors, including their symbolic sizes, for compilation.
        fake_inputs = [fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in example_inputs]
        compiler = _CompilePieces(split, attention, self.pool)
        with fake_mode, enable_python_dispatcher():
            compiler.run(*fake_inputs)
        self.pieces += compiler.pieces  # only once compilation finishes, so a restarted trace leaves none behind
        return split


def compile_piecewise(model: nn.Module, pool=None) -> PiecewiseBackend:
    """Set up compilation on the model. Call mark_dynamic_tokens on inputs before the first model call."""
    backend = PiecewiseBackend(pool)
    # Skip guards so later steps reuse the trace for any token count, including one.
    model.compile(backend=backend, fullgraph=True, options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe})
    return backend


def mark_dynamic_tokens(*tensors: torch.Tensor):
    """Mark dimension 0 as dynamic. Tracing fails if it treats that token count as a constant."""
    for tensor in tensors:
        torch._dynamo.mark_dynamic(tensor, 0)
