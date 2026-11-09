from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch import nn

from lean_vllm.engine.compilation import PiecewiseBackend, compile_piecewise, mark_dynamic_tokens
from lean_vllm.engine.cuda_graphs import (
    capture_graph,
    full_graph_buffers,
    pad_to_bucket,
    replay_full_graph,
    smallest_fitting,
)
from lean_vllm.engine.input_buffers import InputBuffers
from lean_vllm.models.deepseek_mtp import DeepSeekMTP
from lean_vllm.utils.context import Context, set_context


@dataclass(slots=True)
class DraftInputs:
    """Target inputs and batch metadata needed by the drafter.

    last_index, next_token_ids, and block_table use batch order. sampling_rows maps sampling order to
    batch indices, and num_draft_tokens uses sampling order.
    """

    input_ids: torch.Tensor  # target token ids on the device, including tokens not yet available on the host
    positions: torch.Tensor  # target positions on the device, adjusted for previously rejected drafts
    max_position: int  # host upper bound on the last token position across sampling rows, assuming all drafts kept
    last_index: np.ndarray  # index of each batch row's last token in input_ids
    next_token_ids: np.ndarray  # token after each non-sampling row's chunk; -1 for sampling rows
    sampling_rows: np.ndarray  # batch indices of sampling rows, in sampling order
    num_draft_tokens: np.ndarray  # number of drafts verified per sampling row, in sampling order
    block_table: np.ndarray | None  # None in warmup, which has no cache


class FirstPass(nn.Module):
    """Run the first draft pass from token ids.

    A separate module limits piecewise compilation to step 0's MTP layer. Later passes may use other
    MTP layers and run eagerly or replay full CUDA graphs.
    """

    def __init__(self, drafter: DeepSeekMTP, embed_tokens: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.drafter, self.embed_tokens = drafter, embed_tokens

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.drafter(self.embed_tokens(input_ids), positions, hidden_states)


class MTPProposer:
    """Drafts num_speculative_tokens tokens per sampling row with the checkpoint's MTP layers.

    The first pass fills the drafter's cache over the target's whole batch, pairing each target hidden state
    with the next token's embedding. For each sampling row, the last kept token's hidden state is paired with
    the newly sampled token to produce the first draft. Later passes process one token per sampling row:
    the previous pass's draft and output hidden state, one position further on.

    Accepted-draft counts stay on the device. The first pass still processes rejected token positions, but
    their outputs are not selected for drafting. Later passes read the selected positions on the device,
    without waiting for the host to receive verification results. When enabled by the runner, later passes
    replay full CUDA graphs, and the first pass uses piecewise compilation with the target's bucket sizes.
    """

    graph_bs: list[int] = []  # captured batch sizes; none until capture_cudagraphs
    piecewise_bs: list[int] = []  # the first pass's captured token counts; none until capture_piecewise
    compile_backend: PiecewiseBackend | None = None  # the first pass's pieces, once compiled

    def __init__(
        self,
        drafter: DeepSeekMTP,
        embed_tokens: Callable[[torch.Tensor], torch.Tensor],
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
        num_speculative_tokens: int,
        block_size: int,
        max_model_len: int,
        device: torch.device,
    ):
        self.drafter = drafter
        self.embed_tokens = embed_tokens
        self.compute_logits = compute_logits
        self.num_speculative_tokens = num_speculative_tokens
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.device = device
        self.first_pass = FirstPass(drafter, embed_tokens)
        self.buffers = InputBuffers(device)  # separate buffers keep draft passes from overwriting target inputs
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}  # by MTP layer and batch size

    def compile(self, pool) -> None:
        """Set up piecewise compilation with the supplied graph pool. The next first-pass call traces the model."""
        self.compile_backend = compile_piecewise(self.first_pass, pool)

    @torch.inference_mode()
    def propose(
        self,
        inputs: DraftInputs,
        context: dict,
        target_hidden_states: torch.Tensor,
        num_accepted: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return draft token ids with shape [sampling rows, num_speculative_tokens] after the target step.

        context is the target step's attention context. For each sampling row, num_accepted counts the
        accepted drafts and next_token_ids supplies the token sampled after them. Both tensors stay on the device.
        """
        num_rows = num_accepted.size(0)
        not_sampling = inputs.next_token_ids >= 0
        buffers = self.buffers
        buffers.begin()
        fixed_index = buffers.put("fixed_index", inputs.last_index[not_sampling], torch.int64)
        fixed_token_ids = buffers.put("fixed_token_ids", inputs.next_token_ids[not_sampling], torch.int64)
        first_draft_index = inputs.last_index[inputs.sampling_rows] - inputs.num_draft_tokens
        sample_index = buffers.put("first_draft_index", first_draft_index, torch.int64)
        cu_seqlens_q = buffers.put("cu_seqlens_q", np.arange(num_rows + 1), torch.int32)
        table = None
        if inputs.block_table is not None:  # None in warmup
            table = buffers.put("block_table", inputs.block_table[inputs.sampling_rows], torch.int32)
        buffers.end()

        # Shift token ids to pair each target hidden state with the next token, then fix sequence boundaries.
        # At each sampling row's last kept position, use the newly sampled token. Only that position's output
        # produces the first draft; outputs at later, rejected positions are ignored.
        input_ids = inputs.input_ids.roll(-1)
        input_ids[fixed_index] = fixed_token_ids
        sample_index += num_accepted
        input_ids[sample_index] = next_token_ids
        positions = inputs.positions
        with set_context(**{**context, "logits_indices": None}) as first_ctx:
            hidden_states = self._run_first_pass(input_ids, positions, target_hidden_states, first_ctx)
            hidden_states = hidden_states[sample_index]
            drafts = [self._draft(hidden_states)]
        if not num_rows:
            return torch.empty(0, self.num_speculative_tokens, dtype=torch.int64, device=input_ids.device)
        sample_positions = positions[sample_index]
        steps = [
            self._step_context(sample_positions, table, cu_seqlens_q, inputs.max_position, step)
            for step in range(1, self.num_speculative_tokens)
        ]
        graph_bs = smallest_fitting(self.graph_bs, num_rows)
        for step, (step_positions, step_context) in enumerate(steps, 1):
            with set_context(**step_context) as step_ctx:
                if graph_bs is None or step_ctx.block_tables is None:  # past the largest graph, or warmup
                    hidden_states = self.drafter(self.embed_tokens(drafts[-1]), step_positions, hidden_states, step)
                else:
                    layer = step % self.drafter.num_mtp_layers
                    hidden_states = self._replay(layer, graph_bs, drafts[-1], step_positions, hidden_states, step_ctx)
                drafts.append(self._draft(hidden_states))
        return torch.stack(drafts, dim=1)

    def _draft(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Choose greedy drafts, so rejection sampling needs no draft probabilities.

        The target's compute_logits gathers logits on every rank. Each rank takes its own argmax,
        so draft token ids do not need to be broadcast.
        """
        return self.compute_logits(hidden_states).argmax(dim=-1)

    def _run_first_pass(
        self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor, context: Context
    ) -> torch.Tensor:
        """Run the first pass, padding to the smallest captured bucket that fits, if one exists.

        When a bucket fits, copy the target's hidden states into the first-pass buffers before replay can
        overwrite their storage in the shared graph pool. Return only the actual tokens' outputs.
        """
        n = input_ids.size(0)
        bucket = smallest_fitting(self.piecewise_bs, n)
        if bucket is not None:
            input_ids, positions, hidden_states = pad_to_bucket(
                self.piecewise_vars,
                bucket,
                context,
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
            )
        return self._first_pass(input_ids, positions, hidden_states)[:n]

    def _first_pass(
        self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.compile_backend is not None and not self.compile_backend.pieces:  # this call traces
            mark_dynamic_tokens(input_ids, positions, hidden_states)
        return self.first_pass(input_ids, positions, hidden_states)

    def _step_context(
        self,
        sample_positions: torch.Tensor,
        table: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor,
        max_position: int,
        step: int,
    ) -> tuple[torch.Tensor, dict]:
        """Build positions and decode context for one token per sampling row.

        Advance each row's last kept position by step. Exact positions and key lengths stay on the device;
        max_position supplies the host-side upper bound used for max_seqlen_k.
        """
        cu_q = list(range(cu_seqlens_q.numel()))
        # Only drafts no step would schedule pass max_model_len; clamp them so they stay within the RoPE cache.
        positions = (sample_positions + step).clamp_(max=self.max_model_len - 1)
        context: dict = dict(is_prefill=False, max_seqlen_q=1, cu_seqlens_q=cu_seqlens_q, cu_seqlens_q_host=cu_q)
        if table is None:  # warmup: no cache, so each token attends itself alone
            context.update(cu_seqlens_k=cu_seqlens_q, max_seqlen_k=1, cu_seqlens_k_host=cu_q)
        else:
            lens = positions + 1
            blocks = table.gather(1, (positions // self.block_size).unsqueeze(1)).squeeze(1).long()
            context.update(
                cu_seqlens_k=torch.nn.functional.pad(lens.cumsum(0), (1, 0)).int(),
                max_seqlen_k=min(max_position + step, self.max_model_len - 1) + 1,
                slot_mapping=(blocks * self.block_size + positions % self.block_size).int(),
                context_lens=lens.int(),
                block_tables=table,
            )
        return positions, context

    def _replay(
        self,
        layer: int,
        bs: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Replay the graph for layer and bs rows, returning a view of the actual rows' output buffer."""
        return replay_full_graph(
            self.graphs[(layer, bs)],
            self.graph_vars,
            context,
            bs,
            self.graph_backends,
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
        )

    @torch.inference_mode()
    def capture_piecewise(self, size: int) -> None:
        """Capture each compiled first-pass piece for size tokens using the runner's current capture context.

        The runner calls with the largest bucket first to allocate buffers large enough for every bucket.
        """
        if not self.piecewise_bs:
            weight = next(self.drafter.parameters())
            self.piecewise_vars = dict(
                input_ids=torch.zeros(size, dtype=torch.int64, device=self.device),
                positions=torch.zeros(size, dtype=torch.int64, device=self.device),
                hidden_states=torch.zeros(size, self.drafter.hidden_size, dtype=weight.dtype, device=self.device),
            )
        graph_vars = self.piecewise_vars
        self._first_pass(
            graph_vars["input_ids"][:size], graph_vars["positions"][:size], graph_vars["hidden_states"][:size]
        )
        self.piecewise_bs = sorted({*self.piecewise_bs, size})

    @torch.inference_mode()
    def capture_cudagraphs(self, sizes: list[int], max_num_blocks: int, pool, backends: list) -> None:
        """Capture a single-token pass for each batch size and MTP layer used after the first draft.

        Logit computation stays outside the graphs. backends supplies the attention backends whose hooks
        run before each replay.
        """
        layers = sorted({step % self.drafter.num_mtp_layers for step in range(1, self.num_speculative_tokens)})
        if not layers:
            return  # one draft per row, which the first pass makes
        weight = next(self.drafter.parameters())
        self.graph_vars = full_graph_buffers(
            sizes[-1], 1, max_num_blocks, self.max_model_len, self.drafter.hidden_size, weight.dtype, weight.device
        )
        self.graph_vars["hidden_states"] = torch.zeros_like(self.graph_vars["outputs"])
        self.graph_backends = backends
        for bs in reversed(sizes):
            for layer in layers:
                self.graphs[(layer, bs)] = self._capture(partial(self._graph_pass, layer, bs), pool)
        self.graph_bs = sizes

    def _graph_pass(self, layer: int, bs: int) -> None:
        """Run one token per row using inputs and attention context backed by the graph's buffers."""
        graph_vars = self.graph_vars
        with set_context(
            False,
            slot_mapping=graph_vars["slot_mapping"][:bs],
            context_lens=graph_vars["context_lens"][:bs],
            block_tables=graph_vars["block_tables"][:bs],
            full_graph_size=bs,
        ):
            inputs_embeds = self.embed_tokens(graph_vars["input_ids"][:bs])
            hidden_states = graph_vars["hidden_states"][:bs]
            graph_vars["outputs"][:bs] = self.drafter(inputs_embeds, graph_vars["positions"][:bs], hidden_states, layer)

    @staticmethod
    def _capture(run: Callable[[], None], pool) -> torch.cuda.CUDAGraph:
        run()  # warm up outside capture so initial compilation and allocation can finish
        return capture_graph(run, pool)  # run() creates a fresh context so FlashMLA allocates its schedule in this pool
