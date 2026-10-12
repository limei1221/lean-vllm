import logging
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import record_function

from lean_vllm.config import FULL_MODES, PIECEWISE_MODES, Config
from lean_vllm.engine.compilation import PiecewiseBackend, compile_piecewise, mark_dynamic_tokens
from lean_vllm.engine.input_buffers import InputBuffers
from lean_vllm.engine.sampled_tokens import SampledTokens
from lean_vllm.engine.sequence import Sequence
from lean_vllm.kv_transfer import KVConnectorMetadata, KVConnectorOutput, KVOutputAggregator, create_worker_connector
from lean_vllm.layers.attention import Attention, MLAAttention, register_layers
from lean_vllm.layers.sampler import Sampler
from lean_vllm.models import get_drafter_class, get_model_class
from lean_vllm.spec_decode import RejectionSampler
from lean_vllm.spec_decode.mtp_proposer import DraftInputs, MTPProposer
from lean_vllm.utils import device as dev
from lean_vllm.utils.context import Context, get_context, set_context, use_context
from lean_vllm.utils.loader import load_model

logger = logging.getLogger(__name__)

# How long a TP worker may wait for its next call; gloo's 30-minute default would kill an idle server.
CALL_TIMEOUT = timedelta(days=365)


def cudagraph_capture_sizes(max_num_seqs: int, max_num_batched_tokens: int) -> list[int]:
    """Return graph capture sizes using vLLM's default size schedule.

    Start with 1, 2, 4, then multiples of 8 below 256 and multiples of 16 from 256 onward. Limit the sizes to
    min(2 * max_num_seqs, 512, max_num_batched_tokens). Include the token budget itself if it fits this limit.
    Piecewise graphs use these sizes as token counts; full decode graphs also apply row and token limits.
    """
    top = min(max_num_seqs * 2, 512, max_num_batched_tokens)
    sizes = [size for size in (1, 2, 4) if size <= top]
    sizes += range(8, min(top + 1, 256), 8)
    sizes += range(256, top + 1, 16)
    if max_num_batched_tokens <= top:
        sizes.append(max_num_batched_tokens)
    return sorted(set(sizes))


def decode_query_len(lens: np.ndarray, num_speculative_tokens: int) -> int:
    """Choose one query length for MLA decode, which requires the same length for every row in a call.

    Use the most common length at or below 1 + num_speculative_tokens. Choose the shorter length on ties,
    or 1 if no length qualifies.
    """
    short = lens[lens <= 1 + num_speculative_tokens]
    if not short.size:
        return 1

    counts = np.bincount(short)
    most_common_length = np.argmax(counts)
    return int(most_common_length)


@dataclass(slots=True)
class InFlightDrafts:
    """Sampled tokens, drafts, and rejection counts left on the device for the next step, indexed by sampling row.

    Related vLLM state: prev_sampled_token_ids, _draft_token_ids, and valid_sampled_token_count.
    """

    next_token_ids: torch.Tensor  # each row's next token, after any accepted drafts
    draft_token_ids: torch.Tensor  # [rows, num_speculative_tokens], for the next step to verify
    num_rejected: torch.Tensor  # rejected drafts per row; subtract these counts from the next step's positions


class ModelRunner:
    cascade_layers: list[Attention] = []  # one layer per type, backend and head counts; all must allow cascade
    proposer: MTPProposer | None = None  # set when speculative decoding is enabled
    num_speculative_tokens = 0
    graph_query_len = 1  # queries per row in a full decode graph: 1, or 1 + drafts when verifying
    in_flight_drafts: InFlightDrafts | None = None  # speculative outputs from the last launched step
    decodes_latents = True  # all MLA backends support decode attention directly on cached latents

    def __init__(self, config: Config, rank: int):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        Sequence.block_size = self.block_size  # spawned workers never run LLMEngine.__init__
        self.device = dev.get_device()
        self.step_kind = "enforced"  # how the last step ran: see _step_kind
        self._prev_tokens: SampledTokens | None = None  # the step still in flight, if any
        self._prev_rows: dict[int, int] | None = None  # seq_id -> sampling row in _prev_tokens
        model_cls = get_model_class(hf_config)
        self.graph_bs: list[int] = []  # captured full graph sizes, in sequence rows
        self.piecewise_bs: list[int] = []  # captured piecewise graph sizes, in tokens
        self.graphs: dict = {}
        self.graph_pool = None  # shared by both capture kinds
        self.compile_backend: PiecewiseBackend | None = None  # holds the pieces and their graphs
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        dist.init_process_group(
            dev.dist_backend(self.device), f"tcp://localhost:{config.dist_port}", world_size=self.world_size, rank=rank
        )
        if self.world_size > 1:
            # Send worker calls through a CPU group, independent of the model's device backend.
            self.call_group = dist.new_group(backend="gloo", timeout=CALL_TIMEOUT)
        dev.set_device(self.device, rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device(self.device)
        model_kwargs = {"enable_expert_parallel": True} if config.enable_expert_parallel else {}
        self.model = model_cls(hf_config, **model_kwargs)
        register_layers(self.model)  # attention ops need registered layers before warmup
        drafter = None
        if config.speculative is not None:
            # Keep the drafter separate so it can compile and capture graphs independently of the target model.
            drafter = get_drafter_class(hf_config)(hf_config, config.enable_expert_parallel)
            register_layers(drafter)
        # Graph support depends on every attention backend, including the drafter's.
        layers = [module for module in self.model.modules() if isinstance(module, Attention)]
        if drafter is not None:
            layers += [module for module in drafter.modules() if isinstance(module, Attention)]
        if rank == 0:
            counts = Counter(layer.backend.get_name() for layer in layers)
            logger.info("attention backends: %s", ", ".join(f"{name} ({n} layers)" for name, n in counts.items()))
        self.attention_backends = list(dict.fromkeys(type(layer.backend) for layer in layers))  # replay hooks
        groups = {(type(layer), type(layer.backend), layer.num_heads, layer.num_kv_heads): layer for layer in layers}
        self.decodes_latents = all(
            layer.backend.supports_mla_decode() for layer in layers if isinstance(layer, MLAAttention)
        )
        self.cascade_layers = list(groups.values())
        self.enforce_eager = (
            config.enforce_eager
            or self.device.type != "cuda"
            or not model_cls.supports_cuda_graph
            or not all(layer.backend.supports_cuda_graph() for layer in layers)
        )
        mode = "none" if self.enforce_eager else config.cudagraph_mode
        self.cudagraph_mode = self._cudagraph_mode(mode, layers)
        if rank == 0 and self.cudagraph_mode != mode:
            logger.info(
                "some attention layer cannot run in a full graph; cudagraph_mode is %r not %r",
                self.cudagraph_mode,
                mode,
            )
        for module in self.model.modules():
            if isinstance(module, MLAAttention):
                # Limit cached-latent expansion to the configured batch token budget.
                module.max_context_chunk = config.max_num_batched_tokens
        load_model(self.model, config.model)
        if self.cudagraph_mode != "none":
            self.compile_model()
        self.sampler = Sampler()
        if drafter is not None:
            assert config.speculative is not None
            for module in drafter.modules():
                if isinstance(module, MLAAttention):
                    module.max_context_chunk = config.max_num_batched_tokens
            load_model(drafter, config.model)
            self.num_speculative_tokens = config.speculative.num_speculative_tokens
            self.rejection_sampler = RejectionSampler()
            self.proposer = MTPProposer(
                drafter,
                self.model.model.embed_tokens,
                self.model.compute_logits,
                self.num_speculative_tokens,
                self.block_size,
                config.max_model_len,
                self.device,
            )
            if self.cudagraph_mode != "none":
                self.proposer.compile(self.graph_pool)
        self.warmup_model()
        self.allocate_kv_cache()
        self.kv_connector = None
        if config.kv_transfer is not None:
            self.kv_connector = create_worker_connector(config, rank)
            self.kv_connector.register_kv_caches(self.kv_cache)
            self.kv_aggregator = KVOutputAggregator(self.world_size)
        if self.cudagraph_mode in FULL_MODES:
            self.capture_cudagraph()
            if self.proposer is not None:
                self.proposer.capture_cudagraphs(
                    self._decode_graph_sizes(), self._max_num_blocks(), self.graph_pool, self.attention_backends
                )
        if self.cudagraph_mode in PIECEWISE_MODES:
            self.capture_piecewise()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        # After warmup and capture, so every rank's samplers start from the same state.
        torch.manual_seed(config.seed)

        if self.world_size > 1 and rank > 0:
            self.loop()

    @staticmethod
    def _cudagraph_mode(mode: str, layers: list[Attention]) -> str:
        """Choose a graph mode supported by every attention layer.

        Full graphs include attention. If any layer cannot be captured, use piecewise graphs when the requested
        mode allows them; otherwise disable graphs. Piecewise graphs leave attention outside graph capture.
        """
        if mode in FULL_MODES and not all(layer.supports_full_cudagraph() for layer in layers):
            return "piecewise" if mode in PIECEWISE_MODES else "none"
        return mode

    def compile_model(self):
        """Set up whole-model tracing and Inductor compilation for pieces split at attention ops.

        The first warmup call triggers tracing and compilation.
        """
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.compile_backend = compile_piecewise(self.model, self.graph_pool)

    def warmup_model(self):
        dev.empty_cache(self.device)
        dev.reset_peak_memory_stats(self.device)
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs)
        self._dummy_sampler_run()
        dev.empty_cache(self.device)

    @torch.inference_mode()
    def _dummy_sampler_run(self):
        """Measure sampler memory with more sampling rows than the prefill warmup may use.

        Each sequence needs one logits row plus one per draft token.
        """
        num_logits = 1 + self.num_speculative_tokens
        num_rows = max(min(self.config.max_num_seqs * num_logits, self.config.max_num_batched_tokens) // num_logits, 1)
        # Use random values because uninitialized hidden states could break the sampler.
        hidden_states = torch.rand(num_rows * num_logits, self.config.hf_config.hidden_size)
        with set_context(False):  # no logits_indices, so every row samples
            logits = self.model.compute_logits(hidden_states)
        # Exercise non-greedy top-k/top-p sampling, whose sort allocates values and indices for the full vocabulary.
        temperatures = torch.full((num_rows,), 0.5, dtype=torch.float32)
        top_k = torch.full((num_rows,), logits.size(1) - 1, dtype=torch.int64)
        top_p = torch.full((num_rows,), 0.9, dtype=torch.float32)
        try:
            if self.proposer is None:
                self.sampler(logits, temperatures)  # compiles the untruncated variant too
                self.sampler(logits, temperatures, top_k, top_p)
            else:
                num_draft_tokens = torch.full((num_rows,), self.num_speculative_tokens, dtype=torch.int64)
                drafts = torch.zeros(num_rows * self.num_speculative_tokens, dtype=torch.int64)
                self.rejection_sampler(
                    logits, drafts, num_draft_tokens, temperatures, self.num_speculative_tokens, top_k, top_p
                )
        except torch.OutOfMemoryError as error:
            raise RuntimeError(
                f"out of memory warming up the sampler with {num_rows} rows; "
                "lower max_num_seqs or gpu_memory_utilization"
            ) from error

    def allocate_kv_cache(self):
        config = self.config
        # Ask each layer for its cache shape: per-head keys and values, or compressed MLA latents.
        layers = [module for module in self.model.modules() if isinstance(module, Attention)]
        if self.proposer is not None:  # drafter caches use the target model's block tables
            layers += [module for module in self.proposer.drafter.modules() if isinstance(module, Attention)]
        block_numel = sum(math.prod(layer.kv_cache_shape(1, self.block_size)) for layer in layers)
        block_bytes = block_numel * config.hf_config.dtype.itemsize
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = dev.kvcache_bytes(self.device, config) // block_bytes
        assert config.num_kvcache_blocks > 0, "no memory left for the kv cache"
        self.kv_cache = [
            torch.empty(layer.kv_cache_shape(config.num_kvcache_blocks, self.block_size)) for layer in layers
        ]
        for layer, cache in zip(layers, self.kv_cache):
            layer.bind_kv_cache(cache)

    def loop(self):
        while True:
            call = [None, None]
            dist.broadcast_object_list(call, src=0, group=self.call_group)
            method_name, args = call
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            dist.broadcast_object_list([method_name, args], src=0, group=self.call_group)
        method = getattr(self, method_name, None)
        return method(*args)

    def exit(self):
        if self.kv_connector is not None:
            self.kv_connector.shutdown()
        if self.world_size > 1:
            dist.barrier()  # sync ranks
        if self.cudagraph_mode != "none":
            for piece in self.compile_backend.pieces:
                piece.graphs.clear()
            if self.proposer is not None:
                self.proposer.graphs.clear()
                if self.proposer.compile_backend is not None:
                    for piece in self.proposer.compile_backend.pieces:
                        piece.graphs.clear()
            del self.graphs, self.graph_pool
        dev.synchronize(self.device)  # wait for pending device work
        dist.destroy_process_group()  # close distributed communication

    def run(self, seqs: list[Sequence]) -> SampledTokens:
        """Prepare inputs, run the model, and sample tokens. Return SampledTokens for the engine to fetch later."""
        with record_function("prepare_batch"):
            input_ids, positions, temperatures, context = self.prepare_batch(seqs)
        with set_context(**context):
            with record_function("run_model"):
                logits = self.run_model(input_ids, positions, context["is_prefill"])
            with record_function("sample"):
                if self.proposer is not None:
                    verified = self._verify(logits, temperatures)
                    tokens = None
                else:
                    # Every rank samples, from the same logits and seed, so each can feed the next step itself.
                    tokens = self.sampler(logits, temperatures, *self._top_k_top_p)
        if self.proposer is not None:
            with record_function("propose"):
                num_accepted = (verified >= 0).sum(dim=1) - 1
                next_token_ids = verified.gather(1, num_accepted.unsqueeze(1)).squeeze(1)
                drafts = self.proposer.propose(
                    self._draft_inputs, context, self._hidden_states, num_accepted, next_token_ids
                )
            self.in_flight_drafts = InFlightDrafts(next_token_ids, drafts, self._spec_inputs[1] - num_accepted)
            tokens = torch.cat([verified, drafts], dim=1)
        pending = SampledTokens(tokens, self.device)
        self._prev_tokens = pending
        self._prev_rows = {seq.seq_id: i for i, seq in enumerate(self._sampling_rows)}
        return pending

    @property
    def input_buffers(self) -> InputBuffers:
        """Create input buffers on first use, including for test runners that skip __init__."""
        buffers = self.__dict__.get("_input_buffers")
        if buffers is None:
            buffers = self._input_buffers = InputBuffers(self.device)
        return buffers

    def prepare_batch(self, seqs: list[Sequence]):
        """Build input tensors and attention context for a batch of prompt chunks, decode rows, or both.

        As in vLLM's _prepare_inputs, collect sequence data on the host and use NumPy for per-token calculations.
        On CUDA, copy each input tensor through reusable buffers. Fill pending tokens and drafts from the
        previous step's device tensors.
        """
        buffers = self.input_buffers
        buffers.begin()
        num_rows, block_size = len(seqs), self.block_size
        # Start in scheduler order so sampled tokens can be returned in that order.
        lens = np.fromiter((seq.num_scheduled_tokens for seq in seqs), np.int64, num_rows)
        starts = np.fromiter((seq.num_cached_tokens for seq in seqs), np.int64, num_rows)
        planned = np.fromiter((seq.num_planned_tokens for seq in seqs), np.int64, num_rows)
        # A decode row runs its current token plus each draft, producing one logits row per position.
        num_drafts = np.fromiter(
            (0 if seq.is_prefill else seq.num_scheduled_tokens - 1 for seq in seqs), np.int64, num_rows
        )
        query_len = decode_query_len(lens, self.num_speculative_tokens)
        order = np.argsort(lens != query_len, kind="stable")  # the decodes_first ordering, expressed as indices
        batch = [seqs[i] for i in order]
        row_of = np.empty(num_rows, np.int64)
        row_of[order] = np.arange(num_rows)  # each scheduled sequence's row in the batch

        lens, starts = lens[order], starts[order]
        # Sample rows that reach the end of their planned tokens, including any draft positions to verify.
        sampling = np.flatnonzero(starts[row_of] + lens[row_of] >= planned)  # in scheduler order
        # Initially place pending rows as if the previous step accepted every draft.
        # Correct their positions on the device unless attention planning needs exact host lengths.
        in_flight = self.in_flight_drafts
        moved = [i for i, seq in enumerate(batch) if seq.num_pending_tokens] if in_flight is not None else []
        moved_src = [self._prev_row(batch[i]) for i in moved]
        # A batch without prompts can use decode attention when all rows have the same query length and all
        # MLA backends support latent decode. Other batches with drafts use the prefill path, which can split rows.
        has_prompt = any(seq.is_prefill for seq in seqs)
        uniform = not has_prompt and self.decodes_latents and bool((lens == query_len).all())
        is_prefill = has_prompt or (bool(num_drafts.any()) and not uniform)
        if moved and (not self.decodes_latents or (is_prefill and bool((lens[moved] != query_len).any()))):
            # Latent expansion needs exact key lengths on the host. Wait for the previous step's rejection counts.
            assert in_flight is not None
            num_rejected = in_flight.num_rejected.tolist()
            starts[moved] -= [num_rejected[src] for src in moved_src]
            moved = []
        ends = starts + lens
        cu_q = np.zeros(num_rows + 1, np.int64)
        np.cumsum(lens, out=cu_q[1:])
        cu_k = np.zeros(num_rows + 1, np.int64)
        np.cumsum(ends, out=cu_k[1:])
        token_rows = np.repeat(np.arange(num_rows), lens)
        positions = starts[token_rows] + np.arange(cu_q[-1]) - cu_q[token_rows]

        token_ids: list[int] = []
        pending_dst: list[int] = []
        pending_src: list[int] = []
        draft_dst: list[int] = []
        draft_src: list[int] = []  # indices into the flattened in-flight draft tensor
        k = self.num_speculative_tokens
        for seq, start, end in zip(batch, starts.tolist(), ends.tolist()):
            if seq.is_prefill:
                assert not seq.num_pending_tokens, "a prefill row carries a pending token"
                token_ids.extend(seq[start:end])
            else:
                if seq.num_pending_tokens:
                    # The current host token and drafts may be stale; replace them from device tensors below.
                    src = self._prev_row(seq)
                    pending_dst.append(len(token_ids))
                    pending_src.append(src)
                    if in_flight is not None:
                        n = len(seq.spec_token_ids)
                        draft_dst += range(len(token_ids) + 1, len(token_ids) + 1 + n)
                        draft_src += range(src * k, src * k + n)
                token_ids.append(seq.last_token)
                token_ids.extend(seq.spec_token_ids)

        sampling_drafts = num_drafts[sampling]
        last_index = cu_q[1:] - 1
        num_logits = sampling_drafts + 1
        offsets = np.arange(num_logits.sum()) - np.repeat(np.cumsum(num_logits) - num_logits, num_logits)
        logits_indices = np.repeat(last_index[row_of[sampling]] - sampling_drafts, num_logits) + offsets
        sampling_rows = [seqs[i] for i in sampling.tolist()]
        row_temperatures = [seq.temperature for seq in sampling_rows]

        block_tables = slot_mapping = None
        common_prefix_len = 0
        tables = [seq.block_table for seq in batch]
        if any(tables):
            table = np.full((num_rows, max(map(len, tables))), -1, np.int32)
            for i, row in enumerate(tables):
                table[i, : len(row)] = row
            # A token's cache slot is its block's starting slot plus its offset within that block.
            # Skip rows without blocks, as in warmup.
            keep = np.fromiter(map(bool, tables), bool, num_rows)[token_rows]
            kept = positions[keep]
            blocks = table[token_rows[keep], kept // block_size].astype(np.int64)
            slot_mapping = blocks * block_size + kept % block_size
            block_tables = buffers.put("block_tables", table, torch.int32)
            common_prefix_len = self._cascade_prefix_len(table, starts, lens)
        cu_seqlens_q, cu_seqlens_k = cu_q.tolist(), cu_k.tolist()
        context = dict(
            is_prefill=is_prefill,
            cu_seqlens_q=buffers.put("cu_seqlens_q", cu_q, torch.int32),
            cu_seqlens_k=buffers.put("cu_seqlens_k", cu_k, torch.int32),
            max_seqlen_q=int(np.max(lens)),
            max_seqlen_k=int(np.max(ends)),
            cu_seqlens_q_host=cu_seqlens_q,
            cu_seqlens_k_host=cu_seqlens_k,
            slot_mapping=buffers.put("slot_mapping", [] if slot_mapping is None else slot_mapping, torch.int32),
            context_lens=buffers.put("context_lens", ends, torch.int32),
            block_tables=block_tables,
            # A pure-decode batch uses every token row for sampling, so skip the logits gather.
            logits_indices=buffers.put("logits_indices", logits_indices, torch.int64) if is_prefill else None,
            common_prefix_len=common_prefix_len,
            decode_query_len=query_len,
        )
        input_ids = buffers.put("input_ids", np.array(token_ids, np.int64), torch.int64)
        positions_t = buffers.put("positions", positions, torch.int64)
        all_greedy = all(temperature == 0 for temperature in row_temperatures)
        temperatures = None if all_greedy else buffers.put("temperatures", row_temperatures, torch.float32)
        # Pass both top_k and top_p or neither. They share one sort and avoid top-k-only and top-p-only variants.
        top_k = top_p = None
        if any(seq.top_k > 0 or seq.top_p < 1 for seq in sampling_rows):
            top_k = buffers.put("top_k", [seq.top_k for seq in sampling_rows], torch.int64)
            top_p = buffers.put("top_p", [seq.top_p for seq in sampling_rows], torch.float32)
        self._top_k_top_p = (top_k, top_p)
        if pending_dst:
            if in_flight is not None:
                prev = in_flight.next_token_ids
            else:
                assert self._prev_tokens is not None
                prev = self._prev_tokens.device_tokens()
            num_pending = len(pending_dst)
            if pending_dst == list(range(num_pending)) == pending_src:
                # Pending tokens occupy the first num_pending positions in both tensors.
                # The previous tensor may have extra rows for requests that have since finished.
                input_ids[:num_pending] = prev[:num_pending]
            else:
                dst = buffers.put("pending_dst", pending_dst, torch.int64)
                src = buffers.put("pending_src", pending_src, torch.int64)
                input_ids.index_copy_(0, dst, prev.index_select(0, src))
        if draft_dst:
            assert in_flight is not None
            dst = buffers.put("draft_dst", draft_dst, torch.int64)
            src = buffers.put("draft_src", draft_src, torch.int64)
            input_ids.index_copy_(0, dst, in_flight.draft_token_ids.flatten().index_select(0, src))
        if self.proposer is not None:
            num_draft_tokens = buffers.put("num_draft_tokens", sampling_drafts, torch.int64)
            # Each draft token follows the input position whose logits verify it.
            is_draft = offsets < np.repeat(sampling_drafts, num_logits)
            draft_token_ids = input_ids[buffers.put("draft_index", logits_indices[is_draft] + 1, torch.int64)]
            self._spec_inputs = (draft_token_ids, num_draft_tokens)
            samples = np.zeros(num_rows, bool)
            samples[row_of[sampling]] = True
            next_token_ids = [
                -1 if row_samples else seq[end] for seq, end, row_samples in zip(batch, ends.tolist(), samples)
            ]
            self._draft_inputs = DraftInputs(
                input_ids=input_ids,
                positions=positions_t,
                max_position=int(positions[last_index[row_of[sampling]]].max(initial=0)),
                last_index=last_index,
                next_token_ids=np.array(next_token_ids, np.int64),
                sampling_rows=row_of[sampling],
                num_draft_tokens=sampling_drafts,
                block_table=table if block_tables is not None else None,
            )
        if moved:
            assert in_flight is not None and block_tables is not None
            self._move_back(context, positions_t, in_flight, token_rows, moved, moved_src, buffers)
        buffers.end()
        self._sampling_rows = sampling_rows
        return input_ids, positions_t, temperatures, context

    def _prev_row(self, seq: Sequence) -> int:
        """Return this sequence's sampling row in the previous step."""
        row = self._prev_rows.get(seq.seq_id) if self._prev_rows else None
        assert row is not None, "a pending token but no row in the launched step"
        return row

    def _move_back(
        self,
        context: dict,
        positions: torch.Tensor,
        in_flight: InFlightDrafts,
        token_rows: np.ndarray,
        moved: list[int],
        moved_src: list[int],
        buffers: InputBuffers,
    ):
        """Adjust pending rows for drafts rejected by the previous step.

        Update positions, cache slots, and key lengths on the device, as in vLLM's
        update_num_computed_tokens_for_batch_change. Host lengths stay unchanged and remain upper bounds.
        """
        back = torch.zeros(len(context["context_lens"]), dtype=torch.int64, device=positions.device)
        rejected = in_flight.num_rejected.index_select(0, buffers.put("moved_src", moved_src, torch.int64))
        back.index_copy_(0, buffers.put("moved", moved, torch.int64), rejected)
        token_rows_t = buffers.put("token_rows", token_rows, torch.int64)
        positions -= back[token_rows_t]
        context_lens, block_size = context["context_lens"], self.block_size
        context_lens -= back.to(context_lens.dtype)
        context["cu_seqlens_k"][1:] = context_lens.cumsum(0)
        blocks = context["block_tables"][token_rows_t, positions // block_size].long()
        context["slot_mapping"].copy_(blocks * block_size + positions % block_size)

    def _cascade_prefix_len(self, table: np.ndarray, starts: np.ndarray, lens: np.ndarray) -> int:
        """Return the shared cached prefix length for cascade attention, or 0 if it cannot be used.

        Compare this batch's block tables and keep only whole blocks before every row's first query.
        Every representative layer must allow cascade attention. This serves the role of vLLM's
        _compute_cascade_attn_prefix_len, using only the requests scheduled in this step.
        """
        if len(starts) < 2 or not self.cascade_layers:
            return 0
        num_pages = int(starts.min()) // self.block_size
        shared = (table[:, :num_pages] == table[0, :num_pages]).all(axis=0)
        common_prefix_len = (num_pages if shared.all() else int(shared.argmin())) * self.block_size
        if common_prefix_len and all(
            layer.use_cascade_attention(common_prefix_len, lens) for layer in self.cascade_layers
        ):
            return common_prefix_len
        return 0

    @staticmethod
    def decodes_first(seqs: list[Sequence], num_speculative_tokens: int = 0) -> list[Sequence]:
        """Put rows matching decode_query_len first so attention backends can split the batch with a slice.

        Preserve order within each group, as in vLLM's reorder_batch. The chosen query length can include drafts;
        if every row has that length, the order stays unchanged.
        """
        lens = np.array([seq.num_scheduled_tokens for seq in seqs], np.int64)
        query_len = decode_query_len(lens, num_speculative_tokens)
        return sorted(seqs, key=lambda seq: seq.num_scheduled_tokens != query_len)

    def _step_kind(self, is_prefill: bool, num_tokens: int, cascade: bool = False, query_len: int = 1) -> str:
        """Return "graph", "piecewise", or a label explaining why this step runs without a graph.

        Full graphs require a pure-decode batch with query_len equal to graph_query_len and a captured size
        large enough for the batch. Cascade attention cannot use these graphs because they capture standard decode.
        """
        if self.cudagraph_mode == "none":
            return "enforced"
        if not is_prefill and not cascade and self.cudagraph_mode in FULL_MODES and self.graph_bs:
            if query_len == self.graph_query_len and num_tokens <= self.graph_bs[-1] * query_len:
                return "graph"
        if self.cudagraph_mode in PIECEWISE_MODES and self._piecewise_bucket(num_tokens):
            return "piecewise"
        # Run without a graph. Label prefill and mixed batches "prefill", and pure-decode batches "decode".
        return "prefill" if is_prefill else "decode"

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        context = get_context()
        self.step_kind = self._step_kind(
            is_prefill, input_ids.size(0), context.common_prefix_len > 0, context.decode_query_len
        )
        if self.compile_backend is not None and not self.compile_backend.pieces:  # the first call traces the model
            mark_dynamic_tokens(input_ids, positions)
        if self.step_kind == "graph":
            hidden_states = self._replay_full(input_ids, positions)
        elif self.step_kind == "piecewise":
            hidden_states = self._replay_piecewise(input_ids, positions)
        else:
            hidden_states = self.model(input_ids, positions)
        if self.proposer is not None:
            self._hidden_states = hidden_states  # inputs for the drafter after sampling
        return self.model.compute_logits(hidden_states)

    def _replay_full(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Replay a full-model graph for a pure-decode batch, including attention.

        Pad to the next captured batch size. Padding rows have zero context length and do not write to the cache.
        """
        num_tokens = input_ids.size(0)
        bs = num_tokens // self.graph_query_len
        context = get_context()
        graph_bs = next(x for x in self.graph_bs if x >= bs)
        graph = self.graphs[graph_bs]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:num_tokens] = input_ids
        graph_vars["positions"][:num_tokens] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:num_tokens] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
        for backend in self.attention_backends:  # e.g. FlashInfer re-plans the graph's decode
            backend.before_full_graph_replay(context, graph_bs)
        graph.replay()
        return graph_vars["outputs"][:num_tokens]

    def _piecewise_bucket(self, num_tokens: int) -> int | None:
        """Return the smallest fitting piecewise bucket, or None for sizes outside the captured range."""
        bucket = next((size for size in self.piecewise_bs if size >= num_tokens), None)
        return None if bucket is None or num_tokens < self.piecewise_bs[0] else bucket

    def _replay_piecewise(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Pad inputs to a piecewise bucket and replay its graphs; attention runs eagerly on unpadded token rows."""
        num_tokens = input_ids.size(0)
        bucket = self._piecewise_bucket(num_tokens)
        buffers = self.piecewise_vars
        buffers["input_ids"][:num_tokens] = input_ids
        buffers["positions"][:num_tokens] = positions
        context = get_context()
        context.piecewise_size, context.num_actual_tokens = bucket, num_tokens
        return self.model(buffers["input_ids"][:bucket], buffers["positions"][:bucket])[:num_tokens]

    def _verify(self, logits: torch.Tensor, temperatures: torch.Tensor | None) -> torch.Tensor:
        """Return accepted drafts and the next token for each sampling row, padded with -1.

        The result has shape [rows, num_speculative_tokens + 1]. Every rank verifies drafts because each rank
        uses the result to run its drafter.
        """
        assert self.proposer is not None
        draft_token_ids, num_draft_tokens = self._spec_inputs
        return self.rejection_sampler(
            logits, draft_token_ids, num_draft_tokens, temperatures, self.num_speculative_tokens, *self._top_k_top_p
        )

    def kv_connector_step(self, metadata: KVConnectorMetadata) -> KVConnectorOutput | None:
        """Start KV loads, make outgoing blocks available, and collect completed transfers across ranks.

        Rank 0 reports transfers completed on every rank; other ranks return None.
        """
        assert self.kv_connector is not None
        self.kv_connector.start_load_kv(metadata)
        output = self.kv_connector.get_finished()
        if self.world_size == 1:
            return output
        outputs: list | None = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(output, outputs, dst=0, group=self.call_group)
        return self.kv_aggregator.aggregate(outputs) if outputs is not None else None

    def _decode_graph_sizes(self) -> list[int]:
        """Return candidate full decode graph sizes in sequence rows, capped at max_num_seqs."""
        config = self.config
        sizes = cudagraph_capture_sizes(config.max_num_seqs, config.max_num_batched_tokens)
        return [size for size in sizes if size <= config.max_num_seqs]

    def _max_num_blocks(self) -> int:
        """Return the block-table width needed for max_model_len plus speculative token slots.

        Include the extra slots because a row may still hold blocks allocated for rejected drafts.
        """
        return (self.config.max_model_len + self.num_speculative_tokens + self.block_size - 1) // self.block_size

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # Capture 1 + num_speculative_tokens queries per row, as in vLLM's uniform_decode_query_len.
        # Steps with a different query length need a piecewise graph or must run without a graph.
        q = self.graph_query_len = 1 + self.num_speculative_tokens
        self.graph_bs = [bs for bs in self._decode_graph_sizes() if bs * q <= config.max_num_batched_tokens]
        if not self.graph_bs:
            return
        max_bs = self.graph_bs[-1]
        max_num_blocks = self._max_num_blocks()
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        # Size FlashMLA's captured tile schedule and split-KV workspace for max_model_len tokens per row.
        # Zero-filled block tables point to valid block 0. Each replay replaces the tables and context lengths
        # with the current batch's values (see _replay_full), which determine how many keys attention reads.
        context_lens.fill_(config.max_model_len)
        self.graph_vars = dict(
            input_ids=torch.zeros(max_bs * q, dtype=torch.int64),
            positions=torch.zeros(max_bs * q, dtype=torch.int64),
            slot_mapping=torch.zeros(max_bs * q, dtype=torch.int32),
            context_lens=context_lens,
            block_tables=torch.zeros(max_bs, max_num_blocks, dtype=torch.int32),
            outputs=torch.zeros(max_bs * q, hf_config.hidden_size),
        )
        graph_vars = self.graph_vars
        for bs in reversed(self.graph_bs):
            with set_context(
                False,
                slot_mapping=graph_vars["slot_mapping"][: bs * q],
                context_lens=context_lens[:bs],
                block_tables=graph_vars["block_tables"][:bs],
                full_graph_size=bs,
            ) as context:
                run = partial(self._full_graph_pass, context, bs * q)
                run()  # warmup, which also plans FlashInfer
                # Clear the MLA schedule created during warmup so capture allocates it in the graph's pool.
                # Otherwise the graph could keep pointers to memory freed when this context is released.
                context.mla_decode_metadata = None
                self.graphs[bs] = self._capture(run, self.graph_pool)

    def _full_graph_pass(self, context: Context, num_tokens: int) -> None:
        """Run the model on graph buffers using the same context for warmup and capture.

        Reusing the context lets capture find the backend plans created during warmup.
        """
        graph_vars = self.graph_vars
        with use_context(context):
            input_ids, positions = graph_vars["input_ids"][:num_tokens], graph_vars["positions"][:num_tokens]
            graph_vars["outputs"][:num_tokens] = self.model(input_ids, positions)

    @staticmethod
    def _capture(run: Callable[[], None], pool) -> torch.cuda.CUDAGraph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool):
            run()
        torch.cuda.synchronize()
        return graph

    @torch.inference_mode()
    def capture_piecewise(self):
        """Run each bucket from largest to smallest, capturing each piece's graph for that bucket."""
        self.piecewise_bs = cudagraph_capture_sizes(self.config.max_num_seqs, self.config.max_num_batched_tokens)
        largest = self.piecewise_bs[-1]
        input_ids = torch.zeros(largest, dtype=torch.int64)
        positions = torch.zeros(largest, dtype=torch.int64)
        self.piecewise_vars = dict(input_ids=input_ids, positions=positions)
        for size in reversed(self.piecewise_bs):
            # Run attention eagerly between graph pieces, using a fresh prompt with cache writes disabled.
            seq = Sequence([0] * size)
            seq.num_scheduled_tokens = size
            _, _, _, context = self.prepare_batch([seq])
            context["slot_mapping"] = torch.full((size,), -1, dtype=torch.int32)
            with set_context(**context, piecewise_size=size):
                self.model(input_ids[:size], positions[:size])
                if self.proposer is not None:  # the drafter's first pass uses the target's batch and bucket sizes
                    self.proposer.capture_piecewise(size)
            torch.cuda.synchronize()
