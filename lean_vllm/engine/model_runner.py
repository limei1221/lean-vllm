import logging
import pickle
import torch
import torch.distributed as dist
from torch.profiler import record_function
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from lean_vllm.attention import get_attention_backend
from lean_vllm.config import Config, FULL_MODES, PIECEWISE_MODES
from lean_vllm.engine.sequence import Sequence
from lean_vllm.models.qwen3 import Qwen3ForCausalLM
from lean_vllm.layers.attention import register_layers
from lean_vllm.layers.sampler import Sampler
from lean_vllm.utils.context import set_context, get_context, reset_context
from lean_vllm.utils.loader import load_model
from lean_vllm.utils import device as dev

logger = logging.getLogger(__name__)

# Piecewise buckets: the range of step sizes worth capturing, the smallest gap
# between buckets, and the most padding a replay may add over the step it serves.
PIECEWISE_MIN_TOKENS = 64
PIECEWISE_MAX_TOKENS = 512
PIECEWISE_MIN_GAP = 16
PIECEWISE_MAX_PAD = 0.25


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        Sequence.block_size = self.block_size    # spawned workers never run LLMEngine.__init__
        self.device = dev.get_device()
        attention_backend = get_attention_backend()
        if rank == 0:
            logger.info("attention backend: %s", attention_backend.get_name())
        self.step_kind = "enforced"    # how the last step ran: see _step_kind
        self.enforce_eager = (config.enforce_eager or self.device.type != "cuda"
                              or not attention_backend.supports_cuda_graph())
        self.cudagraph_mode = "none" if self.enforce_eager else config.cudagraph_mode
        self.graph_bs: list[int] = []          # captured batch sizes, full graphs
        self.piecewise_bs: list[int] = []      # captured token counts, piecewise graphs
        self.graphs: dict = {}
        self.piecewise_graphs: dict = {}
        self.graph_pool = None                 # shared by both capture kinds
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(dev.dist_backend(self.device), "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        dev.set_device(self.device, rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device(self.device)
        self.model = Qwen3ForCausalLM(hf_config)
        register_layers(self.model)    # before warmup_model, which runs the op
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if self.cudagraph_mode in FULL_MODES:
            self.capture_cudagraph()
        if self.cudagraph_mode in PIECEWISE_MODES:
            self.capture_piecewise()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="lean_vllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="lean_vllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if self.cudagraph_mode != "none":
            del self.graphs, self.piecewise_graphs, self.graph_pool
        dev.synchronize(self.device)
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

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
        dev.empty_cache(self.device)

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = dev.kvcache_bytes(self.device, config) // block_bytes
        assert config.num_kvcache_blocks > 0, "no memory left for the kv cache"
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = dev.make_tensor(block_tables, torch.int32, self.device)
        return block_tables

    def prepare_batch(self, seqs: list[Sequence]):
        """One batch for any mix of prompt chunks and decode rows."""
        input_ids, positions, slot_mapping = [], [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_seqlen_q = max_seqlen_k = 0
        context_lens, logits_indices, temperatures = [], [], []
        is_prefill = any(seq.is_prefill for seq in seqs)

        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            input_ids.extend(seq[start:end] if seq.is_prefill else [seq.last_token])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seq.num_scheduled_tokens)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(seq.num_scheduled_tokens, max_seqlen_q)
            max_seqlen_k = max(end, max_seqlen_k)
            context_lens.append(end)
            if end == seq.num_tokens:    # the prompt is complete, so this row samples
                logits_indices.append(cu_seqlens_q[-1] - 1)
                if self.rank == 0:    # only the sampling rank owns sampling parameters
                    temperatures.append(seq.temperature)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        block_tables = self.prepare_block_tables(seqs) if any(seq.block_table for seq in seqs) else None
        set_context(
            is_prefill,
            cu_seqlens_q=dev.make_tensor(cu_seqlens_q, torch.int32, self.device),
            cu_seqlens_k=dev.make_tensor(cu_seqlens_k, torch.int32, self.device),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            # Equal cumulative lengths mean no row started from cached tokens, which
            # is the batch a backend can attend without reading the cache back.
            keys_are_new=cu_seqlens_k == cu_seqlens_q,
            slot_mapping=dev.make_tensor(slot_mapping, torch.int32, self.device),
            context_lens=dev.make_tensor(context_lens, torch.int32, self.device),
            block_tables=block_tables,
            # A pure-decode batch samples on every row, so the gather is skipped.
            logits_indices=dev.make_tensor(logits_indices, torch.int64, self.device) if is_prefill else None,
        )
        input_ids = dev.make_tensor(input_ids, torch.int64, self.device)
        positions = dev.make_tensor(positions, torch.int64, self.device)
        all_greedy = all(temperature == 0 for temperature in temperatures)
        temperatures = None if all_greedy else dev.make_tensor(temperatures, torch.float32, self.device)
        return input_ids, positions, temperatures, is_prefill

    def _step_kind(self, is_prefill: bool, num_tokens: int) -> str:
        """How this step runs: "graph", "piecewise", or why it must run eager.

        A pure-decode batch has one token per row, so num_tokens is its batch
        size too. Each branch needs the graphs to exist, not just the mode: a
        mode naming a capture that never happened falls through to eager.
        """
        if self.cudagraph_mode == "none":
            return "enforced"
        if not is_prefill and self.cudagraph_mode in FULL_MODES and self.graph_bs:
            if num_tokens <= self.graph_bs[-1]:
                return "graph"
        if self.cudagraph_mode in PIECEWISE_MODES and self._piecewise_bucket(num_tokens):
            return "piecewise"
        # "prefill" covers the large steps the piecewise grid deliberately leaves
        # eager; "oversized" is a decode batch past the full-graph buckets.
        return "prefill" if is_prefill else "oversized"

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        self.step_kind = self._step_kind(is_prefill, input_ids.size(0))
        if self.step_kind == "graph":
            return self.model.compute_logits(self._replay_full(input_ids, positions))
        if self.step_kind == "piecewise":
            return self.model.compute_logits(self._replay_piecewise(input_ids, positions))
        return self.model.compute_logits(self.model(input_ids, positions))

    def _replay_full(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """One graph for the whole model. Pure decode only: attention is inside it."""
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return graph_vars["outputs"][:bs]

    def _piecewise_bucket(self, num_tokens: int) -> int | None:
        """The bucket a step of this size replays in, or None if there is none.

        Shared by the dispatch and the replay so they cannot disagree about which
        bucket -- or whether there is one -- for the same step.
        """
        bucket = next((size for size in self.piecewise_bs if size >= num_tokens), None)
        return None if bucket is None or num_tokens < self.piecewise_bs[0] else bucket

    def _replay_piecewise(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """A graph per piece, with attention run eager between them.

        Pad rows compute alongside the real ones and are dropped. Nothing in a
        piece mixes rows -- every op is per token -- so whatever the pad region
        holds cannot reach a real one, and attention is handed the real rows
        only, so no pad reaches the KV cache either.
        """
        num_tokens = input_ids.size(0)
        bucket = self._piecewise_bucket(num_tokens)
        graphs, buffers = self.piecewise_graphs[bucket], self.piecewise_vars
        buffers["input_ids"][:num_tokens] = input_ids
        buffers["positions"][:num_tokens] = positions

        graphs["head"].replay()
        for layer, pre, post in zip(self.model.model.layers, graphs["pre"], graphs["post"]):
            pre.replay()
            # The real rows only: attention reads this step's sequence layout,
            # which is exactly what cannot go in a graph.
            attn_out = layer.self_attn.attn(
                buffers["q"][:num_tokens], buffers["k"][:num_tokens], buffers["v"][:num_tokens]
            )
            buffers["attn_out"][:num_tokens] = attn_out
            post.replay()
        graphs["tail"].replay()
        return buffers["output"][:num_tokens]

    def run(self, seqs: list[Sequence]) -> list[int]:
        with record_function("prepare_batch"):
            input_ids, positions, temperatures, is_prefill = self.prepare_batch(seqs)
        with record_function("run_model"):
            logits = self.run_model(input_ids, positions, is_prefill)
        with record_function("sample"):
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def _piecewise_buckets(self) -> list[int]:
        """Token counts to capture at: small steps only, none padded past a quarter.

        Capture pays where launching the pieces costs as much as running them,
        which is small steps. Above the top bucket a step stays eager, as in vLLM.
        """
        top = min(PIECEWISE_MAX_TOKENS, self.config.max_num_batched_tokens)
        sizes, size = [], PIECEWISE_MIN_TOKENS
        while size < top:
            sizes.append(size)
            gap = max(int(size * PIECEWISE_MAX_PAD), PIECEWISE_MIN_GAP)
            size += 1 << (gap.bit_length() - 1)    # a power of two, so sizes stay round
        return sorted(set(sizes) | {top})

    @torch.inference_mode()
    def capture_piecewise(self):
        """Capture the model either side of attention, one graph per piece per bucket.

        The pieces read and write fixed buffers, so a replay always finds its
        inputs where the capture left them. They touch no context and no KV
        cache -- that is what makes them capturable while attention is not.
        """
        hf_config = self.config.hf_config
        layers = self.model.model.layers
        self.piecewise_bs = self._piecewise_buckets()
        largest = self.piecewise_bs[-1]
        attn = layers[0].self_attn
        buffers = dict(
            input_ids=torch.zeros(largest, dtype=torch.int64),
            positions=torch.zeros(largest, dtype=torch.int64),
            hidden=torch.zeros(largest, hf_config.hidden_size),
            residual=torch.zeros(largest, hf_config.hidden_size),
            output=torch.zeros(largest, hf_config.hidden_size),
            q=torch.zeros(largest, attn.num_heads, attn.head_dim),
            k=torch.zeros(largest, attn.num_kv_heads, attn.head_dim),
            v=torch.zeros(largest, attn.num_kv_heads, attn.head_dim),
            attn_out=torch.zeros(largest, attn.num_heads, attn.head_dim),
        )
        self.piecewise_vars = buffers
        self.piecewise_graphs = {}

        def capture(run):
            """Warm up, then capture. The warmup pass is what allocates."""
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.graph_pool):
                run()
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            torch.cuda.synchronize()
            return graph

        for size in reversed(self.piecewise_bs):
            def head(size=size):
                buffers["hidden"][:size].copy_(self.model.model.embed_tokens(buffers["input_ids"][:size]))

            def tail(size=size):
                normed, _ = self.model.model.norm(buffers["hidden"][:size], buffers["residual"][:size])
                buffers["output"][:size].copy_(normed)

            def pre(layer, first, size=size):
                # The first layer takes no residual in; its graph bakes that in.
                carried = None if first else buffers["residual"][:size]
                q, k, v, residual = layer.pre_attention(buffers["positions"][:size], buffers["hidden"][:size], carried)
                buffers["q"][:size].copy_(q)
                buffers["k"][:size].copy_(k)
                buffers["v"][:size].copy_(v)
                buffers["residual"][:size].copy_(residual)

            def post(layer, size=size):
                hidden, residual = layer.post_attention(buffers["attn_out"][:size], buffers["residual"][:size])
                buffers["hidden"][:size].copy_(hidden)
                buffers["residual"][:size].copy_(residual)

            self.piecewise_graphs[size] = {
                "head": capture(head),
                "pre": [capture(lambda layer=layer, first=i == 0: pre(layer, first)) for i, layer in enumerate(layers)],
                "post": [capture(lambda layer=layer: post(layer)) for layer in layers],
                "tail": capture(tail),
            }

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
