import logging
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from lean_vllm.attention import get_attention_backend
from lean_vllm.config import Config
from lean_vllm.engine.sequence import Sequence
from lean_vllm.models.qwen3 import Qwen3ForCausalLM
from lean_vllm.layers.sampler import Sampler
from lean_vllm.utils.context import set_context, get_context, reset_context
from lean_vllm.utils.loader import load_model
from lean_vllm.utils import device as dev

logger = logging.getLogger(__name__)


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
        self.eager_reason: str | None = "enforced"    # why the last step ran eager; None if it replayed a graph
        self.enforce_eager = (config.enforce_eager or self.device.type != "cuda"
                              or not attention_backend.supports_cuda_graph())
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(dev.dist_backend(self.device), "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        dev.set_device(self.device, rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device(self.device)
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
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
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
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

    def _eager_reason(self, is_prefill: bool, batch_size: int) -> str | None:
        """Why this step cannot replay a graph. None means it can.

        "prefill" is what piecewise capture would reclaim, "oversized" what a
        larger bucket would; the split decides whether either is worth doing.
        """
        if self.enforce_eager:
            return "enforced"
        if is_prefill:
            return "prefill"
        if batch_size > 512:
            return "oversized"
        return None

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        self.eager_reason = self._eager_reason(is_prefill, input_ids.size(0))
        if self.eager_reason:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
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
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence]) -> list[int]:
        input_ids, positions, temperatures, is_prefill = self.prepare_batch(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

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
        self.graphs = {}
        self.graph_pool = None

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
