import torch
import torch.nn.functional as F

from lean_vllm.attention import triton_cache
from lean_vllm.attention.abstract import AttentionBackend, write_into
from lean_vllm.utils.context import Context
from lean_vllm.utils.device import get_device

_IMPORT_ERROR: ImportError | None = None
try:
    from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache, get_scheduler_metadata
except ImportError as e:  # Hopper only, and built by the cuda extra
    _IMPORT_ERROR = e


class FlashAttention3Backend(AttentionBackend):
    """FlashAttention-3 kernels with a Triton KV-cache scatter. Hopper only.

    Paged calls take a tile schedule made once per step (vLLM's AOT schedule).
    """

    _graph_problems: dict[tuple, dict] = {}  # a full graph's decode schedule, by key: how to remake it before replay
    _graph_schedules: dict[tuple, torch.Tensor] = {}  # what full graphs read their schedule from, per layer shape

    @staticmethod
    def get_name() -> str:
        return "flash_attn_3"

    @staticmethod
    def is_available() -> bool:
        # FA3 is built for Hopper (sm90) only.
        return (
            _IMPORT_ERROR is None
            and triton_cache._IMPORT_ERROR is None
            and get_device().type == "cuda"
            and torch.cuda.get_device_capability()[0] == 9
        )

    @staticmethod
    def supports_cuda_graph() -> bool:
        return True

    @staticmethod
    def supports_head_size(head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping) -> None:
        triton_cache.store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    def store_latents(self, latent, latent_cache, slot_mapping) -> None:
        triton_cache.store_latents(latent, latent_cache, slot_mapping)

    def prefill(self, q, k, v, k_cache, v_cache, context: Context, out=None) -> torch.Tensor:
        # FA3's entry points take no out, so one given is copied into.
        if context.block_tables is None:
            # No pages (warmup, or MLA's new tokens): k and v hold every key.
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
            return write_into(out, o)
        # Keys come from the pages, cold rows too; FA3's varlen entry takes no page table.
        assert context.cu_seqlens_k is not None
        cache_seqlens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
        o = self._paged(
            "prefill",
            q,
            k_cache,
            v_cache,
            context,
            cache_seqlens,
            context.block_tables,
            context.cu_seqlens_q,
            context.max_seqlen_q,
        )
        return write_into(out, o)

    def varlen_with_lse(
        self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, host_cu_seqlens=None
    ):
        o, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=causal,
            return_attn_probs=True,
        )
        return o, lse.transpose(0, 1)  # FA3's varlen lse is [heads, tokens]

    def decode(self, q, k_cache, v_cache, context: Context, out=None) -> torch.Tensor:
        o = self._paged("decode", q.unsqueeze(1), k_cache, v_cache, context, context.context_lens, context.block_tables)
        return write_into(out, o.squeeze(1))  # match the [batch, heads, dim] contract

    def _paged(
        self,
        kind: str,
        q,
        k_cache,
        v_cache,
        context: Context,
        cache_seqlens,
        page_table,
        cu_seqlens_q=None,
        max_seqlen_q=None,
    ):
        """FA3 over the pages, q batched or packed by cu_seqlens_q, on the step's schedule for this problem."""
        schedule = self._schedule(
            kind, q, k_cache, v_cache, context, cache_seqlens, page_table, cu_seqlens_q, max_seqlen_q
        )
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=self.scale,
            causal=True,
            scheduler_metadata=schedule,
        )

    def _schedule(
        self,
        kind: str,
        q,
        k_cache,
        v_cache,
        context: Context,
        cache_seqlens,
        page_table,
        cu_seqlens_q,
        max_seqlen_q,
    ) -> torch.Tensor:
        """FA3's tile schedule for one problem, made by the step's first layer to pose it so the rest skip the
        in-kernel pass. A full graph reads it from a buffer refilled before each replay."""
        page_size = k_cache.size(1)
        shape = (self.num_heads, self.num_kv_heads, self.head_dim, v_cache.size(-1), q.dtype, page_size)
        key = ("fa3", kind, *shape, context.full_graph_size)  # the step cache holds other backends' plans too
        cache = context.attn_metadata
        if key in cache:
            return cache[key]
        varlen = cu_seqlens_q is not None
        # Every argument the kernel derives its split count from must match, or it reads a schedule for another.
        problem = dict(
            batch_size=cu_seqlens_q.numel() - 1 if varlen else q.size(0),
            max_seqlen_q=max_seqlen_q if varlen else q.size(1),
            max_seqlen_k=page_table.size(1) * page_size,  # the kernel takes it from the page table's width
            num_heads_q=self.num_heads,
            num_heads_kv=self.num_kv_heads,
            headdim=self.head_dim,
            qkv_dtype=q.dtype,
            headdim_v=v_cache.size(-1),
            cu_seqlens_q=cu_seqlens_q,
            page_size=page_size,
            causal=True,
        )
        schedule = get_scheduler_metadata(cache_seqlens=cache_seqlens, **problem)
        if context.full_graph_size is not None:  # the capture's warmup; the capture finds it here
            assert not varlen, "a full graph holds decode only"
            FlashAttention3Backend._graph_problems[key] = problem
            schedule = _into_graph_schedule(key, schedule)
        cache[key] = schedule
        return schedule

    @classmethod
    def before_full_graph_replay(cls, context: Context, batch_size: int) -> None:
        """Remake the schedule each full graph at batch_size reads, for this step's lengths
        before every replay."""
        lens = None
        for key, problem in FlashAttention3Backend._graph_problems.items():
            if key[-1] != batch_size:
                continue
            if lens is None:  # the graph's rows past the step's hold no keys
                assert context.context_lens is not None
                lens = F.pad(context.context_lens, (0, batch_size - context.context_lens.numel()))
            _into_graph_schedule(key, get_scheduler_metadata(cache_seqlens=lens, **problem))


def _into_graph_schedule(key: tuple, schedule: torch.Tensor) -> torch.Tensor:
    """schedule copied into the buffer full graphs read, its tail zeroed, since a stale tail misdirects
    thread blocks. One buffer per layer shape, sized by the first capture, which is the largest batch."""
    buffers = FlashAttention3Backend._graph_schedules
    shape = key[:-1]
    if shape not in buffers:
        buffers[shape] = torch.zeros_like(schedule)
    buffer, n = buffers[shape], schedule.numel()
    assert n <= buffer.numel(), "full graphs must capture their largest batch size first"
    buffer[:n] = schedule
    buffer[n:] = 0
    return buffer[:n]
