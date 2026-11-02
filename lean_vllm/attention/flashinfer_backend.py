from typing import Any

import torch

from lean_vllm.attention import triton_cache
from lean_vllm.attention.abstract import AttentionBackend
from lean_vllm.utils.context import Context
from lean_vllm.utils.device import get_device

_IMPORT_ERROR: ImportError | None = None
try:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
except ImportError as e:  # the cuda extra installs it on Linux
    _IMPORT_ERROR = e

# Workspace size for split-KV partial results; all wrappers share one buffer.
WORKSPACE_BYTES = 256 * 1024 * 1024


class FlashInferBackend(AttentionBackend):
    """Run FlashInfer prefill and decode with shared plans and Triton KV-cache writes on sm80 and newer GPUs."""

    supported_kinds = ("decoder",)
    _workspace: torch.Tensor | None = None
    _wrappers: dict[tuple, Any] = {}  # keyed by attention configuration and graph size; one current plan per wrapper
    _ragged: dict[tuple, list] = {}  # keyed by layer configuration: [wrapper, cumulative lengths used by its plan]
    _graph_pages: tuple[torch.Tensor, ...] | None = None  # shared page-table buffers, sized for the largest full graph

    @staticmethod
    def get_name() -> str:
        return "flashinfer"

    @staticmethod
    def is_available() -> bool:
        return (
            _IMPORT_ERROR is None
            and triton_cache._IMPORT_ERROR is None
            and get_device().type == "cuda"
            and torch.cuda.get_device_capability()[0] >= 8
        )

    @staticmethod
    def supports_cuda_graph() -> bool:
        return True  # full graphs re-plan decode before each replay; piecewise graphs run attention eagerly

    @staticmethod
    def split_decodes() -> bool:
        return True

    @staticmethod
    def supports_head_size(head_size: int) -> bool:
        return head_size in (64, 128, 256)

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping) -> None:
        triton_cache.store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    def prefill(self, q, k, v, k_cache, v_cache, context: Context, out=None) -> torch.Tensor:
        if context.block_tables is None:  # without pages, k and v contain all keys and values (e.g. during warmup)
            return self._planned_ragged(q, k, context).run(q, k, v, out=out)
        layout, pages = _paged_kv(k_cache, v_cache)
        return self._planned("paged", q, k_cache.dtype, context, k_cache.size(1), layout).run(q, pages, out=out)

    def decode(self, q, k_cache, v_cache, context: Context, out=None) -> torch.Tensor:
        layout, pages = _paged_kv(k_cache, v_cache)
        return self._planned("decode", q, k_cache.dtype, context, k_cache.size(1), layout).run(q, pages, out=out)

    @classmethod
    def before_full_graph_replay(cls, context: Context, batch_size: int) -> None:
        """Re-plan the decode wrappers captured for batch_size using this step's sequence lengths and pages.

        Planning updates the fixed page-table buffers that the graph reads during replay.
        """
        pages = {}
        for key, wrapper in cls._wrappers.items():
            if key[0] == "decode" and key[-1] == batch_size:
                page_size = key[7]
                if page_size not in pages:
                    pages[page_size] = cls._pages(context, page_size, batch_size)
                _plan_decode(wrapper, key, pages[page_size])

    def _planned_ragged(self, q: torch.Tensor, k: torch.Tensor, context: Context):
        """Return a cached causal ragged wrapper, re-planning when the cumulative query or key lengths change."""
        cu_q = _host_cumulative(context.cu_seqlens_q_host, context.cu_seqlens_q)
        cu_k = _host_cumulative(context.cu_seqlens_k_host, context.cu_seqlens_k)
        shape = (self.num_heads, self.num_kv_heads, self.head_dim, self.scale, q.dtype, k.dtype)
        entry = FlashInferBackend._ragged.get(shape)
        if entry is None:
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(self._workspace_for(q.device), "NHD")
            entry = FlashInferBackend._ragged[shape] = [wrapper, None]
        lengths = (cu_q.tolist(), cu_k.tolist())
        if entry[1] != lengths:
            entry[0].plan(
                cu_q,
                cu_k,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.head_dim,
                causal=True,
                sm_scale=self.scale,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
            )
            entry[1] = lengths
        return entry[0]

    def _planned(
        self, kind: str, q: torch.Tensor, kv_dtype: torch.dtype, context: Context, page_size: int, layout: str
    ):
        """Return the wrapper for kind, planning once per context and sharing it across layers with the same key."""
        graph_size = context.full_graph_size if kind == "decode" else None
        key = (
            kind,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.scale,
            q.dtype,
            kv_dtype,
            page_size,
            layout,
            graph_size,
        )
        if key in context.attn_metadata:
            return context.attn_metadata[key]
        wrapper = FlashInferBackend._wrappers.get(key)
        if wrapper is None:
            wrapper = FlashInferBackend._wrappers[key] = self._make_wrapper(kind, q.device, context, layout, graph_size)
        shape = (self.num_heads, self.num_kv_heads, self.head_dim)
        options = dict(sm_scale=self.scale, q_data_type=q.dtype, kv_data_type=kv_dtype)
        if kind == "decode":
            _plan_decode(wrapper, key, self._pages(context, page_size, graph_size))
        else:  # FlashInfer's causal mask is bottom-right aligned, as the contract asks
            qo_indptr = _host_cumulative(context.cu_seqlens_q_host, context.cu_seqlens_q)
            indptr, indices, last_page_len = self._pages(context, page_size)
            wrapper.plan(qo_indptr, indptr, indices, last_page_len, *shape, page_size, causal=True, **options)
        context.attn_metadata[key] = wrapper
        return wrapper

    @staticmethod
    def _workspace_for(device: torch.device) -> torch.Tensor:
        if FlashInferBackend._workspace is None:
            # Zeroed, as FlashInfer requires on its first use.
            FlashInferBackend._workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        return FlashInferBackend._workspace

    def _make_wrapper(self, kind: str, device: torch.device, context: Context, layout: str, graph_size: int | None):
        workspace = self._workspace_for(device)
        if kind == "paged":  # prefill with paged KV
            return BatchPrefillWithPagedKVCacheWrapper(workspace, layout)
        # Use tensor cores for more than 4 query heads per KV head; the CUDA-core kernel supports few group sizes.
        use_tensor_cores = self.num_heads // self.num_kv_heads > 4
        if graph_size is None:
            return BatchDecodeWithPagedKVCacheWrapper(workspace, layout, use_tensor_cores=use_tensor_cores)
        # Each captured batch size gets a wrapper backed by the shared page-table buffers.
        if FlashInferBackend._graph_pages is None:
            assert context.block_tables is not None
            # The largest graph is captured first, so these buffers fit all captured batch sizes.
            rows, width = context.block_tables.shape
            FlashInferBackend._graph_pages = (
                torch.zeros(rows + 1, dtype=torch.int32, device=device),
                torch.zeros(rows * width, dtype=torch.int32, device=device),
                torch.zeros(rows, dtype=torch.int32, device=device),
            )
        indptr, indices, last_page_len = FlashInferBackend._graph_pages
        assert graph_size <= last_page_len.numel(), "full graphs must capture their largest batch size first"
        return BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            layout,
            use_cuda_graph=True,
            use_tensor_cores=use_tensor_cores,
            paged_kv_indptr_buffer=indptr[: graph_size + 1],
            paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=last_page_len[:graph_size],
        )

    @staticmethod
    def _pages(context: Context, page_size: int, num_rows: int | None = None) -> tuple[torch.Tensor, ...]:
        """Return page offsets (indptr), packed page indices, and the number of tokens in each row's last page.

        Offsets and last-page lengths are on the host; indices are on the same device as context.block_tables.
        When num_rows is given, pad to that many rows with empty page ranges for graph replay.
        """
        cu_k = _host_cumulative(context.cu_seqlens_k_host, context.cu_seqlens_k, context.context_lens)
        kv_lens = cu_k[1:] - cu_k[:-1]
        num_pages = (kv_lens + page_size - 1) // page_size
        indptr = torch.zeros_like(cu_k)
        indptr[1:] = num_pages.cumsum(0)
        rows = torch.repeat_interleave(num_pages)
        pages = torch.arange(rows.numel()) - indptr[rows]
        block_tables = context.block_tables
        assert block_tables is not None
        flat = (rows * block_tables.size(1) + pages).to(block_tables.device, non_blocking=True)
        indices = block_tables.flatten()[flat]
        last_page_len = kv_lens - (num_pages - 1) * page_size
        if num_rows is not None:  # padded rows have no page indices and use 1 as a placeholder last-page length
            pad = num_rows - kv_lens.numel()
            indptr = torch.cat([indptr, indptr[-1:].expand(pad)])
            last_page_len = torch.cat([last_page_len, last_page_len.new_ones(pad)])
        return indptr, indices, last_page_len


def _plan_decode(wrapper, key: tuple, pages: tuple[torch.Tensor, ...]) -> None:
    """Plan a decode wrapper using its cached configuration and page table, without needing a layer instance."""
    _, num_heads, num_kv_heads, head_dim, scale, q_dtype, kv_dtype, page_size, _, _ = key
    wrapper.plan(
        *pages, num_heads, num_kv_heads, head_dim, page_size, sm_scale=scale, q_data_type=q_dtype, kv_data_type=kv_dtype
    )


def _paged_kv(k_cache: torch.Tensor, v_cache: torch.Tensor) -> tuple[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return the layout name and cache views FlashInfer expects, restoring HND axis order when needed."""
    if k_cache.stride(1) < k_cache.stride(2):  # HND storage: tokens have a smaller stride than heads
        return "HND", (k_cache.transpose(1, 2), v_cache.transpose(1, 2))
    return "NHD", (k_cache, v_cache)


def _host_cumulative(host: list[int] | None, device: torch.Tensor | None, lens: torch.Tensor | None = None):
    """Return cumulative lengths as a CPU int32 tensor, using host values when available.

    Otherwise, copy device values to the host, first computing cumulative lengths from lens if needed.
    """
    if host is None:
        if device is None:  # cumulative lengths are missing; use the per-sequence lengths
            assert lens is not None
            device = torch.nn.functional.pad(lens.cumsum(0), (1, 0))
        host = device.tolist()
    return torch.tensor(host, dtype=torch.int32)
