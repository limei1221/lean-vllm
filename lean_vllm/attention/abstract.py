from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from lean_vllm.utils.context import Context, split_decodes_and_prefills


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """Describe one attention layer's requirements so the selector can choose a compatible backend."""

    head_size: int  # query/key head width; for MLA, the expanded head width used during prefill
    num_heads: int
    num_kv_heads: int
    dtype: torch.dtype
    latent_dim: int = 0  # an MLA layer's cached latent width; 0 for a layer that caches keys and values

    @property
    def kind(self) -> str:
        return "mla" if self.latent_dim else "decoder"


def write_into(out: torch.Tensor | None, o: torch.Tensor) -> torch.Tensor:
    """Return o, or copy it into out and return out when an output buffer is provided."""
    return o if out is None else out.copy_(o)


class AttentionBackend(ABC):
    """Execution strategy for one attention layer. See docs/attention-backends.md."""

    # Supported dtypes and layer kinds, checked by validate.
    supported_dtypes: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)
    supported_kinds: tuple[str, ...] = ("decoder", "mla")

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Identifier matching LEAN_VLLM_ATTENTION_BACKEND."""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Whether this backend can run on the current machine and build."""

    @staticmethod
    def supports_cuda_graph() -> bool:
        """Return False to disable compilation and CUDA graphs for models using this backend."""
        return False

    @staticmethod
    def supports_full_cudagraph() -> bool:
        """Return whether decode supports full CUDA graph capture and replay."""
        return True

    @staticmethod
    def split_decodes() -> bool:
        """Return whether forward splits mixed batches into decode and prefill groups.

        When True, leading rows with one query use decode and the remaining rows use prefill.
        When False, the entire batch uses prefill if context.is_prefill is True.
        """
        return False

    @staticmethod
    def supports_head_size(head_size: int) -> bool:
        return True

    @classmethod
    def validate(cls, spec: LayerSpec) -> list[str]:
        """Return reasons the layer's configuration is unsupported, or an empty list if it is supported."""
        reasons = []
        if spec.dtype not in cls.supported_dtypes:
            reasons.append(f"dtype {spec.dtype} is not one of {list(cls.supported_dtypes)}")
        if not cls.supports_head_size(spec.head_size):
            reasons.append(f"head size {spec.head_size} is not supported")
        if spec.num_heads % spec.num_kv_heads:
            reasons.append(f"{spec.num_heads} query heads do not group over {spec.num_kv_heads} key heads")
        if spec.kind not in cls.supported_kinds:
            reasons.append(f"{spec.kind} layers are not supported")
        return reasons

    @staticmethod
    def supports_mla_decode() -> bool:
        """Return whether mla_decode attends directly to cached latents without expanding keys and values."""
        return False

    @staticmethod
    def supports_full_cudagraph_mla_decode() -> bool:
        """Return whether mla_decode supports full CUDA graph replay.

        Return False if replay cannot refresh state that depends on the current step, such as a host-built schedule.
        MLA decode then runs outside captured graph pieces.
        """
        return True

    @staticmethod
    def mla_block_size() -> int | None:
        """Return the required cache page size in tokens, or None if this backend imposes no page-size restriction."""
        return None

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int) -> tuple[int, ...]:
        """Per-layer shape of one of the key/value cache tensors."""
        return (num_blocks, block_size, num_kv_heads, head_dim)

    @abstractmethod
    def store_kvcache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write new keys and values into the paged cache in place, skipping tokens whose slot_mapping entry is -1."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one attention step after its keys and values are cached, writing into out if provided.

        Pure decode batches use decode. If split_decodes is True, mixed batches send leading one-query rows
        to decode and the rest to prefill. Otherwise, the entire mixed batch uses prefill.
        """
        if not context.is_prefill:
            return self.decode(q, k_cache, v_cache, context, out=out)
        if not self.split_decodes():
            return self.prefill(q, k, v, k_cache, v_cache, context, out=out)
        n, decodes, prefills = split_decodes_and_prefills(context)
        if decodes is None:
            assert prefills is not None
            return self.prefill(q, k, v, k_cache, v_cache, prefills, out=out)
        if prefills is None:
            return self.decode(q, k_cache, v_cache, decodes, out=out)
        # Both groups write into slices of the same output tensor, so their results need no concatenation.
        if out is None:
            out = torch.empty_like(q)
        self.decode(q[:n], k_cache, v_cache, decodes, out=out[:n])
        self.prefill(q[n:], k[n:], v[n:], k_cache, v_cache, prefills, out=out[n:])
        return out

    @abstractmethod
    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run causal attention over packed variable-length sequences with a bottom-right aligned mask.

        q is [num_tokens, num_heads, head_dim] and holds only the query tokens routed to prefill.
        With context.block_tables set, the caller has cached this step's k and v; attention reads all keys
        and values from the cache. Otherwise, k and v contain all keys and values needed for attention.
        Write into out and return it if provided; otherwise, return a new output tensor.
        """

    @abstractmethod
    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend to the paged cache with one query per sequence; q is [batch_size, num_heads, head_dim].

        Write into out if provided and return the output tensor.
        """

    def store_latents(
        self,
        latent: torch.Tensor,
        latent_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write latents [num_tokens, latent_dim] into the paged cache in place.

        Skip tokens whose slot_mapping entry is -1.
        """
        raise NotImplementedError(f"the {self.get_name()} backend stores no MLA latents")

    def varlen_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        host_cu_seqlens: tuple[list[int], list[int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention output and log-sum-exp [num_tokens, num_heads] using the supplied keys and values.

        When causal is True, use a bottom-right aligned mask. MLA layers use the log-sum-exp to merge outputs
        from current tokens and chunks of expanded cached latents. host_cu_seqlens contains host copies of
        cu_seqlens_q and cu_seqlens_k so backends can plan without copying these lengths from the device.
        """
        raise NotImplementedError(f"the {self.get_name()} backend has no varlen_with_lse, so serves no MLA layer")

    def mla_decode(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        v_dim: int,
        context: Context,
    ) -> torch.Tensor:
        """Attend to a paged MLA latent cache with one key head shared by all query heads.

        Each sequence has the same number of queries, corresponding to its last tokens, with a causal mask:
        one query during ordinary decode, or 1 + draft tokens during verification.
        A prompt chunk of that length also qualifies.
        q is [batch_size * queries, num_heads, latent_dim] and latent_cache is [num_blocks, block_size, latent_dim].
        Values are the first v_dim entries of each latent. Return [batch_size * queries, num_heads, v_dim].
        """
        raise NotImplementedError(f"the {self.get_name()} backend has no MLA decode")

    @classmethod
    def before_full_graph_replay(cls, context: Context, batch_size: int) -> None:
        """Refresh backend state before replaying the full graph captured for batch_size.

        Use this step's pure-decode context to update state the graph reads but cannot recompute during replay.
        """
