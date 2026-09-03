from abc import ABC, abstractmethod

import torch

from inferweave.utils.context import Context


class AttentionBackend(ABC):
    """Execution strategy for one attention layer. See docs/attention-backends.md."""

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Identifier matching INFERWEAVE_ATTENTION_BACKEND."""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Whether this backend can run on the current machine and build."""

    @staticmethod
    def supports_cuda_graph() -> bool:
        """False if decode branches on tensor values, so cannot be captured."""
        return False

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
        """Scatter new keys/values into the paged cache in place. Slot -1 skips."""

    @abstractmethod
    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Causal attention over packed varlen sequences, bottom-right aligned.

        q is [num_tokens, num_heads, head_dim]; k and v hold new tokens only.
        Keys come from the paged cache when context.block_tables is set.
        """

    @abstractmethod
    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Single-query attention against the paged cache. q is [batch, heads, dim]."""
