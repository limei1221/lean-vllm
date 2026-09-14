from contextlib import contextmanager
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    keys_are_new: bool = False    # no row carries cached keys, so k/v holds the whole batch
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None    # rows that sample; None means all of them
    num_keys: int = 0    # cu_seqlens_k[-1], kept on the host so gathers can size without a sync
    key_slots: torch.Tensor | None = None    # filled on first use by layers.attention.key_slots

_CONTEXT = Context()

def get_context():
    return _CONTEXT

@contextmanager
def set_context(is_prefill: bool, **kwargs):
    """The context for one forward pass; the previous one comes back on exit, even on error."""
    global _CONTEXT
    previous, _CONTEXT = _CONTEXT, Context(is_prefill, **kwargs)
    try:
        yield _CONTEXT
    finally:
        _CONTEXT = previous

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
