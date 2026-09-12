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

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, keys_are_new=False, slot_mapping=None, context_lens=None, block_tables=None, logits_indices=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, keys_are_new, slot_mapping, context_lens, block_tables, logits_indices)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
