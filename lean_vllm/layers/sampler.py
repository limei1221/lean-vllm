import torch
from torch import nn


def apply_top_k_top_p(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    """Mask each row's logits with top-k filtering, then top-p filtering.

    Top-p uses probabilities computed after top-k filtering. A k of 0 disables
    top-k filtering, and a p of 1 disables top-p filtering. Pass None for both
    filters to skip sorting the vocabulary. Update logits in place and return
    the same tensor.
    """
    if k is None and p is None:
        return logits
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        vocab_size = logits.size(1)
        k = torch.where(k > 0, k, vocab_size).clamp_max(vocab_size)
        # The k-th largest value is the cutoff; ties with it stay in.
        cutoff = logits_sort.gather(1, (vocab_size - k.long()).unsqueeze(1))
        logits_sort.masked_fill_(logits_sort < cutoff, -float("inf"))
    if p is not None:
        # Remove the least likely tokens whose cumulative probability is at most 1 - p.
        # Always keep the most likely token.
        probs_sum = logits_sort.softmax(dim=-1, dtype=torch.float32).cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


class Sampler(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
    ):
        if temperatures is None:
            return logits.argmax(dim=-1)
        return self.sample(logits, temperatures, top_k, top_p)

    @torch.compile(dynamic=True)  # Allow different batch sizes during warmup and serving without recompiling.
    def sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
    ):
        logits = logits.float()
        greedy_tokens = logits.argmax(dim=-1)
        logits = logits.div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        logits = apply_top_k_top_p(logits, top_k, top_p)
        probs = torch.softmax(logits, dim=-1)
        # Exponential racing trick: argmax(probs / E), with independent E ~ Exp(1),
        # samples one token per row according to probs (equivalent to the Gumbel-max trick).
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
