import torch
from torch import nn

from lean_vllm.layers.sampler import apply_top_k_top_p
from lean_vllm.spec_decode.config import PLACEHOLDER_TOKEN_ID


class RejectionSampler(nn.Module):
    """Verify each row's draft tokens using rejection sampling (Leviathan et al. 2023).

    MTP chooses drafts greedily, so the draft distribution assigns probability 1 to the chosen token.
    For a sampling row, accept each draft with its target probability. Replace the first rejected draft
    by sampling from the target distribution with that draft token excluded, and discard later drafts.
    For a greedy row, accept drafts that match the target's argmax and replace the first mismatch with
    the target's argmax. A row that accepts every draft adds a token from its bonus position.
    """

    def forward(
        self,
        logits: torch.Tensor,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: torch.Tensor,
        temperatures: torch.Tensor | None,
        max_num_drafts: int,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return tokens with shape [rows, max_num_drafts + 1], padded with PLACEHOLDER_TOKEN_ID.

        logits has shape [sum(num_draft_tokens + 1), vocab]. Each row's draft logits are followed by its
        bonus logits. draft_token_ids contains the draft tokens flattened in the same row order;
        num_draft_tokens gives the draft count for each row.
        temperatures, top_k and top_p each have one value per row when provided. None for temperatures
        makes all rows greedy; a zero temperature makes that row greedy. Sampling probabilities use
        temperature scaling, then top-k and top-p filtering.
        """
        num_rows, num_drafts = num_draft_tokens.numel(), draft_token_ids.numel()
        device = logits.device
        rows = torch.arange(num_rows, device=device)
        num_logits = num_draft_tokens + 1
        first = torch.cumsum(num_logits, 0) - num_logits  # start of each row's logits in the flattened tensor
        # Map each draft to its batch row, position within that row, and index in the flattened logits.
        # Passing output_size avoids synchronizing with the host to determine the repeated tensor's size.
        draft_rows = torch.repeat_interleave(rows, num_draft_tokens, output_size=num_drafts)
        draft_cols = (
            torch.arange(num_drafts, device=device) - (torch.cumsum(num_draft_tokens, 0) - num_draft_tokens)[draft_rows]
        )
        draft_index = first[draft_rows] + draft_cols

        greedy = logits.argmax(dim=-1)
        if temperatures is None:
            accepted = greedy[draft_index] == draft_token_ids
            candidates = greedy
        else:
            row_temperatures = torch.repeat_interleave(temperatures, num_logits, output_size=logits.size(0))
            scaled = logits.float() / row_temperatures.clamp_min(1e-10).unsqueeze(1)
            if top_k is not None:
                top_k = torch.repeat_interleave(top_k, num_logits, output_size=logits.size(0))
            if top_p is not None:
                top_p = torch.repeat_interleave(top_p, num_logits, output_size=logits.size(0))
            probs = torch.softmax(apply_top_k_top_p(scaled, top_k, top_p), dim=-1)
            draft_probs = probs[draft_index, draft_token_ids]
            is_greedy = temperatures[draft_rows] == 0
            accepted = torch.where(
                is_greedy, greedy[draft_index] == draft_token_ids, torch.rand_like(draft_probs) < draft_probs
            )
            # Exclude the draft token at every draft position; leave bonus probabilities unchanged.
            # Only the candidate at the first rejection or, if all drafts pass, the bonus position is used.
            probs[draft_index, draft_token_ids] = 0
            sampled = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
            candidates = torch.where(row_temperatures == 0, greedy, sampled)

        # Count consecutive accepted drafts from the start of each row. Unused columns stay zero,
        # including the final column, so the count cannot exceed the row's draft count.
        keep = torch.zeros(num_rows, max_num_drafts + 1, dtype=torch.int32, device=device)
        keep[draft_rows, draft_cols] = accepted.int()
        num_accepted = keep.cumprod(dim=1).sum(dim=1)
        output = torch.full((num_rows, max_num_drafts + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int64, device=device)
        output[draft_rows, draft_cols] = draft_token_ids
        output.masked_fill_(
            torch.arange(max_num_drafts + 1, device=device) >= num_accepted.unsqueeze(1), PLACEHOLDER_TOKEN_ID
        )
        output[rows, num_accepted] = candidates[first + num_accepted]
        return output
