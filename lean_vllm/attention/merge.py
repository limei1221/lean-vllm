"""Combine attention outputs and log-sum-exp values from two non-overlapping key sets."""

import torch

from lean_vllm.attention import triton_merge


def merge_attention(o_a, lse_a, o_b, lse_b) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the combined attention output and log-sum-exp for two non-overlapping key sets."""
    # Both partial outputs use the same queries; only the keys and values are split.
    # Weight output b by its share of the total softmax denominator:
    # weight_b = exp(lse_b) / (exp(lse_a) + exp(lse_b)).
    weight_b = torch.sigmoid(lse_b - lse_a).unsqueeze(-1)  # [num_tokens, num_heads, 1]
    # Blend the outputs in float32: o = (1 - weight_b) * o_a + weight_b * o_b.
    o = torch.lerp(o_a.float(), o_b.float(), weight_b)  # [num_tokens, num_heads, head_dim]
    return o.to(o_a.dtype), torch.logaddexp(lse_a, lse_b)


def merge_attention_(o_a, lse_a, o_b, lse_b, out=None) -> None:
    """Write the merged output to out (or o_a if out is None) and the merged log-sum-exp to lse_a.

    Use one Triton kernel launch when o_a is on CUDA and Triton is available; otherwise use PyTorch.
    """
    if o_a.is_cuda and triton_merge._IMPORT_ERROR is None:
        triton_merge.merge_attn_states_(o_a, lse_a, o_b, lse_b, out)
        return
    o, lse = merge_attention(o_a, lse_a, o_b, lse_b)
    (o_a if out is None else out).copy_(o)
    lse_a.copy_(lse)
