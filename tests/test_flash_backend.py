"""The vLLM FlashAttention API boundary, runnable without CUDA kernels."""

from unittest.mock import Mock

import pytest
import torch

from lean_vllm.attention import flash_backend
from lean_vllm.utils.context import Context


@pytest.fixture
def kernel(monkeypatch):
    kernel = Mock(side_effect=lambda q, *args, **kwargs: q.clone())
    monkeypatch.setattr(flash_backend, "flash_attn_varlen_func", kernel, raising=False)
    return kernel


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("context_lens", [None, torch.tensor([19, 5], dtype=torch.int32)])
def test_prefill_passes_lengths_in_the_form_required_by_vllm(kernel, paged, context_lens):
    backend = flash_backend.FlashAttentionBackend(4, 32, 0.137, 2)
    q, k, v = torch.randn(8, 4, 32), torch.randn(24, 2, 32), torch.randn(24, 2, 32)
    kc, vc = torch.randn(3, 16, 2, 32), torch.randn(3, 16, 2, 32)
    context = Context(
        cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 19, 24], dtype=torch.int32),
        max_seqlen_q=5, max_seqlen_k=19, context_lens=context_lens,
        block_tables=torch.tensor([[2, 0], [1, -1]], dtype=torch.int32) if paged else None,
    )
    out = backend.prefill(q, k, v, kc, vc, context)
    args, kwargs = kernel.call_args
    assert args[1] is (kc if paged else k)
    assert args[2] is (vc if paged else v)
    assert kwargs["block_table"] is context.block_tables
    if paged:
        assert kwargs.get("cu_seqlens_k") is None
        torch.testing.assert_close(kwargs["seqused_k"], torch.tensor([19, 5], dtype=torch.int32))
    else:
        assert kwargs.get("seqused_k") is None
        assert kwargs["cu_seqlens_k"] is context.cu_seqlens_k
    assert kwargs["softmax_scale"] == 0.137
    assert kwargs["causal"] is True
    assert kwargs["fa_version"] == 2
    torch.testing.assert_close(out, q)


def test_decode_uses_one_query_per_sequence_without_reading_device_lengths(kernel, monkeypatch):
    backend = flash_backend.FlashAttentionBackend(4, 32, 0.137, 2)
    q = torch.randn(3, 4, 32)
    kc, vc = torch.randn(4, 16, 2, 32), torch.randn(4, 16, 2, 32)
    context = Context(
        context_lens=torch.tensor([19, 7, 0], dtype=torch.int32),
        block_tables=torch.tensor([[2, 0], [1, -1], [0, 0]], dtype=torch.int32),
    )
    # Device-to-host scalar reads would break CUDA graph capture.
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", lambda self: pytest.fail("device scalar read"))
        out = backend.decode(q, kc, vc, context)
    args, kwargs = kernel.call_args
    assert args[0] is q
    assert args[1] is kc and args[2] is vc
    torch.testing.assert_close(kwargs["cu_seqlens_q"], torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    assert kwargs["seqused_k"] is context.context_lens
    assert kwargs.get("cu_seqlens_k") is None
    assert kwargs["max_seqlen_q"] == 1
    assert kwargs["max_seqlen_k"] == 32
    assert kwargs["block_table"] is context.block_tables
    assert kwargs["softmax_scale"] == 0.137
    assert kwargs["causal"] is True
    assert kwargs["fa_version"] == 2
    torch.testing.assert_close(out, q)
