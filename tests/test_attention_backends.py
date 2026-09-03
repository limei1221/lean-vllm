"""Backends checked against dense_attention, an independent oracle using no SDPA or paging."""

import pytest
import torch

from inferweave.attention import BACKENDS
from inferweave.utils.context import Context

torch.manual_seed(0)

NUM_HEADS = 8
NUM_KV_HEADS = 2  # exercises GQA head broadcasting
HEAD_DIM = 32
BLOCK_SIZE = 4
SCALE = 0.137    # not head_dim**-0.5, so a dropped scale argument is detectable

def _cases():
    """Every (backend, device) pair runnable here."""
    cases = []
    for backend_cls in BACKENDS:
        if not backend_cls.is_available():
            continue
        if backend_cls.get_name() == "torch":
            devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        else:
            devices = ["cuda"]
        cases += [(backend_cls, d) for d in devices]
    return cases


CASES = _cases()


@pytest.fixture(params=CASES, ids=[f"{b.get_name()}-{d}" for b, d in CASES])
def case(request):
    backend_cls, device = request.param
    return backend_cls(NUM_HEADS, HEAD_DIM, SCALE, NUM_KV_HEADS), torch.device(device)


@pytest.fixture
def backend(case):
    return case[0]


@pytest.fixture
def device(case):
    return case[1]


def dense_attention(q, k, v, scale=SCALE):
    """Causal attention one head at a time. q is [lq, H, D], k/v [lk, Hkv, D] full sequence."""
    lq, num_heads, _ = q.shape
    lk, num_kv_heads, _ = k.shape
    k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

    out = torch.empty_like(q)
    for h in range(num_heads):
        scores = (q[:, h, :].float() @ k[:, h, :].float().T) * scale
        for j in range(lq):
            scores[j, lk - lq + j + 1:] = float("-inf")
        out[:, h, :] = (scores.softmax(dim=-1) @ v[:, h, :].float()).to(q.dtype)
    return out


def make_cache(num_blocks, device, dtype=torch.float32):
    shape = (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)


def slots_for(block_table, start, end):
    """Flat cache slots for token positions [start, end) of one sequence."""
    return [block_table[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE for p in range(start, end)]


def write_prefix(k_cache, v_cache, k_full, v_full, block_table, num_cached):
    """Seed the cache as if computed on an earlier step."""
    for p, slot in enumerate(slots_for(block_table, 0, num_cached)):
        k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[slot] = k_full[p]
        v_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[slot] = v_full[p]


def randn(*shape, device):
    return torch.randn(*shape, device=device)


def test_prefill_without_cache(backend, device):
    """Varlen causal prefill with no cached tokens."""
    seqlens = [5, 1, 12]
    qs = [randn(n, NUM_HEADS, HEAD_DIM, device=device) for n in seqlens]
    ks = [randn(n, NUM_KV_HEADS, HEAD_DIM, device=device) for n in seqlens]
    vs = [randn(n, NUM_KV_HEADS, HEAD_DIM, device=device) for n in seqlens]

    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    context = Context(
        is_prefill=True, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
    )
    empty = torch.tensor([], device=device)
    out = backend.prefill(torch.cat(qs), torch.cat(ks), torch.cat(vs), empty, empty, context)

    expected = torch.cat([dense_attention(q, k, v) for q, k, v in zip(qs, ks, vs)])
    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)


def test_prefill_with_prefix_cache(backend, device):
    """Chunked prefill: seq 0 resumes after 10 cached tokens, seq 1 starts cold."""
    num_cached = [10, 0]
    num_new = [6, 5]
    block_tables_list = [[0, 1, 2, 3, 4, 5], [6, 7, -1, -1, -1, -1]]
    k_cache, v_cache = make_cache(8, device)

    q_list, k_full, v_full = [], [], []
    slot_mapping = []
    for i, (cached, new) in enumerate(zip(num_cached, num_new)):
        total = cached + new
        k_full.append(randn(total, NUM_KV_HEADS, HEAD_DIM, device=device))
        v_full.append(randn(total, NUM_KV_HEADS, HEAD_DIM, device=device))
        q_list.append(randn(new, NUM_HEADS, HEAD_DIM, device=device))
        write_prefix(k_cache, v_cache, k_full[i], v_full[i], block_tables_list[i], cached)
        slot_mapping += slots_for(block_tables_list[i], cached, total)

    # only new tokens reach the layer
    k_new = torch.cat([k[c:] for k, c in zip(k_full, num_cached)])
    v_new = torch.cat([v[c:] for v, c in zip(v_full, num_cached)])
    backend.store_kvcache(
        k_new, v_new, k_cache, v_cache,
        torch.tensor(slot_mapping, dtype=torch.int32, device=device),
    )

    totals = [c + n for c, n in zip(num_cached, num_new)]
    context = Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, *torch.tensor(num_new).cumsum(0).tolist()], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, *torch.tensor(totals).cumsum(0).tolist()], dtype=torch.int32, device=device),
        max_seqlen_q=max(num_new), max_seqlen_k=max(totals),
        block_tables=torch.tensor(block_tables_list, dtype=torch.int32, device=device),
    )
    out = backend.prefill(torch.cat(q_list), k_new, v_new, k_cache, v_cache, context)

    expected = torch.cat([dense_attention(q, k, v) for q, k, v in zip(q_list, k_full, v_full)])
    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)


def test_decode(backend, device):
    """One query per sequence against differing cached context lengths."""
    context_lens = [7, 3, 16]
    block_tables_list = [[0, 1, 2, 3], [4, 5, -1, -1], [6, 7, 8, 9]]
    k_cache, v_cache = make_cache(10, device)

    k_full, v_full = [], []
    for i, n in enumerate(context_lens):
        k_full.append(randn(n, NUM_KV_HEADS, HEAD_DIM, device=device))
        v_full.append(randn(n, NUM_KV_HEADS, HEAD_DIM, device=device))
        write_prefix(k_cache, v_cache, k_full[i], v_full[i], block_tables_list[i], n)

    q = randn(len(context_lens), NUM_HEADS, HEAD_DIM, device=device)
    context = Context(
        is_prefill=False,
        context_lens=torch.tensor(context_lens, dtype=torch.int32, device=device),
        block_tables=torch.tensor(block_tables_list, dtype=torch.int32, device=device),
    )
    out = backend.decode(q, k_cache, v_cache, context)

    expected = torch.cat([dense_attention(q[i:i + 1], k_full[i], v_full[i]) for i in range(len(context_lens))])
    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)


def test_store_kvcache_skips_negative_slots(backend, device):
    """Slot -1 must leave that cache row untouched."""
    k_cache, v_cache = make_cache(2, device)
    k_cache.fill_(7.0)
    v_cache.fill_(7.0)

    key = randn(3, NUM_KV_HEADS, HEAD_DIM, device=device)
    value = randn(3, NUM_KV_HEADS, HEAD_DIM, device=device)
    slot_mapping = torch.tensor([0, -1, 5], dtype=torch.int32, device=device)
    backend.store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    flat_k = k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)
    torch.testing.assert_close(flat_k[0], key[0])
    torch.testing.assert_close(flat_k[5], key[2])
    assert (flat_k[1:5] == 7.0).all(), "untouched slots were overwritten"


def test_decode_matches_equivalent_prefill(backend, device):
    """Decode must equal a 1-token prefill over the same context."""
    seqlen = 9
    block_table = [0, 1, 2]
    k_cache, v_cache = make_cache(3, device)
    k_full = randn(seqlen, NUM_KV_HEADS, HEAD_DIM, device=device)
    v_full = randn(seqlen, NUM_KV_HEADS, HEAD_DIM, device=device)
    write_prefix(k_cache, v_cache, k_full, v_full, block_table, seqlen)

    q = randn(1, NUM_HEADS, HEAD_DIM, device=device)
    bt = torch.tensor([block_table], dtype=torch.int32, device=device)

    decoded = backend.decode(q, k_cache, v_cache, Context(
        is_prefill=False,
        context_lens=torch.tensor([seqlen], dtype=torch.int32, device=device),
        block_tables=bt,
    ))
    prefilled = backend.prefill(q, k_full[-1:], v_full[-1:], k_cache, v_cache, Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, seqlen], dtype=torch.int32, device=device),
        max_seqlen_q=1, max_seqlen_k=seqlen,
        block_tables=bt,
    ))
    torch.testing.assert_close(decoded, prefilled, atol=2e-3, rtol=2e-3)


def test_top_left_causal_alignment_would_be_wrong(backend, device):
    """With lq < lk a top-left mask differs; fails if the test stops discriminating."""
    lq, lk = 4, 10
    block_table = [0, 1, 2]
    k_cache, v_cache = make_cache(3, device)
    k_full = randn(lk, NUM_KV_HEADS, HEAD_DIM, device=device)
    v_full = randn(lk, NUM_KV_HEADS, HEAD_DIM, device=device)
    write_prefix(k_cache, v_cache, k_full, v_full, block_table, lk)

    q = randn(lq, NUM_HEADS, HEAD_DIM, device=device)
    out = backend.prefill(q, k_full[-lq:], v_full[-lq:], k_cache, v_cache, Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, lq], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, lk], dtype=torch.int32, device=device),
        max_seqlen_q=lq, max_seqlen_k=lk,
        block_tables=torch.tensor([block_table], dtype=torch.int32, device=device),
    ))

    torch.testing.assert_close(out, dense_attention(q, k_full, v_full), atol=2e-3, rtol=2e-3)

    top_left = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k_full.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1).transpose(0, 1).unsqueeze(0),
        v_full.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1).transpose(0, 1).unsqueeze(0),
        is_causal=True, scale=SCALE,
    ).squeeze(0).transpose(0, 1)
    assert not torch.allclose(out, top_left, atol=1e-2), "top-left and bottom-right masks agree; test is not discriminating"


def test_gqa_fallback_matches_broadcast(backend, device, monkeypatch):
    """Force the torch<2.5 path that materializes kv heads instead of broadcasting."""
    if backend.get_name() != "torch":
        pytest.skip("fallback is specific to the torch backend")

    from inferweave.attention import torch_backend

    seqlen = 11
    block_table = [0, 1, 2]
    k_cache, v_cache = make_cache(3, device)
    k_full = randn(seqlen, NUM_KV_HEADS, HEAD_DIM, device=device)
    v_full = randn(seqlen, NUM_KV_HEADS, HEAD_DIM, device=device)
    write_prefix(k_cache, v_cache, k_full, v_full, block_table, seqlen)

    q = randn(4, NUM_HEADS, HEAD_DIM, device=device)
    context = Context(
        is_prefill=False,
        context_lens=torch.tensor([seqlen] * 4, dtype=torch.int32, device=device),
        block_tables=torch.tensor([block_table] * 4, dtype=torch.int32, device=device),
    )
    expected = torch.cat([dense_attention(q[i:i + 1], k_full, v_full) for i in range(4)])

    monkeypatch.setattr(torch_backend, "_SDPA_ENABLE_GQA", False)
    torch.testing.assert_close(backend.decode(q, k_cache, v_cache, context), expected, atol=2e-3, rtol=2e-3)
