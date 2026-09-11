"""Backends checked against dense_attention, an independent oracle using no SDPA or paging."""

import pytest
import torch

from lean_vllm.attention import BACKENDS
from lean_vllm.utils.context import Context

torch.manual_seed(0)

NUM_HEADS = 8
NUM_KV_HEADS = 2  # exercises GQA head broadcasting
HEAD_DIM = 32
SCALE = 0.137    # not head_dim**-0.5, so a dropped scale argument is detectable

# vLLM FlashAttention supports 16-token pages; exercise the production default.
BLOCK_SIZE = 16
DTYPE = {"torch": torch.float32, "flash_attn": torch.float16}    # flash kernels are fp16/bf16 only

# Tolerances per dtype for comparison against the fp32 oracle. bf16 is not checked
# against a fixed tolerance; see test_low_precision_no_worse_than_naive.
TOLERANCE = {torch.float32: 2e-3, torch.float16: 6e-3}


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


@pytest.fixture
def block_size():
    return BLOCK_SIZE


@pytest.fixture
def dtype(backend):
    """Backend-specific dtype. Tests parametrizing `dtype` shadow this fixture."""
    return DTYPE[backend.get_name()]


@pytest.fixture
def tol(dtype):
    assert dtype in TOLERANCE, f"no fixed tolerance for {dtype}; check against naive arithmetic instead"
    return TOLERANCE[dtype]


def dense_attention(q, k, v, scale=SCALE, compute_dtype=torch.float32):
    """Causal attention one head at a time. q is [lq, H, D], k/v [lk, Hkv, D] full sequence."""
    lq, num_heads, _ = q.shape
    lk, num_kv_heads, _ = k.shape
    k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

    out = torch.empty_like(q)
    for h in range(num_heads):
        scores = (q[:, h, :].to(compute_dtype) @ k[:, h, :].to(compute_dtype).T) * scale
        for j in range(lq):
            scores[j, lk - lq + j + 1:] = float("-inf")
        out[:, h, :] = (scores.softmax(dim=-1) @ v[:, h, :].to(compute_dtype)).to(q.dtype)
    return out


def make_cache(num_blocks, device, block_size, dtype):
    shape = (num_blocks, block_size, NUM_KV_HEADS, HEAD_DIM)
    return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)


def slots_for(block_table, block_size, start, end):
    """Flat cache slots for token positions [start, end) of one sequence."""
    return [block_table[p // block_size] * block_size + p % block_size for p in range(start, end)]


def write_prefix(k_cache, v_cache, k_full, v_full, block_table, block_size, num_cached):
    """Seed the cache as if computed on an earlier step."""
    # Explicit long dtype to avoid float indices from empty prefixes
    slots = torch.tensor(slots_for(block_table, block_size, 0, num_cached),
                         dtype=torch.long, device=k_cache.device)
    k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[slots] = k_full[:num_cached]
    v_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[slots] = v_full[:num_cached]


def randn(*shape, device, dtype):
    """Generate random tensors in fp32 then cast to target dtype for reproducibility."""
    return torch.randn(*shape, device=device).to(dtype)


def test_prefill_without_cache(backend, device, dtype, tol):
    """Varlen causal prefill with no cached tokens."""
    seqlens = [5, 1, 12]    # no paging on this path
    qs = [randn(n, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens]
    ks = [randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens]
    vs = [randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens]

    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    context = Context(
        is_prefill=True, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
    )
    empty = torch.tensor([], device=device, dtype=dtype)
    out = backend.prefill(torch.cat(qs), torch.cat(ks), torch.cat(vs), empty, empty, context)

    expected = torch.cat([dense_attention(q, k, v) for q, k, v in zip(qs, ks, vs)])
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


def test_prefill_with_prefix_cache(backend, device, block_size, dtype, tol):
    """Chunked prefill: seq 0 resumes mid-page after cached prefix, seq 1 starts cold."""
    num_cached = [2 * block_size + 2, 0]
    num_new = [6, 5]
    block_tables_list = [[0, 1, 2, 3, 4, 5], [6, 7, -1, -1, -1, -1]]
    k_cache, v_cache = make_cache(8, device, block_size, dtype)

    q_list, k_full, v_full = [], [], []
    slot_mapping = []
    for i, (cached, new) in enumerate(zip(num_cached, num_new)):
        total = cached + new
        k_full.append(randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        v_full.append(randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        q_list.append(randn(new, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype))
        write_prefix(k_cache, v_cache, k_full[i], v_full[i], block_tables_list[i], block_size, cached)
        slot_mapping += slots_for(block_tables_list[i], block_size, cached, total)

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
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


def test_decode(backend, device, block_size, dtype, tol):
    """One query per sequence against differing cached context lengths."""
    context_lens = [block_size + 3, 3, 4 * block_size]    # mid-page, part-page, full
    block_tables_list = [[0, 1, 2, 3], [4, 5, -1, -1], [6, 7, 8, 9]]
    k_cache, v_cache = make_cache(10, device, block_size, dtype)

    k_full, v_full = [], []
    for i, n in enumerate(context_lens):
        k_full.append(randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        v_full.append(randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        write_prefix(k_cache, v_cache, k_full[i], v_full[i], block_tables_list[i], block_size, n)

    q = randn(len(context_lens), NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    context = Context(
        is_prefill=False,
        context_lens=torch.tensor(context_lens, dtype=torch.int32, device=device),
        block_tables=torch.tensor(block_tables_list, dtype=torch.int32, device=device),
    )
    out = backend.decode(q, k_cache, v_cache, context)

    expected = torch.cat([dense_attention(q[i:i + 1], k_full[i], v_full[i]) for i in range(len(context_lens))])
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


def test_store_kvcache_skips_negative_slots(backend, device, block_size, dtype):
    """Slot -1 must leave that cache row untouched."""
    k_cache, v_cache = make_cache(2, device, block_size, dtype)
    k_cache.fill_(7.0)
    v_cache.fill_(7.0)

    key = randn(3, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    value = randn(3, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    slot_mapping = torch.tensor([0, -1, 5], dtype=torch.int32, device=device)
    backend.store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    flat_k = k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)
    torch.testing.assert_close(flat_k[0], key[0])
    torch.testing.assert_close(flat_k[5], key[2])
    assert (flat_k[1:5] == 7.0).all(), "untouched slots were overwritten"


def test_decode_matches_equivalent_prefill(backend, device, block_size, dtype, tol):
    """Decode must equal a 1-token prefill over the same context."""
    seqlen = 2 * block_size + 1
    block_table = [0, 1, 2]
    k_cache, v_cache = make_cache(3, device, block_size, dtype)
    k_full = randn(seqlen, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v_full = randn(seqlen, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    write_prefix(k_cache, v_cache, k_full, v_full, block_table, block_size, seqlen)

    q = randn(1, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
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
    torch.testing.assert_close(decoded, prefilled, atol=tol, rtol=tol)


def test_top_left_causal_alignment_would_be_wrong(backend, device, block_size, dtype, tol):
    """With lq < lk a top-left mask differs; fails if the test stops discriminating."""
    lq, lk = 4, 2 * block_size + 2
    block_table = [0, 1, 2]
    k_cache, v_cache = make_cache(3, device, block_size, dtype)
    k_full = randn(lk, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v_full = randn(lk, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    write_prefix(k_cache, v_cache, k_full, v_full, block_table, block_size, lk)

    q = randn(lq, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    out = backend.prefill(q, k_full[-lq:], v_full[-lq:], k_cache, v_cache, Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, lq], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, lk], dtype=torch.int32, device=device),
        max_seqlen_q=lq, max_seqlen_k=lk,
        block_tables=torch.tensor([block_table], dtype=torch.int32, device=device),
    ))

    torch.testing.assert_close(out, dense_attention(q, k_full, v_full), atol=tol, rtol=tol)

    top_left = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k_full.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1).transpose(0, 1).unsqueeze(0),
        v_full.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1).transpose(0, 1).unsqueeze(0),
        is_causal=True, scale=SCALE,
    ).squeeze(0).transpose(0, 1)
    # well clear of the noise floor the assert_close above already allows
    assert not torch.allclose(out, top_left, atol=5 * tol), \
        "top-left and bottom-right masks agree; test is not discriminating"


def test_gqa_fallback_matches_broadcast(backend, device, block_size, dtype, tol, monkeypatch):
    """Force the torch<2.5 path that materializes KV heads instead of broadcasting."""
    if backend.get_name() != "torch":
        pytest.skip("fallback is specific to the torch backend")

    from lean_vllm.attention import torch_backend

    seqlen = 2 * block_size + 3
    block_table = [0, 1, 2]
    k_cache, v_cache = make_cache(3, device, block_size, dtype)
    k_full = randn(seqlen, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v_full = randn(seqlen, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    write_prefix(k_cache, v_cache, k_full, v_full, block_table, block_size, seqlen)

    q = randn(4, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    context = Context(
        is_prefill=False,
        context_lens=torch.tensor([seqlen] * 4, dtype=torch.int32, device=device),
        block_tables=torch.tensor([block_table] * 4, dtype=torch.int32, device=device),
    )
    expected = torch.cat([dense_attention(q[i:i + 1], k_full, v_full) for i in range(4)])

    monkeypatch.setattr(torch_backend, "_SDPA_ENABLE_GQA", False)
    torch.testing.assert_close(backend.decode(q, k_cache, v_cache, context), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_low_precision_no_worse_than_naive(backend, device, block_size, dtype):
    """Backend error against fp32 oracle must not exceed naive arithmetic in the same dtype.

    At bf16 with 8-bit mantissa, fixed atol is meaningless. Instead verify the backend
    loses no more accuracy than naive arithmetic in the same precision.
    """
    num_cached, num_new = 2 * block_size + 2, 6
    total = num_cached + num_new
    block_table = [0, 1, 2, 3]
    k_cache, v_cache = make_cache(4, device, block_size, dtype)

    k_full = randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v_full = randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    q = randn(num_new, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)

    write_prefix(k_cache, v_cache, k_full, v_full, block_table, block_size, num_cached)
    backend.store_kvcache(
        k_full[num_cached:], v_full[num_cached:], k_cache, v_cache,
        torch.tensor(slots_for(block_table, block_size, num_cached, total), dtype=torch.int32, device=device),
    )
    out = backend.prefill(q, k_full[num_cached:], v_full[num_cached:], k_cache, v_cache, Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, num_new], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, total], dtype=torch.int32, device=device),
        max_seqlen_q=num_new, max_seqlen_k=total,
        block_tables=torch.tensor([block_table], dtype=torch.int32, device=device),
    ))

    # identical inputs, so the only difference is the precision of the arithmetic
    ref = dense_attention(q.float(), k_full.float(), v_full.float())
    naive = dense_attention(q, k_full, v_full, compute_dtype=dtype).float()

    backend_err = (out.float() - ref).abs().max().item()
    naive_err = (naive - ref).abs().max().item()
    print(f"\n{dtype} backend_err={backend_err:.3e} naive_err={naive_err:.3e} "
          f"ratio={backend_err / max(naive_err, 1e-12):.2f}")
    assert backend_err <= 2 * naive_err + 1e-6, (
        f"backend error {backend_err:.3e} exceeds twice naive {dtype} error {naive_err:.3e}")


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_low_precision_cache_roundtrip_is_exact(backend, device, block_size, dtype):
    """store_kvcache must not perturb values; only attention arithmetic may lose precision."""
    block_table = [0, 1]
    k_cache, v_cache = make_cache(2, device, block_size, dtype)
    n = 2 * block_size
    key = randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    value = randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)

    slots = slots_for(block_table, block_size, 0, n)
    backend.store_kvcache(key, value, k_cache, v_cache,
                          torch.tensor(slots, dtype=torch.int32, device=device))

    torch.testing.assert_close(k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[slots], key, atol=0, rtol=0)
    torch.testing.assert_close(v_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[slots], value, atol=0, rtol=0)


def _mixed_batch(device, block_size, dtype, num_cached, num_new, block_tables_list, num_blocks):
    """Seed a cache and return (q, k_new, v_new, k_cache, v_cache, k_full, v_full)."""
    k_cache, v_cache = make_cache(num_blocks, device, block_size, dtype)
    q_list, k_full, v_full, slot_mapping = [], [], [], []
    for i, (cached, new) in enumerate(zip(num_cached, num_new)):
        total = cached + new
        k_full.append(randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        v_full.append(randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        q_list.append(randn(new, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype))
        write_prefix(k_cache, v_cache, k_full[i], v_full[i], block_tables_list[i], block_size, cached)
        slot_mapping += slots_for(block_tables_list[i], block_size, cached, total)
    k_new = torch.cat([k[c:] for k, c in zip(k_full, num_cached)])
    v_new = torch.cat([v[c:] for v, c in zip(v_full, num_cached)])
    return q_list, k_new, v_new, k_cache, v_cache, k_full, v_full, slot_mapping


def _mixed_context(device, num_cached, num_new, block_tables_list):
    totals = [c + n for c, n in zip(num_cached, num_new)]
    return Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, *torch.tensor(num_new).cumsum(0).tolist()], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, *torch.tensor(totals).cumsum(0).tolist()], dtype=torch.int32, device=device),
        max_seqlen_q=max(num_new), max_seqlen_k=max(totals),
        block_tables=torch.tensor(block_tables_list, dtype=torch.int32, device=device),
    )


def test_mixed_batch_of_chunks_and_decodes(backend, device, block_size, dtype, tol):
    """One batch holding a decode row, a resumed chunk and a cold prefill."""
    num_cached = [2 * block_size + 5, block_size + 1, 0]
    num_new = [1, 7, 4]    # decode, resumed chunk, cold
    block_tables_list = [[0, 1, 2], [3, 4, -1], [5, -1, -1]]

    q_list, k_new, v_new, k_cache, v_cache, k_full, v_full, slots = _mixed_batch(
        device, block_size, dtype, num_cached, num_new, block_tables_list, 6
    )
    backend.store_kvcache(k_new, v_new, k_cache, v_cache,
                          torch.tensor(slots, dtype=torch.int32, device=device))
    context = _mixed_context(device, num_cached, num_new, block_tables_list)
    out = backend.prefill(torch.cat(q_list), k_new, v_new, k_cache, v_cache, context)

    expected = torch.cat([dense_attention(q, k, v) for q, k, v in zip(q_list, k_full, v_full)])
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


def test_mixed_batch_matches_running_the_rows_separately(backend, device, block_size, dtype, tol):
    """One mixed call must equal the prefill call plus the decode call it replaces."""
    num_cached = [block_size + 1, 3 * block_size]
    num_new = [6, 1]    # a chunk, then a decode row; order must not matter
    block_tables_list = [[0, 1, -1, -1], [2, 3, 4, 5]]

    q_list, k_new, v_new, k_cache, v_cache, _, _, slots = _mixed_batch(
        device, block_size, dtype, num_cached, num_new, block_tables_list, 6
    )
    backend.store_kvcache(k_new, v_new, k_cache, v_cache,
                          torch.tensor(slots, dtype=torch.int32, device=device))

    merged = backend.prefill(torch.cat(q_list), k_new, v_new, k_cache, v_cache,
                             _mixed_context(device, num_cached, num_new, block_tables_list))

    chunk = backend.prefill(q_list[0], k_new[:num_new[0]], v_new[:num_new[0]], k_cache, v_cache, Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, num_new[0]], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, num_cached[0] + num_new[0]], dtype=torch.int32, device=device),
        max_seqlen_q=num_new[0], max_seqlen_k=num_cached[0] + num_new[0],
        block_tables=torch.tensor([block_tables_list[0]], dtype=torch.int32, device=device),
    ))
    decoded = backend.decode(q_list[1], k_cache, v_cache, Context(
        is_prefill=False,
        context_lens=torch.tensor([num_cached[1] + num_new[1]], dtype=torch.int32, device=device),
        block_tables=torch.tensor([block_tables_list[1]], dtype=torch.int32, device=device),
    ))
    torch.testing.assert_close(merged, torch.cat([chunk, decoded]), atol=tol, rtol=tol)


def test_decode_cuda_graph_replays_new_lengths_with_padding(backend, device, block_size, dtype, tol):
    """A captured decode must use updated lengths and tolerate empty padding rows."""
    if backend.get_name() != "flash_attn":
        pytest.skip("requires vLLM FlashAttention on CUDA")

    total = block_size + 3
    k_cache, v_cache = make_cache(3, device, block_size, dtype)
    k = randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v = randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    write_prefix(k_cache, v_cache, k, v, [2, 0], block_size, total)
    q = randn(2, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    context = Context(
        context_lens=torch.tensor([block_size + 1, 0], dtype=torch.int32, device=device),
        block_tables=torch.tensor([[2, 0], [1, 1]], dtype=torch.int32, device=device),
    )
    # Initialize kernels before capture, as ModelRunner does.
    backend.decode(q, k_cache, v_cache, context)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = backend.decode(q, k_cache, v_cache, context)

    context.context_lens[0] = total
    q.copy_(randn(2, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype))
    graph.replay()
    torch.testing.assert_close(out[:1], dense_attention(q[:1], k, v), atol=tol, rtol=tol)
    torch.testing.assert_close(out[1], torch.zeros_like(out[1]), atol=0, rtol=0)
