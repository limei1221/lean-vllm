# Attention Backend Abstraction

Status: interface + `TorchAttention` + `FlashAttentionBackend` landed, and
CUDA-graph capture is gated on `supports_cuda_graph()`.
Not yet done: FlashInfer / FlashMLA.

## Problem

`layers/attention.py` imported `flash_attn` and Triton at module scope and
called them directly. Three consequences:

1. The package could not be imported at all without CUDA, so nothing could be
   developed or tested off a GPU — including the scheduler and block manager,
   which contain no device code.
2. There was no reference implementation to check kernels against, although the
   portfolio quality bar requires differential correctness testing.
3. Adding FlashInfer or FlashMLA later would have meant `if use_flashinfer:`
   branches inside model code, which is what this project exists to avoid.

## Design

Models declare attention *semantics*; a backend owns *execution*.

```
Qwen3Attention
      |
      v
layers.attention.Attention        # owns the layer's KV cache slice
      |
      v
AttentionBackend                  # store_kvcache / prefill / decode
      |
      +-- TorchAttention          # SDPA, any device, reference oracle
      |
      +-- FlashAttentionBackend   # flash-attn + Triton scatter, CUDA only
```

`Attention.__init__` resolves a backend class once and instantiates it per
layer. `qwen3.py` did not change.

The interface is deliberately three methods, not one. `store_kvcache` belongs
to the backend because the *cache layout* is a backend concern — FlashInfer and
FlashMLA want different layouts, which is why `get_kv_cache_shape` is on the
interface too, ready for the model runner to consult once the device layer
lands.

### Tensor contract

Identical across backends; sequences are packed, not padded.

| | shape |
|---|---|
| `prefill` q | `[num_tokens, num_heads, head_dim]` |
| `prefill` k, v | `[num_tokens, num_kv_heads, head_dim]`, new tokens only |
| `prefill` returns | `[num_tokens, num_heads, head_dim]` |
| `decode` q | `[batch_size, num_heads, head_dim]` |
| `decode` returns | `[batch_size, num_heads, head_dim]` |

`flash_attn_with_kvcache` returns a singleton query axis; the flash backend
squeezes it so both backends return the same rank. An abstraction whose
implementations return different shapes is not an abstraction.

### Causal masking is bottom-right aligned

The one design decision worth stating loudly. Under prefix caching or chunked
prefill a sequence has `num_query_tokens < num_key_tokens`, and the queries are
the **final** `lq` positions of the `lk`-long key sequence. Query `j` attends to
key positions `0 ..= lk - lq + j`.

`torch.nn.functional.scaled_dot_product_attention(is_causal=True)` implements
the opposite, **top-left** alignment, and does not raise when `lq != lk` — it
silently returns a plausible but wrong answer, on exactly the code paths this
repo has been fixing recently (`fix chunked prefill bugs`, `fix cache hit`).
`TorchAttention` therefore builds the mask from absolute positions and never
passes `is_causal`. FlashAttention has used bottom-right alignment since 2.1, so
the two agree.

`test_top_left_causal_alignment_would_be_wrong` asserts both that the backend
matches the oracle *and* that the top-left result differs, so the test fails if
it ever stops discriminating.

## Backend selection

`get_attention_backend()` resolves in order: explicit argument,
`$INFERWEAVE_ATTENTION_BACKEND`, then the first available entry of `BACKENDS`.
`TorchAttention.is_available()` is unconditionally true and sits last, so
resolution cannot fail. Requesting an unavailable backend by name raises rather
than silently falling back — a silent downgrade to a 50x slower backend during a
benchmark is worse than a crash.

Selection is currently one global choice. Per-layer dispatch on head count,
dtype, sequence length or prefill-vs-decode belongs behind the same call.

## Testing

The oracle in `tests/test_attention_backends.py` is `dense_attention`: a direct
transcription of the definition, looping over heads, with no SDPA and no paging.
Agreement with it is evidence rather than tautology.

Tests are parametrized over `(backend, device)` pairs filtered by
`is_available()`, so the same file runs on a CPU-only laptop and, on a GPU box,
additionally compares `TorchAttention` and `FlashAttentionBackend` on identical
hardware.

The suite was validated by mutation testing — deliberately breaking the backend
and confirming tests fail. Two mutations initially survived and both indicated
real gaps:

| mutation | outcome |
|---|---|
| top-left causal alignment | caught |
| ignore `-1` slots in `store_kvcache` | caught |
| off-by-one in page gather | caught |
| swap k/v cache in gather | caught |
| decode reads wrong query row | caught |
| drop `scale` argument | **survived** — tests used `head_dim**-0.5`, SDPA's default. Fixed by choosing a non-default scale. |
| `repeat` instead of `repeat_interleave` for GQA | **survived** — the fallback path is dead on torch >= 2.5. Fixed by forcing it under monkeypatch. |

## Trade-offs

`TorchAttention` is optimized for being obviously correct, not fast. It walks
sequences in a Python loop and gathers each one's pages into a contiguous
tensor before calling SDPA. That costs a host sync per forward pass
(`.tolist()` on the sequence-length tensors) and memory traffic proportional to
context length that FlashAttention's on-the-fly page walk avoids. It also makes
decode data-dependent, so the backend reports
`supports_cuda_graph() == False`.

This is the right trade for an oracle and for laptop development. It is the
wrong trade for the Torch-vs-Flash crossover benchmarks, which will otherwise
measure the Python loop rather than the attention. Batching the loop into padded
tensors is the obvious next optimization, and should happen before those numbers
are published.

## Next

1. Have `allocate_kv_cache` call `get_kv_cache_shape`; the hook exists but the
   model runner still hardcodes the FlashAttention layout.
2. Batch the per-sequence loop in `TorchAttention` before publishing any
   Torch-vs-Flash crossover numbers.
3. FlashInfer / FlashMLA backends, then per-layer dispatch.
