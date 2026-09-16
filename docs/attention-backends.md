# Attention Backend Abstraction

Status: interface + `TorchAttention` + `FlashAttention3Backend` +
`FlashMLABackend` landed, and CUDA-graph capture is gated on
`supports_cuda_graph()`, plus `supports_mla_decode()` for an MLA model's full
graphs. Not yet done: FlashInfer.

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
AttentionBackend                  # store_kvcache / prefill / decode / varlen_with_lse
      |
      +-- TorchAttention          # SDPA, any device, reference oracle
      |
      +-- FlashAttention3Backend  # flash-attn 3 + Triton scatter, Hopper only
            |
            +-- FlashMLABackend   # FA3, plus FlashMLA's decode over MLA latents
```

`Attention.__init__` resolves a backend class once and instantiates it per
layer. `qwen3.py` did not change.

The interface is deliberately three methods, not one. `store_kvcache` belongs
to the backend because the *cache layout* is a backend concern — FlashInfer and
FlashMLA want different layouts, which is why `get_kv_cache_shape` is on the
interface too. The model runner asks each layer for its cache shape, and a
plain `Attention` layer answers from the backend. `MLAAttention` answers with
its own latent layout, covered in [deepseek-v2.md](deepseek-v2.md).

### Tensor contract

Identical across backends; sequences are packed, not padded.

| | shape |
|---|---|
| `prefill` q | `[num_tokens, num_heads, head_dim]` |
| `prefill` k, v | `[num_tokens, num_kv_heads, head_dim]`, new tokens only |
| `prefill` returns | `[num_tokens, num_heads, head_dim]` |
| `decode` q | `[batch_size, num_heads, head_dim]` |
| `decode` returns | `[batch_size, num_heads, head_dim]` |
| `varlen_with_lse` k, v | `[num_keys, num_kv_heads, head_dim]`, no cache |
| `varlen_with_lse` returns | output as `prefill`, and lse `[num_tokens, num_heads]` |
| `mla_decode` q | `[batch_size, num_heads, latent_dim]` |
| `mla_decode` returns | `[batch_size, num_heads, v_dim]` |
| `store_latents` latent | `[num_tokens, latent_dim]`, slot `-1` skips |

`varlen_with_lse` serves MLA, which attends its cached context in chunks and
merges them by log-sum-exp. FA3 returns the lse through `return_attn_probs`, as
`[num_heads, num_tokens]`, so the flash backend transposes it. SDPA returns no
lse, so the torch backend writes that attention out.

`mla_decode` is optional, reported by `supports_mla_decode()`. It attends a
paged MLA latent cache as one key head, with each latent's first `v_dim` entries
as the value. `TorchAttention` implements it as the reference and
`FlashMLABackend` with FlashMLA's dense decode kernel, which reads 64-token
pages only, so it reports `mla_block_size() == 64`.

`store_latents` is the latent cache's scatter, the MLA counterpart of
`store_kvcache` and split from it because one cache is written, not two. It
takes the same `-1` for a row a CUDA graph padded: the flash backend masks that
in the Triton kernel, whose block overhangs a latent width that is no power of
two, and the torch backend pays a host sync to drop those rows.

`flash_attn_with_kvcache` returns a singleton query axis in the decode shape;
the flash backend squeezes it so both backends return the same rank. An
abstraction whose implementations return different shapes is not an
abstraction.

### Which FlashAttention-3 entry point runs a prefill

FA3 has two, and serving reaches both. The question each step asks is where its
keys are, not whether it is a prefill:

| step | call |
|---|---|
| no row carries cached keys | `flash_attn_varlen_func` on this step's k/v |
| some row resumes | `flash_attn_with_kvcache` with `page_table` |

FA2's varlen entry point took a `block_table`, so one call covered both. FA3's
does not, so a step that has to read keys back goes through the kvcache entry
point instead, which accepts packed queries through `cu_seqlens_q` and one key
length per row through `cache_seqlens`.

`keys_are_new` on the context is that question answered on the host, where the
runner already knows it: cumulative query and key lengths are equal exactly when
no row started from cached tokens. Reading it off the tensors instead would cost
a sync per layer. Cold prompts and the first chunk of a long one take the varlen
path and read k and v straight, with no page walk; a prefix-cache hit, a resumed
chunk, or a decode row mixed into the batch sends the whole step through the
pages.

Which one a run actually exercises is worth knowing before reading any number
from it. Chunked prefill admits new prompts into the same step as the running
decodes, so a loaded server reaches the varlen path rarely: it belongs to steps
with nothing running, to `bench_offline.py`, and to the chunked-prefill-off arm,
whose steps are whole prompts and nothing else. Whether it is faster there is
unmeasured.

The move to FA3 is also what makes the 16-token page the default. FA2 rejected
any paged block size that was not a multiple of 256, which forced a 256-token
block and made the KV-cache comparison against vLLM a comparison at vLLM's
non-default setting. FA3 walks a page table of any size.

`is_available()` requires compute capability 9 rather than any CUDA device:
FA3's kernels are Hopper's. On anything else selection falls through to
`TorchAttention`, or raises if FA3 was named explicitly. Dao-AILab publishes no
FA3 wheel, so `uv sync --extra cuda` installs a third-party build of it, pinned
by URL and hash in `pyproject.toml`.

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

## Mixed batches

A step may hold prompt chunks and decode rows together. No backend change was
needed for that: `prefill` already takes packed varlen sequences with
`num_query_tokens < num_key_tokens` per row, so a decode row is simply a row
whose query length is 1, and the bottom-right mask is already the right one.
`test_mixed_batch_of_chunks_and_decodes` checks a batch of all three shapes —
decode row, resumed chunk, cold prefill — against the dense oracle, and
`test_mixed_batch_matches_running_the_rows_separately` checks that one mixed
call equals the separate `prefill` and `decode` calls it replaces.

`decode` survives as the pure-decode fast path, because it is the only shape a
CUDA graph can capture. The runner selects it only when **no** row is a prompt
chunk. "Every query length is 1" would be the wrong test: a prompt whose last
chunk happens to be one token long also has query length 1, and it must take the
varlen path so that `logits_indices` decides whether it samples.

## Backend selection

`get_attention_backend()` resolves in order: explicit argument,
`$LEAN_VLLM_ATTENTION_BACKEND`, then the first available entry of `BACKENDS`.
`TorchAttention.is_available()` is unconditionally true and sits last, so
resolution cannot fail. MLA layers pass `mla=True`, which tries `MLA_BACKENDS`
first, so `FlashMLABackend` is picked for DeepSeek-V2 when it is built and never
reported for Qwen3. `Config` switches an MLA model's `kvcache_block_size` to the
page size that backend requires, with a warning. Requesting an unavailable backend by name raises rather
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
additionally compares `TorchAttention` and `FlashAttention3Backend` on
identical hardware.

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

1. Batch the per-sequence loop in `TorchAttention` before publishing any
   Torch-vs-Flash crossover numbers.
2. A FlashInfer backend, then per-layer dispatch.
