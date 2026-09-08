# Piecewise CUDA graphs

Graph capture for prefill and mixed steps, which used to run eager. Status:
built and measured on an A100. It pays at small prefill chunks and costs a
little at large ones; the crossover is the whole result.

## The number this rests on

From `results/graphs/rate` on the A100, `--max-num-seqs 256`:

| rate | eager steps | % of steps | % of forward-pass time | ms/prefill step | ms/graph step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 773 | 13.4% | 44.6% | 80.5 | 15.5 |
| 12 | 474 | 18.3% | 65.6% | 157.2 | 18.4 |
| 16 | 450 | 19.8% | 71.8% | 179.9 | 17.4 |
| 24 | 419 | 19.1% | 72.6% | 194.8 | 17.3 |

Every one of those steps is eager for being prefill or mixed; none for batch
size, which `--max-num-seqs 256` makes unreachable.

**72% is the ceiling on what is touched, not on what is recoverable.** Capture
removes CPU-side dispatch and launch cost; it does not make a matmul faster. The
`chunked` suite puts that cost at roughly 24ms per forward pass — eager 40.5ms
against 16.2ms captured, on the same decode batch at rate 4. If it holds for
prefill, the recoverable slice is ~24ms of a 180ms step, so **≈9% of forward
time** at rate 16. That is the same order as the 9.3% gap against vLLM.

## Why attention has to stay eager

A whole prefill step cannot go in one graph. `flash_attn_varlen_func` takes
`max_seqlen_q` and `max_seqlen_k` as Python ints and reads `cu_seqlens` values,
so a capture is only valid for the exact sequence layout it saw. The layout
changes every step.

Everything else in the model is per-token and shape-stable: norms, projections,
rope, MLP. Splitting the graph at attention leaves those capturable, which is
what "piecewise" means and why the op boundary came first.

## Two routes

**Route A, torch.compile.** vLLM's: Dynamo traces, an FX splitter cuts at the
attention op, inductor compiles each piece, each piece is captured per shape.
Gains kernel fusion as well as launch overhead. Costs a dependency on Dynamo and
inductor internals, which move between torch versions, and vLLM's version of it
is ~600 lines.

**Route B, manual capture.** Restructure a decoder layer into two explicit
callables either side of attention and capture each with `torch.cuda.CUDAGraph`,
the way `capture_cudagraph` already captures the whole decode model. No compile
stack. Roughly 250 lines. No fusion — recovers launch and dispatch cost only.

**Route B was built.** Its ceiling is now visible: at large chunks the step is
compute-bound and launch cost is already hidden, so Route B has nothing left to
reclaim there. Fusion is the only lever that would, which is the case for
revisiting Route A — see the measurements below.

## Shape buckets and the padding tax

Pieces are `[num_tokens, hidden]`, so buckets are token counts, not rows —
unlike the decode path, which buckets on batch size. Pad up to the bucket, let
the pad tokens compute garbage, and discard it: `slot_mapping` is already −1 for
pads, so the cache write skips them, and attention sees only the real slice.

The tax is real and it is new. A step with 300 real tokens padded to 512 does
70% more non-attention work. Decode-side padding is cheap because batches are
small; here it is not. Two consequences:

- Buckets need to be fine near the bottom. Something like 256, 512, 768, 1024,
  1536, 2048, 3072, 4096, 6144, 8192 rather than powers of two alone.
- The dispatcher should fall back to eager below the smallest bucket, where
  padding would cost more than dispatch.

Measured, the tax is what decides the result: it is smaller than the launch cost
at 2048-token chunks and larger than it at 8192.

## Memory, which does not come out of the KV cache

Static buffers are `max_num_batched_tokens × hidden`, plus q/k/v. At 8192 and
Qwen3-8B's hidden size that is ~64MB for one bf16 hidden-state buffer, and a
handful are needed.

The design assumed this would shrink the KV cache. It does not.
`capture_piecewise` runs *after* `allocate_kv_cache`, which has already spent
the whole `gpu_memory_utilization` budget, so the buffers and the graph pool
overshoot the budget instead. Measured on the 0.6B model at 8192:
`num_kvcache_blocks` was identical in every mode, and at
`--gpu-memory-utilization 0.98` the process ended up on 77.92GB against a
77.66GB budget.

Two consequences, opposite in sign:

- The A/B is fair for free — both arms get the same cache, so the confound the
  runbook flags for `--max-num-batched-tokens` does not apply here.
- On a tighter card or a larger model, capture OOMs rather than trading cache
  for graphs. Sizing the cache after capture, or reserving the buffers before
  `kvcache_bytes()` measures, is the fix if that bites.

## Dispatch

`_eager_reason` became a mode chooser, and `Config` gained `cudagraph_mode`
(`none` | `full` | `piecewise` | `full_and_piecewise`), mirroring vLLM's names:

| step | mode |
| --- | --- |
| pure decode, rows ≤ largest row bucket | full graph, the path that exists today |
| anything else, tokens ≤ largest token bucket and ≥ smallest | piecewise |
| otherwise | eager |

`eager_steps` gained `piecewise` as a kind, so `/metrics.json` keeps answering
where the time went.

One label is wrong. `_step_kind` ends `"prefill" if is_prefill else "oversized"`,
so a decode step that falls *below* the smallest token bucket is counted as
`oversized`. Under `--cudagraph-mode piecewise` alone that is most of the run:
23 of 27 steps in one measurement. It needs a third label, or the metric misreads
a bucket list that is too coarse at the bottom as one that is too short at the
top.

## Staging

Each step leaves the suite green.

1. ~~Attention as a custom op~~ — done, `a77ecac`.
2. ~~`cudagraph_mode` config, the dispatcher, and the metric kind~~ — done.
3. ~~Split `Qwen3DecoderLayer` into `pre_attention()` and `post_attention()`~~ — done.
4. ~~Capture and replay the pieces~~ — done and verified on an A100.
5. ~~Verify, then measure~~ — done on Qwen3-0.6B; the 8B sweep is still owed.

Steps 2 and 3 were written off the GPU. `ModelRunner.__init__` read
`self.enforce_eager` two lines before assigning it, so every construction raised
`AttributeError` — the first thing the A100 said.

## Verification

**Greedy output is not token-identical to eager, and cannot be.** Padding to a
bucket changes the row count of every GEMM in the pieces, and cuBLAS selects a
different kernel by M for the tall-K ones — `o_proj` at K=2048 and `down_proj`
at K=3072 both move by ~1 ULP between M=300 and M=512, while `qkv_proj`,
`gate_up_proj`, both RMSNorms, rope and the embedding are bitwise stable. One
ULP at layer 0 reaches ~0.25% relative by layer 27, which is enough to flip a
greedy argmax: one prompt of six diverged at token 4 in one configuration, none
of thirty-six in another.

So the check that catches a mis-capture is the one that holds exactly:

- **Replay equals the same pieces run eagerly at the same padded width,
  bitwise.** Verified over all 28 layers. This is the property a stale pointer
  or a mis-ordered buffer breaks, and unlike token equality it has no tolerance
  to argue about.
- **Pad rows are inert.** Every pad row filled with NaN, and pad ids and
  positions with junk, leaves the real rows bitwise unchanged — on pure prefill,
  chunked, and mixed steps. Nothing in a piece mixes rows, and this is what says
  so.
- **RMSNorm writes in place, and only copies when it has to cast.** `x.float()`
  is a copy in bf16, which is what the runner runs; in fp32 it returns the same
  tensor and the `mul_` rewrites the caller's. Static capture buffers make that
  aliasing a live hazard rather than a latent one, so a piece must not be handed
  a buffer anything else still needs. It is also why the piece tests run bf16.
- **Host syncs inside a piece break capture.** `layers/` and `models/` are clean
  of `.item()`, `.tolist()` and `.cpu()` today — the only `.tolist()` is in
  `torch_backend`, inside attention, which stays eager. Re-check after any change
  to the layers, since this is a property that rots quietly.

## What the A100 said

Qwen3-0.6B, A100-80GB, 128 prompts of 200–1024 tokens, 64 output tokens each,
greedy, `--max-num-seqs 256`, KV cache identical across arms.

| budget | mode | elapsed | tok/s | prefill step |
| ---: | --- | ---: | ---: | --- |
| 2048 | none | 3.89s | 2108 | — |
| 2048 | full | 2.04s | 4007 | 36 eager @ 40.6ms |
| 2048 | full_and_piecewise | **1.42s** | **5776** | 36 piecewise @ **23.0ms** |
| 8192 | none | 3.29s | 2488 | — |
| 8192 | full | 1.23s | **6681** | 9 eager @ 58.7ms |
| 8192 | full_and_piecewise | 1.24s | 6580 | 9 piecewise @ 61.8ms |

**There is a crossover and it sits below 8192 tokens on this model.** At 2048 the
prefill step loses 43% and end-to-end throughput gains 44%. At 8192 the step is
already compute-bound, the launch cost is hidden behind it, and the padding tax
is all that is left: 5% slower.

The two guesses that were wrong in the other direction:

- **Capture cost is cheap.** 2.9s for the ten buckets at 8192 — 580 captures on
  a 28-layer model — and 1.0s for the six at 2048, against a guess of tens of
  seconds. The 8B default budget is 814 captures of larger pieces, so still
  seconds rather than a minute.
- **Buffers and pool are small.** Piecewise added 0.25GB over `full` at 8192 and
  0.14GB at 2048, on the 0.6B model — but see the memory section for where it
  comes from.

Owed: the same table on Qwen3-8B. A 0.6B model is far more launch-bound per
FLOP, so its crossover is not the 8B one, and the 8B crossover is what decides
whether the default mode should be `full_and_piecewise` or `full`.

## Kill criteria

Stop and revert if any of these hold. Judged on 0.6B; the 8B sweep can still
overturn them.

- ~~Measured per-pass overhead at rate 16 is well under 24ms.~~ At 2048-token
  chunks it is 17.6ms of a 40.6ms step — real, and reclaimed.
- **Reclaimed time is under ~3% of forward-pass time end to end.** Passes at
  2048 (+44% throughput), **fails at 8192** (−1.5%). This is the live one: the
  8B run at its production `--max-num-batched-tokens` decides it.
- ~~The KV cache loses enough to raise preemptions at rate 16.~~ The cache does
  not shrink at all, and no arm preempted.
