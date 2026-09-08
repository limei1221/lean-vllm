# Piecewise CUDA graphs

Design for extending graph capture to prefill and mixed steps, which today run
eager. Status: the custom-op seam has landed; nothing else has.

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

**Recommendation: Route B.** One architecture, one shape, and a GPU budget
measured in hours; the failure modes of Route A are the ones that are hardest to
debug on rented hardware. Revisit Route A if fusion turns out to be where the
rest of the gap lives.

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

## Memory, which comes out of the KV cache

Static buffers are `max_num_batched_tokens × hidden`, plus q/k/v. At 8192 and
Qwen3-8B's hidden size that is ~64MB for one bf16 hidden-state buffer, and a
handful are needed. `kvcache_bytes()` sizes the cache from what is left after
peak allocation, so **capture shrinks the KV cache**, which changes preemption
behaviour, which changes the benchmark.

Pin `--num-kvcache-blocks` on both arms of any A/B, or the result measures two
things at once. This is the same confound the runbook already flags for
`--max-num-batched-tokens`.

## Dispatch

`_eager_reason` becomes a mode chooser, and `Config` gains `cudagraph_mode`
(`none` | `full` | `piecewise` | `full_and_piecewise`), mirroring vLLM's names:

| step | mode |
| --- | --- |
| pure decode, rows ≤ largest row bucket | full graph, the path that exists today |
| anything else, tokens ≤ largest token bucket and ≥ smallest | piecewise |
| otherwise | eager |

`eager_steps` gains `piecewise` as a kind, so `/metrics.json` keeps answering
where the time went.

## Staging

Each step leaves the suite green.

1. ~~Attention as a custom op~~ — done, `a77ecac`.
2. ~~`cudagraph_mode` config, the dispatcher, and the metric kind~~ — done.
3. ~~Split `Qwen3DecoderLayer` into `pre_attention()` and `post_attention()`~~ — done.
4. Capture and replay the pieces. GPU only.
5. Verify, then measure.

Steps 2 and 3 were done off the GPU, so that session only spends its time on
step 4.

## Verification

Correctness first, on a small model:

- Greedy output token-identical between `--cudagraph-mode none` and
  `piecewise`, over a prompt set that spans one chunk, several chunks, and a
  mixed step. This is the check M3 used for chunked prefill, and it catches the
  failure that matters: a mis-captured graph replays stale pointers and returns
  plausible wrong tokens rather than raising.
- **RMSNorm writes in place, and only copies when it has to cast.** `x.float()`
  is a copy in bf16, which is what the runner runs; in fp32 it returns the same
  tensor and the `mul_` rewrites the caller's. Static capture buffers make that
  aliasing a live hazard rather than a latent one, so a piece must not be handed
  a buffer anything else still needs. It is also why the piece tests run bf16.
- Host syncs inside a piece break capture. `layers/` and `models/` are clean of
  `.item()`, `.tolist()` and `.cpu()` today — the only `.tolist()` is in
  `torch_backend`, inside attention, which stays eager. Re-check after any change
  to the layers, since this is a property that rots quietly.

Then the rate sweep, comparing `step_seconds` by kind against
`results/graphs/rate`, with the cache pinned.

## Kill criteria

Stop and revert if any of these hold after step 4:

- Measured per-pass overhead at rate 16 is well under 24ms. The 24ms comes from
  a rate-4 decode batch; at saturation the CPU runs ahead of a GPU that is 96%
  busy, so much of it may already be hidden.
- Reclaimed time is under ~3% of forward-pass time end to end.
- The KV cache loses enough to raise preemptions at rate 16, where the archive
  currently shows zero.
