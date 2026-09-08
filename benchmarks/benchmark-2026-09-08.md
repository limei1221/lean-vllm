# Online serving benchmark — 2026-09-08

Source: `results-2026-09-08.tar.gz`. A100 80GB SXM, Qwen3-8B, lean-vLLM at
`75649b8` against vLLM 0.28.0 on the same box, one after the other.

Every run: 1000 requests, lognormal trace (input 512, output 128, σ=0.8),
seed 0, greedy, 3 warmup requests, KV cache pinned to 327680 tokens on both
engines. Identical prefill token counts across runs confirm the trace replayed
the same every time.

## Headline

lean-vLLM saturates **9.3% below vLLM**: 8.63 vs 9.52 req/s, 1526 vs 1684
output tok/s. Both engines completed 1000/1000 requests in every run, with zero
rejections and zero failures.

The two curves are indistinguishable up to 4 req/s and separate at 8.

| offered | lean goodput | vLLM goodput | lean ttft_p99 | vLLM ttft_p99 | lean tpot_p50 | vLLM tpot_p50 |
| ------: | -----------: | -----------: | ------------: | ------------: | ------------: | ------------: |
| 1 | 1.01 | 1.01 | 0.31 | 0.27 | 0.0126 | 0.0115 |
| 2 | 2.00 | 2.00 | 0.28 | 0.30 | 0.0138 | 0.0123 |
| 4 | 3.84 | 3.86 | 0.35 | 0.35 | 0.0166 | 0.0143 |
| 8 | 7.08 | 7.15 | 0.49 | 0.48 | 0.0281 | 0.0230 |
| 12 | **8.59** | **9.52** | 7.29 | 1.23 | 0.1105 | 0.0565 |
| 16 | 8.63 | 9.37 | 27.52 | 20.65 | 0.1472 | 0.1324 |
| 24 | 8.63 | 9.30 | 47.66 | 41.86 | 0.1483 | 0.1368 |

Rate 12 is where it hurts: same offered load, and lean-vLLM's ttft_p99 is
**5.9x** vLLM's (7.29s vs 1.23s) because it is already past its knee while vLLM
is not. Past the knee both engines behave sanely — lean-vLLM is flat at 8.63,
vLLM decays 2% from its peak.

## Where the gap comes from

**Not the scheduler.** At 1 req/s there is no queue and nothing to schedule, and
lean-vLLM is already 9.3% slower per output token (12.6ms vs 11.5ms tpot_p50) —
almost exactly the saturated throughput gap. The cost is in per-step model
execution, so the place to look is kernels and the decode path, not the policy
code.

**Chunked prefill is the one scheduler-side loss.** With it off, lean-vLLM
reaches 9.76 req/s at rate 16 — 9.3% better than with it on, and past vLLM's
9.37 at the same offered rate. It pays for that with 101 preemptions and 4%
recomputed prefill, and still wins.

Two smaller handicaps in the headline run, both from the pinned server args
(`--max-num-batched-tokens 8192 --max-num-seqs 256`, applied to both engines):
lean-vLLM's own defaults (16384/512) give 8.93 instead of 8.63 at rate 16.

## Scheduler A/Bs

All at lean-vLLM defaults unless stated, rates 4/8/16, goodput in req/s.

### CUDA graphs are worth 30-37%

| arm | rate 4 | rate 8 | rate 16 |
| --- | -----: | -----: | ------: |
| chunked on, graphs on | 3.84 | 7.07 | 8.93 |
| chunked on, eager | 3.42 | 5.66 | 6.84 |
| chunked off, graphs on | 3.84 | 7.06 | **9.76** |
| chunked off, eager | 3.40 | 5.60 | 7.13 |

Eager mode costs 12% at rate 4 and 30-37% at rate 16. Mean step time at rate 4
is 16.2ms with graphs and 40.5ms without — 2.5x, on the same batch shapes. Under
load only 80-86% of steps are captured (the rest are prefill and uncaptured
shapes), so the win is smaller than the step-time ratio suggests but still the
largest single effect in the archive.

### Chunked prefill costs throughput and tail latency here

At rate 16 with graphs on, turning it **off** gives +9.3% goodput (9.76 vs
8.93), ttft_p99 11.5s instead of 20.7s, and e2e_p99 77.6s instead of 87.3s. Its
only win is preemptions: 14 with, 101 without.

That is backwards from the intent and worth a look. Mean batch is *larger* with
chunking on (399 vs 346 tokens) and step time is *longer* (52.2ms vs 40.1ms), so
the chunked path appears to be assembling more expensive steps rather than
smoothing them.

### Token budget only moves ttft

At rate 8, goodput is flat across 512 / 2048 / 8192 (7.05 / 7.07 / 7.08). What
changes is ttft_p99 — 1.06s / 0.54s / 0.50s — and the captured-graph fraction,
which drops to 0.71 at budget 512. Nothing here argues against the 8192 default.

### Both fairness suites are null results, and the reason is visible

`long-prefill-token-threshold` and `fcfs` vs `priority`, mixed trace (20% long
prompts at 3072 tokens), rate 8:

| | long ttft_p50 | long ttft_p99 | short ttft_p99 | e2e_p99 |
| --- | ---: | ---: | ---: | ---: |
| threshold 0 | 0.360 | 1.058 | 0.808 | 9.08 |
| threshold 512 | 0.699 | 1.738 | **0.489** | 10.29 |
| fcfs | 0.365 | 1.026 | 0.848 | 9.10 |
| priority | 0.362 | 1.063 | 0.889 | 9.32 |

The threshold does what it is meant to — short requests' ttft_p99 improves 39%,
long prompts pay 94% on ttft_p50 — but the effect is small and overall e2e_p99
gets worse.

`fcfs` vs `priority` shows nothing at all, and the server metrics say why: mean
queue time is **1.0ms** in both arms. At rate 8 the mixed trace never builds a
queue, so there is no ordering for a policy to change. This suite tested nothing.
Rerun it at rate 12-16, where queue time reaches 0.9-18s.

## Cache pressure behaves as designed

KV cache cut 8x, to 40960 tokens, against the matching roomy arm (chunked on,
graphs on):

| offered | roomy | tight | preemptions | ttft_p99 tight |
| ------: | ----: | ----: | ----------: | -------------: |
| 4 | 3.84 | 3.83 | 0 | 0.34 |
| 8 | 7.07 | 6.98 | 111 | 3.28 |
| 16 | 8.93 | 7.26 | 164 | 58.87 |

Preemption is nearly free at rate 8 (-1.3%) and expensive at 16 (-18.7%), where
recomputation adds 6.8% extra prefill tokens and mean queue time hits 28s.

## Machine caveats

The 1410 MHz clock lock did not hold. Across GPU-busy samples the card sat at
mean 1377 MHz with a floor of 1155, `SwPowerCap` asserted on 38% of them, and
power peaking at 475W. Temperature never passed 63°C, so this is the power cap,
not cooling.

It scales with load, which is the awkward part — the heaviest arms are the most
throttled:

| run | mean SM clock | samples power-capped |
| --- | ------------: | -------------------: |
| lean rate 1 | 1400 | 8% |
| lean rate 24 | 1329 | 83% |
| vLLM rate 1 | 1395 | 13% |
| vLLM rate 24 | 1326 | 90% |

The head-to-head survives it: at every matched rate the two engines saw the same
clocks within 1% (rate 8: 1377 both; rate 24: 1329 vs 1326). The A/B suites ran
back-to-back within minutes of each other and are similarly matched. Absolute
tok/s numbers are understated by roughly the clock deficit at the top of the
curve; the comparisons are not.

## Three problems with the data itself

1. **`kv_cache_usage` is 0.00 in every run**, including the tight-cache run that
   preempted 164 times. It is sampled after the queue drains, so it records the
   idle value. Record a high-water mark instead — as it stands the field is
   unusable, and cache occupancy is exactly what the tight suite is about.
2. **The rate suite and the A/B suites ran different server configs.** The rate
   curve pinned 8192/256; every other suite used the 16384/512 defaults. The
   suites are internally consistent, but no number from the A/B tables can be
   compared to the headline curve directly. Worth pinning one config across the
   whole sweep.
3. **`nvidia-smi.log` timestamps are one hour behind the result files' mtimes.**
   A timezone mismatch on the box. Aligned by matching run durations to GPU-busy
   blocks, which is exact but should not be necessary — have the sweep stamp
   start and end times into each result JSON.

## What to run next

1. Full lean-vLLM rate curve with `--no-enable-chunked-prefill`. One data point
   says it beats vLLM at rate 16; confirm across the curve before believing it.
2. Rerun `policy` and `starvation` at rate 12-16. At rate 8 there is no queue,
   so both suites measured nothing.
3. Profile a decode step. The 9.3% single-stream tpot deficit at rate 1 is the
   whole saturated throughput gap, and it is upstream of every scheduler knob.
