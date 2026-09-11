# Online serving benchmark — 11 September 2026

These runs compared lean-vLLM against vLLM back to back on the same machine in
the same session, using Qwen3-8B on an NVIDIA A100-SXM4-80GB. This is the
comparison the [10 September report](benchmark-2026-09-10.md) called for: its
vLLM numbers came from 8 September under different clock conditions, and it
asked for the engines to run back to back. Results are in
[`results/`](../results/).

All 12 runs completed — six offered loads per engine, 1,000 requests each —
with no rejections, failures, or preemptions.

lean-vLLM ran at commit `c17da3663b9fc7e5540797661c837c86584846c4` (capture
piecewise graphs for small steps only) on a clean working tree. vLLM 0.28.0
ran with `FULL_AND_PIECEWISE` cudagraph mode, matching lean-vLLM's
`full_and_piecewise`.

The main findings are:

- **The engines are at parity below saturation.** At offered loads of 1, 4,
  and 8 requests/s, goodput and output throughput agree within 0.7%.
- **lean-vLLM trails by 5.7–7.0% goodput at saturation.** It plateaus at
  8.72–8.74 requests/s; vLLM plateaus at 9.25–9.39.
- **The latency tail diverges at the knee.** At 12 requests/s, p99 TTFT is
  6.27 seconds for lean-vLLM against 1.08 for vLLM, and median TPOT is 60%
  higher.
- **The deficit is located in eager mixed steps.** At saturation, the 16–18%
  of steps that mix prefill with decode run eagerly and consume 63–69% of
  forward time. This matches the ceiling the
  [piecewise CUDA graphs doc](../docs/piecewise-cuda-graphs.md) predicted.
- **Clock conditions were matched.** Mean busy SM clocks agreed within 2% at
  every load, with no power-cap throttling. Unlike previous comparisons, the
  deficit cannot be attributed to clock differences.

## Setup and metric definitions

### Hardware and runtime

| Component | Value |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU memory | 81,920 MiB |
| NVIDIA driver | 580.159.03 |
| Linux kernel | `6.8.0-134-generic` |
| Python | 3.12.14 |
| lean-vLLM PyTorch | `2.9.1+cu128` (CUDA 12.8 build) |
| vLLM | 0.28.0, PyTorch `2.13.0+cu130` |
| System time at metadata collection | 11 September 2026, 08:47:18 UTC (`+0000`) |
| Persistence mode | Enabled |
| Application graphics clock | 1,275 MHz |
| Maximum graphics clock | 1,410 MHz |
| Power limit / maximum power limit | 400.00 W / 500.00 W |

The two engines ran different PyTorch builds; vLLM 0.28.0 requires the newer
one. This is inherent to comparing against current vLLM and is listed under
measurement limits.

### Workload and server settings

| Setting | Value |
| --- | --- |
| Requests per run | 1,000, plus 3 warmup requests |
| Workload | Lognormal lengths: input parameter 512, output parameter 128, σ = 0.8 |
| Sampling | Seed 0, greedy decoding |
| Server limits | `--max-num-batched-tokens 8192 --max-num-seqs 256` on both engines |
| Chunked prefill | Enabled on both engines |
| Graph mode | `full_and_piecewise` on both engines |
| KV cache capacity | 327,680 tokens |
| Maximum model length | 4,096 |

Each 1,000-request run recorded 661,306 prompt tokens in the lean-vLLM server
counters, the same trace as the 10 September batch.

### Reading the results

| Metric | Meaning | Better direction |
| --- | --- | --- |
| Offered load | Target request arrival rate, in requests/s | Test input |
| Goodput | Completed requests divided by total run time, including time to drain the queue; no latency cutoff is applied | Higher |
| TTFT | Time to first token: how long a request waits before output begins | Lower |
| TPOT | Time per output token after the first token | Lower |
| E2E | End-to-end time to complete a request | Lower |
| p50 / p99 | Median / 99th percentile; p99 describes the slow tail | Lower for latency |

## 1. Throughput: parity below saturation, a 6–7% deficit above

| Offered load (req/s) | Goodput, lean-vLLM (req/s) | Goodput, vLLM (req/s) | Difference | Output tok/s, lean-vLLM | Output tok/s, vLLM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.01 | 1.01 | −0.0% | 179 | 179 |
| 4 | 3.84 | 3.86 | −0.5% | 679 | 683 |
| 8 | 7.10 | 7.15 | −0.7% | 1,255 | 1,264 |
| 12 | 8.74 | **9.39** | −7.0% | 1,545 | 1,660 |
| 16 | 8.72 | **9.25** | −5.7% | 1,543 | 1,636 |
| 24 | 8.74 | **9.27** | −5.7% | 1,546 | 1,640 |

Both engines saturate between offered loads of 8 and 12. lean-vLLM's model-busy
fraction is 0.97 at saturation, so the deficit is per-step cost, not idle time:
the engine is fully occupied doing slower work, not waiting.

## 2. Latency: the tail diverges at the knee

### Time to first token

| Offered load (req/s) | p50 TTFT, lean-vLLM (s) | p50 TTFT, vLLM (s) | p99 TTFT, lean-vLLM (s) | p99 TTFT, vLLM (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | **0.055** | 0.064 | 0.295 | **0.261** |
| 4 | **0.068** | 0.078 | 0.347 | 0.351 |
| 8 | **0.106** | 0.124 | 0.659 | **0.455** |
| 12 | 0.351 | **0.320** | 6.272 | **1.076** |
| 16 | 6.886 | **4.895** | 26.816 | **22.015** |
| 24 | 17.706 | **15.227** | 46.914 | **42.125** |

lean-vLLM gives the median request its first token sooner below saturation, but
its p99 TTFT is already worse at load 8, and at load 12 it is **5.8× worse**
(6.27 against 1.08 seconds). This is the same signature the 10 September batch
measured for chunked prefill: p99 TTFT at load 12 fell from 14.57 to 1.66
seconds when chunking was disabled.

### Time per output token

| Offered load (req/s) | p50 TPOT, lean-vLLM (ms) | p50 TPOT, vLLM (ms) | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 12.3 | 11.5 | +7.2% |
| 4 | 16.0 | 14.4 | +11.0% |
| 8 | 26.5 | 23.3 | +13.6% |
| 12 | 110.4 | 68.8 | +60.5% |
| 16 | 145.3 | 133.9 | +8.5% |
| 24 | 146.0 | 137.9 | +5.8% |

Median TPOT is higher for lean-vLLM at every load, including load 1, where
decode runs entirely in full CUDA graphs. Part of the gap is a constant
per-step cost outside the graph; at load 12 it compounds with decode tokens
waiting behind slow eager steps. p99 E2E is also higher for lean-vLLM at every
load, from 11.9 against 10.9 seconds at load 1 to 90.6 against 85.2 at load 24.

## 3. Where lean-vLLM's time went: eager mixed steps

Commit `c17da36` caps the piecewise capture grid at 512 tokens, so a step that
mixes prefill chunks with decode rows larger than 512 tokens dispatches as
eager. The server counters locate the deficit there. Cumulative counters are
`server.after − server.before` per run:

| Offered load (req/s) | Steps | Eager steps | Share of steps | Time in eager steps (s) | Share of forward time | Mean eager step (ms) | Mean graph-replaying step (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 69,721 | 492 | 0.7% | 38.8 | 4.5% | 79 | 11.6 |
| 4 | 16,378 | 483 | 2.9% | 40.7 | 15.9% | 84 | 12.7 |
| 8 | 5,917 | 453 | 7.7% | 45.9 | 33.4% | 101 | 14.5 |
| 12 | 2,503 | 414 | 16.5% | 70.4 | 63.1% | 170 | 17.1 |
| 16 | 2,198 | 401 | 18.2% | 76.7 | 68.8% | 191 | 17.0 |
| 24 | 2,145 | 375 | 17.5% | 76.4 | 68.7% | 204 | 16.8 |

At saturation, one eager step costs as much as ten graph-replaying steps, and
those steps hold 63–69% of all forward time. The
[piecewise CUDA graphs doc](../docs/piecewise-cuda-graphs.md) predicted 65–73%
of forward time in eager prefill steps at these loads, with roughly 9%
recoverable through capture — the same order as the measured 5.7–7.0% goodput
deficit.

Why these steps are not captured: the 10 September batch measured that padding
large steps into large buckets costs more than capture saves (`full` at 8.72
req/s beat `full_and_piecewise` at 8.18 at load 16), because lean-vLLM's
piecewise graphs are manual captures without kernel fusion. At compute-bound
large steps there is no launch overhead left to reclaim. vLLM's piecewise path
is torch.compile-based, so its pieces are fused and its large steps are
genuinely cheaper compute. Closing this gap is the case the doc makes for
Route A.

## 4. Clock conditions: the comparison is valid

GPU clocks could not be pinned. The `nvidia-smi` log was offset from
result-file timestamps by one hour, as in the 10 September batch; a −60-minute
alignment placed 342 of 417 samples inside run windows. Mean SM clocks over
GPU-busy samples (utilization ≥ 90%):

| Offered load (req/s) | Mean busy SM clock, lean-vLLM (MHz) | Mean busy SM clock, vLLM (MHz) | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 1,407 | 1,404 | 0.2% |
| 4 | 1,376 | 1,389 | 0.9% |
| 8 | 1,354 | 1,358 | 0.3% |
| 12 | 1,312 | 1,319 | 0.5% |
| 16 | 1,332 | 1,306 | 2.0% |
| 24 | 1,314 | 1,309 | 0.4% |

No sample in any run window asserted `SwPowerCap`; peak power ranged from 409
to 470 W. At load 16 the slower engine (lean-vLLM) held the higher clock, so
clock conditions do not explain any part of the deficit. This is the first
lean-vLLM against vLLM comparison on this machine where that can be said.

## Relation to the 10 September findings

- **The new capture buckets recovered the piecewise overhead.** With the old
  bucket grid, `full_and_piecewise` reached 8.18 req/s at load 16 on
  10 September; with the 512-token cap it reached 8.72, matching what `full`
  mode achieved the day before. Mean busy clocks differed by 0.3% between the
  two runs (1,336 against 1,332 MHz). The dates differ, so this is strong
  evidence rather than proof; a same-session `full` arm would settle it.
- **Chunked prefill remains the open lever.** On 10 September, disabling
  chunking raised lean-vLLM to 9.16 req/s at load 12 and 9.42 at load 16 —
  at or above vLLM's 9.39 and 9.25 in this batch. Both runs of that comparison
  predate this session, so the hypothesis that chunking off closes the
  saturation gap needs its own back-to-back run.

## Measurement limits

- **vLLM exposes no server counters here.** Model-busy fraction, batch tokens,
  and preemptions are null in the vLLM results, so the step accounting in
  section 3 is lean-vLLM-only. The client-side metrics are comparable.
- **Different PyTorch builds.** lean-vLLM ran torch 2.9.1+cu128 and vLLM ran
  torch 2.13.0+cu130. Some part of the per-step gap may belong to the torch
  versions rather than the engines.
- **Timestamps required reconstruction again.** Run windows were inferred from
  file modification times and `duration_seconds` with a −60-minute timezone
  alignment. Result JSON should record explicit start and end times; this is
  the second batch with the same defect.
- **Cross-date comparisons rest on near-identical clocks**, not on
  same-session runs. The within-session findings above do not depend on them.

## Next experiments, in priority order

1. **Run lean-vLLM with chunked prefill off, back to back with vLLM, at
   offered loads of 12 and 16.** The 10 September numbers suggest this
   configuration reaches vLLM's plateau; this session makes that test cheap
   and valid.
2. **Add a `full` arm to the same session.** Confirm that the 512-token cap
   made `full_and_piecewise` match `full`, and decide whether `full` should be
   the default.
3. **Verify vLLM 0.28's piecewise capture sizes.** If vLLM captures piecewise
   graphs up to `max_num_batched_tokens`, the same `FULL_AND_PIECEWISE` flag
   means different behavior on the two engines, and the comparison at
   saturation is between eager and compiled-fused large steps, not between two
   capture schemes.
4. **Prototype Route A piecewise capture** (torch.compile with an FX split at
   attention) for large mixed steps. The eager share at saturation is 63–69%
   of forward time; manual capture cannot reclaim it, but kernel fusion can.
5. **Record explicit run start and end times in result JSON** so the GPU-log
   alignment no longer has to be reconstructed.
