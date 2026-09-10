# Online serving benchmark — 10 September 2026

These runs tested lean-vLLM with Qwen3-8B on an NVIDIA A100-SXM4-80GB, using the
`feature/online-serving` branch. Results are in [`results/`](../results/).
All 34 runs completed without rejections or failures: 32 runs used 1,000
requests each, and the two low-load runs used 200 each.

The benchmark ran under commit `91f933ee1b06ae7951ee9b1b0325b7c3fd023803`.

The main findings are:

- **Disabling chunked prefill improved throughput by 16% at an offered load of
  16 requests/s**, from 8.13 to 9.42 completed requests/s. Requests received their
  first token sooner, but token generation was slower at high load and required
  more preemptions.
- **Full decode CUDA graphs helped; piecewise graphs reduced throughput on this
  workload.** At 16 requests/s, `full` achieved 8.72 completed requests/s, compared
  with 8.18 for the default `full_and_piecewise` mode.
- **Priority scheduling substantially reduced short-request waiting time under
  load.** The benefit came at the expense of longer waits for long requests.
- **The combination of chunked prefill off and `--cudagraph-mode full` still needs
  testing.** The two improvements were measured separately and cannot yet be
  treated as a combined gain.

**There was no vLLM run in this batch.** Any vLLM numbers below come from the
[8 September report](benchmark-2026-09-08.md). Different GPU clock conditions
prevent a reliable comparison across the two dates.

## Setup and metric definitions

### Hardware and runtime

| Component | Value |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU memory | 81,920 MiB |
| NVIDIA driver | 580.159.04 |
| Linux kernel | `6.8.0-136-generic` |
| Python | 3.12.3 |
| PyTorch | `2.9.1+cu128` (CUDA 12.8 build) |
| System time at metadata collection | 10 September 2026, 15:05:43 UTC (`+0000`) |
| Persistence mode | Enabled |
| Application graphics clock | 1,155 MHz |
| Maximum graphics clock | 1,410 MHz |
| Power limit / maximum power limit | 400.00 W / 400.00 W |

These settings describe the hardware snapshot. GPU clocks and power observed
during the runs are reported separately under measurement limits.

### Workload and server settings

| Setting | Value |
| --- | --- |
| Requests per run | 1,000, or 200 for the low-load runs; plus 3 warmup requests |
| Main workload | Lognormal lengths: input parameter 512, output parameter 128, σ = 0.8 |
| Sampling | Seed 0, greedy decoding |
| Server limits | `--max-num-batched-tokens 8192 --max-num-seqs 256` throughout |
| Standard KV cache capacity | 327,680 tokens |
| Default graph mode in these runs | `full_and_piecewise` |
| Exceptions | Fairness tests use a mixed workload; cache-pressure tests use a smaller cache |

The 1,000-request lognormal runs each recorded 661,306 prompt tokens in the
server counters, consistent with replaying the same trace. The mixed workload is
specified in the fairness section and should be compared only within that suite.

### Reading the results

The tables use these metrics:

| Metric | Meaning | Better direction |
| --- | --- | --- |
| Offered load | Target request arrival rate, in requests/s | Test input |
| Goodput | Completed requests divided by total run time, including time to drain the queue; no latency cutoff is applied | Higher |
| TTFT | Time to first token: how long a request waits before output begins | Lower |
| TPOT | Time per output token after the first token | Lower |
| E2E | End-to-end time to complete a request | Lower |
| p50 / p99 | Median / 99th percentile; p99 describes the slow tail | Lower for latency |

Prefill processes the input prompt; decode generates output tokens. Chunked
prefill splits prompt processing across steps. Preemption pauses a request to
free KV cache space and can require recomputing tokens later.

## 1. Disabling chunked prefill increased capacity, with tradeoffs

Both configurations below used the default `full_and_piecewise` graph mode.
Throughput was similar up to 10 requests/s. At 12 requests/s, the difference
became substantial: disabling chunking raised goodput from **8.14 to 9.16
requests/s** and reduced p99 TTFT from **14.57 to 1.66 seconds**, an 8.8× reduction.

### Throughput and time to first token

| Offered load (req/s) | Goodput, chunking on (req/s) | Goodput, chunking off (req/s) | p99 TTFT, on (s) | p99 TTFT, off (s) |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 3.84 | 3.84 | 0.48 | 0.46 |
| 8 | 7.06 | 7.05 | 0.82 | 0.71 |
| 10 | 8.17 | 8.32 | 1.09 | 0.92 |
| 12 | 8.14 | **9.16** | 14.57 | **1.66** |
| 14 | 8.04 | **9.43** | 27.34 | 6.47 |
| 16 | 8.13 | 9.42 | 35.08 | 15.49 |
| 24 | 8.30 | 9.42 | 52.73 | 34.80 |

With chunking on, goodput levels off around 8.1–8.3 requests/s and tail latency
rises sharply between offered loads of 10 and 12. With chunking off, goodput
reaches about 9.4 requests/s and the sharp latency increase occurs later,
between 12 and 14. This is the shift in the saturation “knee.”

### Token generation and preemption

Disabling chunking does not improve every latency metric. At offered loads of
16 and 24 requests/s, median time per output token increases:

| Offered load (req/s) | p50 TPOT, chunking on (ms) | p50 TPOT, chunking off (ms) |
| ---: | ---: | ---: |
| 4 | 17.2 | 18.1 |
| 8 | 31.0 | 34.5 |
| 10 | 59.7 | 54.8 |
| 12 | 145.0 | 97.3 |
| 14 | 159.9 | 151.9 |
| 16 | 160.5 | 195.2 |
| 24 | 157.4 | 205.8 |

Chunking off also caused **50, 135, and 168 preemptions** at offered loads of
14, 16, and 24 respectively, versus zero with chunking on. At 16 requests/s,
recomputation added 5.2% extra prefill tokens.

For this workload, disabling chunking trades slower token generation at high
load and more recomputation for higher throughput and a shorter initial wait.

## 2. Full decode graphs helped; piecewise graphs added overhead

The graph-mode comparison used the same workload at offered loads of 8 and
16 requests/s. `full` uses full decode graphs; `piecewise` captures portions of
other execution steps; `full_and_piecewise` combines the two. Uncaptured steps
run eagerly.

| Graph mode | Goodput at load 8 (req/s) | Goodput at load 16 (req/s) | Steps replaying a graph at load 16 |
| --- | ---: | ---: | ---: |
| `none` | 5.34 | 6.36 | 0% |
| `piecewise` | 5.26 | 6.03 | 25% |
| `full` | 7.08 | **8.72** | 79% |
| `full_and_piecewise` | 7.04 | 8.18 | 100% |

Full decode graphs increased goodput by **33–37% relative to no graphs**.
Adding piecewise capture to full graphs reduced goodput by **6.2% at load 16**.
Piecewise capture alone showed no benefit over eager execution; clock differences
make the size of that comparison less certain.

### Where the extra time went

At load 16, total decode-graph time was almost unchanged: 31.3 seconds with
`full`, versus 31.4 seconds with `full_and_piecewise`. The difference was in
steps that also performed prefill:

| Execution of non-decode steps | Total time (s) | Step count | Mean time per step (ms) |
| --- | ---: | ---: | ---: |
| Eager, with `full` | 82.1 | 468 | 175 |
| Captured, with `full_and_piecewise` | 88.7 | 420 | 211 |

The captured path spent about 8% more total time processing the same prefill
work. Padding is a likely contributor: these steps averaged 1,574 tokens, while
the next capture bucket was 2,048 tokens, roughly 30% larger. The timings locate
the overhead in this path, but do not isolate padding as its sole cause.

Replaying graphs for 100% of steps therefore did not translate into better
performance. The combination of `full` mode and chunked prefill off remains
unmeasured.

## 3. Priority scheduling helped short requests when the queue grew

These tests used a **mixed workload with a 20% probability of long, 3,072-token
prompts**. They are separate from the lognormal throughput curve above.
Mean queue times at offered loads of 12 and 16 ranged from roughly 3 to 17
seconds, making scheduling decisions consequential.

### Priority scheduling: shorter waits for short requests

In the policy comparison, short requests had priority 0 and long requests had
priority 1. Lower numbers run first under `priority`; `fcfs` serves requests in
arrival order.

| Policy at load 16 | Short p50 TTFT (s) | Short p99 TTFT (s) | Long p50 TTFT (s) | Short p99 E2E (s) | Goodput (req/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fcfs` | 14.68 | 40.01 | 14.91 | 57.22 | 9.63 |
| `priority` | **1.57** | **4.54** | 39.00 | 41.50 | 10.05 |

Priority scheduling reduced both median and p99 TTFT for short requests by
about **89%**, while goodput rose by about 4%. Long requests paid for this:
their median TTFT increased from 14.91 to 39.00 seconds.

Load 12 showed the same pattern: short-request median TTFT fell from 7.84 to
0.83 seconds. By contrast, the earlier load-8 experiment had a mean queue time
of only 1.0 ms and showed little difference between policies.

### Long-prefill threshold: no benefit at these loads

A separate comparison tested `--long-prefill-token-threshold`, which limits how
much of a long prompt can be processed in one step. These runs used the mixed
workload without the policy comparison's priority assignments.

| Threshold at load 16 | Short p50 TTFT (s) | Short p99 TTFT (s) | Long p50 TTFT (s) | Short p99 E2E (s) | Goodput (req/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 15.32 | 42.06 | 15.71 | 59.26 | 9.46 |
| 512 | 15.58 | 43.90 | 18.39 | 61.16 | 9.21 |

Setting the threshold to 512 worsened every metric shown at load 16, including
roughly 2.6% lower goodput. It also hurt at load 12. In the earlier load-8 test,
it had improved short-request p99 TTFT by 39%.

This suggests the threshold is useful over a narrower load range: limiting work
within a step offers less help when requests spend seconds waiting in the queue.
Testing loads between 8 and 12 would clarify where the benefit disappears.

## 4. A smaller KV cache reduced throughput

The cache-pressure tests reduced capacity eightfold, from **327,680 to 40,960
tokens**, with chunked prefill enabled in both configurations.

| Offered load (req/s) | Goodput, standard cache (req/s) | Goodput, small cache (req/s) | Small-cache preemptions | Peak usage, standard / small | Mean queue time, small cache (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 7.06 | 6.85 | 169 | 25% / 100% | 1.8 |
| 16 | 8.13 | 6.89 | 146 | 74% / 100% | 31.9 |

The standard cache had no preemptions at either load. With the smaller cache,
goodput fell by about **3% at load 8** and **15% at load 16**. The earlier report
showed losses of 1.3% and 18.7% respectively.

The new `kv_cache_usage_peak` metric captures the pressure: the small cache
reached 100% usage. The ordinary `kv_cache_usage` field still reads 0.00 because
it is sampled after the queue drains. Use the peak field to assess capacity.

## 5. Decode profiling identified areas to investigate

At an offered load of 1 request/s, median TPOT was **12.5 ms**, close to the
previous lean-vLLM result of 12.6 ms. The archived vLLM result was 11.5 ms;
a fresh comparison is needed to establish the current gap.

A separate decode profile used **32 sequences, each with a 512-token prompt**:

| Component of one decode step | Time (ms) | Share |
| --- | ---: | --- |
| Total elapsed time | 15.8 | 100% of elapsed time |
| GPU device time | 13.4 | 85% of elapsed time |
| Weight matrix multiplications (GEMMs) | 10.0 | 75% of device time |
| Attention | 2.3 | 17% of device time |
| RMSNorm, 145 launches | 0.6 | 4% of device time |

The profile suggests two areas for follow-up:

- **Work outside GPU execution.** The 2.4 ms difference between elapsed and
  device time is about 15% of the step. Sampling, scheduling, and detokenisation
  outside the captured graph are candidates to investigate.
- **Attention efficiency.** The original profile analysis estimated that
  attention achieved roughly half the throughput implied by its KV traffic,
  while weight GEMMs were already near the memory-bandwidth limit.

This batch-32 profile cannot explain the low-load result by itself. A
single-stream profile is needed before assigning the remaining gap to a specific
component.

## Measurement limits

### GPU clocks varied, especially across dates

GPU clocks could not be pinned: the container refused `nvidia-smi -lgc` even as
root because clock control belongs to the host driver. Across GPU-busy samples:

| Measurement | 10 September | 8 September |
| --- | ---: | ---: |
| Mean SM clock | 1,351 MHz | 1,377 MHz |
| Minimum SM clock | 1,245 MHz | 1,155 MHz |
| Samples with `SwPowerCap` asserted | 72% | 38% |
| Peak power | 466 W | 475 W |

Clock differences were smaller within the current batch:

| Comparison at load 16 | Mean busy SM clocks (MHz), in listed order | Interpretation |
| --- | --- | --- |
| Chunking on / off | 1,328 / 1,345 | About 1.3% clock difference, versus 16% higher goodput with chunking off |
| `full` / `full_and_piecewise` | 1,323 / 1,336 | The slower configuration had the higher clock |
| `none` / `piecewise` | 1,354 / 1,324 | A 2.2% clock difference weakens conclusions about this smaller throughput difference |

These measurements support the main within-batch findings, but are not a
substitute for controlled clocks. In particular, **9.42 requests/s for lean-vLLM
in this batch versus the archived 9.37 for vLLM at load 16 does not establish
parity**. The engines need to run back to back on the same machine.

### Result metadata is incomplete

- **Timestamps required reconstruction.** `nvidia-smi.log` was offset from the
  result-file timestamps by one hour. Applying a −60-minute alignment placed
  94% of in-window samples in GPU-busy periods. Run windows had to be inferred
  from file modification times and `duration_seconds`; result JSON should record
  explicit start and end times.
- **Workloads differ between suites.** The fairness results of roughly 9.2–10.1
  requests/s cannot be compared directly with the main curve's 8.13 requests/s
  at load 16. Use each suite's own baseline.

## Next experiments, in priority order

1. **Measure chunked prefill off with `--cudagraph-mode full`.** Run the
   seven-point throughput curve to test whether the two separate improvements
   combine.
2. **Run vLLM at offered loads of 8, 12, and 16 requests/s.** Run these reference
   points back to back with the first experiment on the same machine. Estimated
   time for the vLLM runs: about 15 minutes.
3. **Repeat the graph-mode comparison with the updated capture buckets.** The
   new grid stops at 512 tokens, so larger mixed steps should run eagerly instead
   of padding to large buckets. The hypothesis is that `full_and_piecewise`
   will approach `full` on this workload. If it still trails, investigate replay
   overhead and consider making `full` the default.
4. **Repeat the `starvation` suite at loads of 10–12 requests/s.** Locate where
   the long-prefill threshold stops helping short requests.
5. **Profile single-stream decode.** Determine which components contribute to
   low-load per-token latency, then compare with a fresh vLLM run.
