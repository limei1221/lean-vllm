# Online serving benchmark — 13 September 2026

lean-vLLM against vLLM 0.26.0 on Qwen3-8B, one H100, with chunked prefill and
full + piecewise CUDA graphs on both engines. lean-vLLM ran two curves, async
scheduling off and on; vLLM ran its default, which is on. Results are in
[`results/20260913T101307Z/`](../results/20260913T101307Z/).

All fifteen serving runs completed: five offered loads per curve, 1,000 requests
each, no rejections, failures, or preemptions. Every run replayed the same trace
of 659,770 prompt tokens and 176,838 generated tokens.

The main findings are:

- **Below saturation the engines are at parity.** At load 1 goodput is identical
  and lean-vLLM's median TPOT is 0.2 ms (3%) above vLLM's. At load 24 lean-vLLM
  is 0.2% ahead on goodput.
- **At the plateau lean-vLLM trails by 5–7%.** With async scheduling on it
  reaches 24.45 requests/s at load 48 and 24.10 at load 64, against vLLM's 25.72
  and 26.05.
- **Async scheduling is worth 7–9%.** It adds 9.2% at load 24, 8.7% at load 48,
  and 6.9% at load 64 over the off curve, and lifts GPU utilization under load
  from 82–87% to 92–98%.
- **Pipelining removes almost all offline GPU idle.** The offline CUDA trace
  shows the GPU idle 22.4% of the step window with async scheduling off and 3.2%
  with it on, with the same kernel time in both.
- **Large prefill steps are the likely home of the remaining gap.** At load 48,
  prefill and mixed steps outside lean-vLLM's piecewise graphs, mostly those
  above 512 tokens, are 18% of steps but 52% of engine step time, and they run
  eager. vLLM runs the same steps through
  inductor-compiled code.
- **Latency still trails vLLM past load 1.** Median TPOT is 1.11–1.29× vLLM's
  from load 24 up, and p99 TTFT is 3.3–3.7× at loads 24–32.

## Setup

### Hardware and software

| Component | Value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3, 81,559 MiB |
| NVIDIA driver | 580.126.09 |
| Linux kernel | `6.8.12-680-6063-coreweave-amd64-f81899c8` |
| Python | 3.12.11 |
| PyTorch, both engines | `2.11.0+cu130` |
| lean-vLLM | commit `0af72d2`; working tree differs only in `uv.lock` mirror URLs |
| vLLM | 0.26.0 |
| Attention | FlashAttention-3 on both engines |
| Persistence mode | Enabled |
| Application graphics clock | 1,980 MHz, equal to the maximum |
| Power limit | 700 W (maximum 700 W) |

### Workload and server settings

| Setting | Both engines |
| --- | --- |
| Requests per run | 1,000, plus 3 warmup requests |
| Workload | Lognormal lengths: input 512, output 128, σ = 0.8 |
| Sampling | Greedy, seed 0 |
| Batch token budget / maximum sequences | 8,192 / 256 |
| Chunked prefill | Enabled |
| Graph mode | Full + piecewise |
| KV cache | 327,680 tokens in 16-token blocks |
| Maximum model length | 4,096 |
| Offered loads | 1, 24, 32, 48, 64 requests/s |

vLLM's async scheduling was confirmed on in every server log
([`vllm-async-scheduling.txt`](../results/20260913T101307Z/vllm-async-scheduling.txt)),
so the like-for-like pair is lean-vLLM's `async-scheduling=True` curve against
vLLM. vLLM also ran with prefix caching on, its default. lean-vLLM recorded a
0.0 prefix-cache hit rate on this trace, so prefix caching should not favour
either engine.

### How the two engines execute a step

| | lean-vLLM | vLLM 0.26.0 |
| --- | --- | --- |
| Decode batches | Full CUDA graphs | Full CUDA graphs, 35 sizes |
| Prefill and mixed steps, 64–512 tokens | Manually captured piecewise graphs; attention eager | torch.compile (inductor) pieces, captured at 51 sizes up to 512 |
| Prefill and mixed steps above 512 or under 64 tokens | Eager | Inductor-compiled pieces; captured under 64, not above 512 |
| Async scheduling | Launch step k, then await and reconcile step k-1 | On by default |

With async scheduling off, lean-vLLM awaits step k-1's tokens before scheduling
step k, and only detokenization overlaps the forward pass. With it on,
scheduling and batch preparation also run ahead of the await
([`docs/online-serving.md`](../docs/online-serving.md#pipelined-steps)).

### Reading the results

| Metric | Meaning | Better direction |
| --- | --- | --- |
| Goodput | Completed requests divided by total run time, including queue drain; no latency cutoff | Higher |
| TTFT | Time to first token | Lower |
| TPOT | Time per output token after the first | Lower |
| E2E | End-to-end request time | Lower |
| p50 / p99 | Median / 99th percentile | Lower for latency |

Goodput counts the drain after the last arrival, so it sits below the offered
load even when the engine keeps up. Compare engines at the same load, not
against the offered rate.

## 1. Throughput

| Offered load (req/s) | lean-vLLM off | lean-vLLM on | vLLM | on vs vLLM | on vs off |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.01 | 1.01 | 1.01 | −0.0% | +0.0% |
| 24 | 17.79 | 19.43 | 19.39 | **+0.2%** | +9.2% |
| 32 | 21.97 | 22.54 | 23.07 | −2.3% | +2.6% |
| 48 | 22.49 | 24.45 | 25.72 | −4.9% | +8.7% |
| 64 | 22.53 | 24.10 | **26.05** | −7.5% | +6.9% |

Goodput is in requests/s. Output token throughput follows goodput, since every
run generates the same tokens: at load 48 and 64 lean-vLLM on produces 4,323
and 4,261 tok/s, lean-vLLM off 3,978 and 3,985, and vLLM 4,548 and 4,607.

Both lean-vLLM curves level off from load 48: at 22.5 requests/s with async
scheduling off and at 24.1–24.5 with it on. vLLM is still climbing slightly at
load 64. The gap therefore opens only once the engines are capacity-bound, and
widens as vLLM's extra capacity is used.

Load 32 is the one point where async scheduling adds little (+2.6%). Its
latency is also out of line with the neighbouring loads (section 2), so treat
it as a single noisy run until it is repeated.

## 2. Latency

| Offered load (req/s) | p50 TTFT on / vLLM (ms) | p99 TTFT on / vLLM (s) | p50 TPOT on / vLLM (ms) | p50 E2E on / vLLM (s) | p99 E2E on / vLLM (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30 / 30 | 0.12 / 0.12 | 7.0 / 6.8 | 0.92 / 0.89 | 6.54 / 6.38 |
| 24 | 102 / 72 | 1.01 / 0.30 | 14.4 / 12.7 | 2.04 / 1.74 | 13.33 / 11.49 |
| 32 | 452 / 144 | 2.39 / 0.65 | 24.8 / 19.2 | 3.93 / 2.72 | 20.18 / 16.12 |
| 48 | 2,114 / 1,206 | 6.75 / 4.56 | 43.3 / 37.7 | 8.60 / 6.73 | 27.93 / 24.77 |
| 64 | 5,012 / 3,287 | 11.96 / 9.03 | 43.1 / 38.9 | 11.89 / 9.48 | 28.63 / 25.53 |

At load 1 the engines are indistinguishable except for a constant 0.2 ms per
token, the per-step overhead outside the graph. Past load 1 lean-vLLM trails
on every metric. Median TPOT is 1.11–1.29× vLLM's. The TTFT tail is worst at
loads 24–32, where p99 TTFT is 3.3–3.7× vLLM's, and narrows to 1.3–1.5× at the
plateau, where both engines queue.

Async scheduling helps latency as much as throughput. Against the off curve at
load 48 it cuts p50 TTFT from 3.41 s to 2.11 s and p99 TPOT from 132 ms to
107 ms. At load 24 it cuts p50 TPOT by a third, from 21.5 ms to 14.4 ms.

Load 32 with async scheduling on is the outlier. Its p50 TTFT (452 ms) is
worse than the off curve's (250 ms), and 55% of its inter-token gaps are under
1 ms: tokens often reach the client in pairs. The median of the remaining gaps
is 28.9 ms, against 14.3 ms for vLLM at that load. No other lean-vLLM on point
bunches more than 17% of its gaps. vLLM bunches 28% at loads 48 and 64, so
bunched delivery under load is not unique to lean-vLLM.

## 3. GPU utilization and clock conditions

Sampled once per second and aligned to each run's recorded start and finish:

| Offered load (req/s) | util off | util on | util vLLM | power off (W) | power on (W) | power vLLM (W) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67% | 70% | 70% | 337 | 341 | 342 |
| 24 | 82% | 97% | 98% | 542 | 613 | 598 |
| 32 | 87% | 98% | 97% | 608 | 618 | 613 |
| 48 | 82% | 93% | 97% | 597 | 618 | 616 |
| 64 | 86% | 92% | 97% | 597 | 609 | 610 |

With async scheduling on, lean-vLLM keeps the GPU as busy as vLLM up to load
32 and draws the same power at every load. At loads 48 and 64 its utilization
is 3–5 points below vLLM's, in line with the 5–7% throughput gap.

| Offered load (req/s) | mean SM clock on (MHz) | mean SM clock vLLM (MHz) | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 1,976 | 1,975 | +0.1% |
| 24 | 1,874 | 1,869 | +0.3% |
| 32 | 1,771 | 1,787 | −0.9% |
| 48 | 1,750 | 1,745 | +0.3% |
| 64 | 1,747 | 1,710 | +2.2% |

Clocks are averaged over busy samples. Under load both engines drop below the
1,980 MHz lock as `SwPowerCap` engages at the 700 W limit; no other throttle
reason appeared, and the GPU peaked at 61 °C. At load 64 lean-vLLM held the
higher clock and still trailed, so clock conditions do not explain the gap.

## 4. Where the step time goes

### Offline: pipelining removes the GPU idle

The offline CUDA trace records device activity, so it gives a true busy/idle
split. It is an offline `generate` on Qwen3-0.6B: 256 sequences with inputs and
outputs of 100–1,024 tokens, 200 steps captured after 200 skipped. GPU-busy
time is the union of all kernel and copy intervals over the captured window.

| | Async off | Async on |
| --- | ---: | ---: |
| Window, 200 steps | 2,321 ms | 1,877 ms |
| Step time | 11.6 ms | **9.4 ms** |
| Kernel-busy time | 1,798 ms | 1,812 ms |
| FlashAttention share of kernel time | 79.7% | 79.6% |
| GPU busy | 77.6% | **96.8%** |
| GPU idle | 22.4% | **3.2%** |
| Host `await_tokens` per step | 6.27 ms | 3.70 ms |
| Host `launch` per step (schedule excluded) | 2.83 ms | 3.02 ms |
| Host `detokenize` per step | 1.90 ms | 2.05 ms |

The device work is the same in both arms. The step is 19% faster with async
scheduling on only because batch preparation and launch now happen while the
GPU is still running the previous step, so the host blocks for less time in
`await_tokens`. At 3.2% idle there is little left to win offline from further
host overlap.

### Online at load 48: large eager steps dominate

lean-vLLM's server counters break engine step time down by how each step ran.
Figures are for the measured run, with warmup subtracted:

| | load 24, on | load 48, off | load 48, on |
| --- | ---: | ---: | ---: |
| Steps | 4,096 | 2,149 | 2,152 |
| Eager prefill steps (outside 64–512 tokens) | 410 (10.0%) | 390 (18.1%) | 385 (17.9%) |
| Mean eager prefill step | 33.5 ms | 58.2 ms | **47.5 ms** |
| Mean full or piecewise graph step | 8.8 ms | 9.4 ms | 9.7 ms |
| Eager share of step time | 29.8% | 57.8% | **51.6%** |

At the plateau, the steps that run eager take half the engine's time. Async
scheduling shortens them by 18%, from 58.2 ms to 47.5 ms, because their host
dispatch overlaps the previous step. The graph steps stay near 9.5 ms.

vLLM captures the same 512-token range but compiles every step size with
inductor, so its large steps get fused kernels instead of running eager. That
difference is the leading candidate for the plateau gap. It fits the data:
the engines match at load 1, where eager steps are 1.8% of step time, and at
load 24, where they are 30%, and diverge only where they pass half. This run
does not isolate it, though.

### Online host trace at load 48

A CPU-only step-loop trace was captured for one load-48 run per arm, 600 steps
each ([`profile-48/`](../results/20260913T101307Z/profile-48/)). The profiler
is expensive here. It cut goodput by 14% with async scheduling on (20.93 vs
24.45 requests/s) and by 19% with it off. The traced step reads 34.9 ms against
the 16.3 ms the counters report for the unprofiled run. Only the proportions
below are meaningful, not the absolute times:

| Host phase, mean per step | Async off | Async on |
| --- | ---: | ---: |
| `schedule` | 0.27 ms | 0.26 ms |
| `launch` | 22.46 ms | 18.45 ms |
| ↳ `prepare_batch` | 3.44 ms | 3.17 ms |
| ↳ `run_model` | 18.39 ms | 14.82 ms |
| `await_tokens` | 8.20 ms | 5.67 ms |
| `reconcile` | 0.40 ms | 0.26 ms |
| `detokenize` | 3.59 ms | 3.31 ms |
| Traced step | 41.8 ms | 34.9 ms |

The host's own work is dominated by `run_model`, the dispatch of the forward
pass. Its median is 6.3 ms with async scheduling on, but steps above 10 ms
account for 96% of its total: a minority of op-heavy steps carries the host
cost. The profiler inflates exactly those steps, so this points the same way
as the counters above but does not size the effect.
`prepare_batch` (about 3 ms) and `detokenize` (about 3 ms) are steady and
already overlap the GPU.

## Measurement limits

- **One run per point.** No point was repeated, so differences of a few percent
  between neighbouring loads, such as the weak async gain at load 32, are within
  run-to-run noise until repeated.
- **The GPU busy/idle split is offline, on Qwen3-0.6B.** CUPTI cannot collect
  device activity from the online engine thread. The online 8B idle is inferred
  from sampled utilization, not measured.
- **The online host trace distorts what it measures.** In-process
  `record_function` tracing cost 14–19% of goodput and roughly doubled the
  traced step, and it inflates eager steps most because they dispatch the most
  ops.
- **vLLM exposes no server counters here.** The step breakdown by kind is
  lean-vLLM only; vLLM's large-step cost is inferred from its configuration,
  not measured.
- **vLLM is not the latest release.** 0.26.0 is the last release that pins
  torch 2.11, which the FlashAttention-3 wheel links against; matching torch
  was chosen over a newer vLLM.

## Next experiments, in priority order

1. **Cut the cost of steps above 512 tokens.** They are half the engine time at
   the plateau. Either extend capture to larger token buckets, weighing the
   padding tax recorded in
   [`docs/online-serving.md`](../docs/online-serving.md#cuda-graphs), or
   compile the non-attention pieces for fusion. Then rerun loads 48 and 64.
2. **Repeat load 32, and repeat every plateau point.** This confirms or
   dismisses the load-32 TTFT and paired-token anomaly, and puts error bars on
   the 5–7% gap.
3. **Profile the online loop with less overhead.** Record per-phase timings
   with `perf_counter` counters instead of `record_function`, so the online
   host split can be read without the 14–19% distortion.
