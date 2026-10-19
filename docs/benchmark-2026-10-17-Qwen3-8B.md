# Qwen3-8B serving benchmark, 17 October 2026

lean-vLLM against vLLM 0.26.0 on Qwen3-8B, one H100, with chunked prefill,
full + piecewise CUDA graphs, and Inductor compilation on both engines.
lean-vLLM ran two curves, async scheduling off and on; vLLM ran its default,
which is on. Results are in
[`results/20261017T175202Z-Qwen3-8B/`](../results/20261017T175202Z-Qwen3-8B/).

All fifteen serving runs completed: five offered loads per curve, 1,000 requests
each, no rejections, failures, or preemptions. Every run replayed the same trace
of 659,770 prompt tokens and 176,838 generated tokens.

This run follows the 3 October report, which is retired and in the git history.
lean-vLLM moved from `06e2725` to `cac48a2`: among other changes the engine
core now runs in its own process behind a ZMQ client, CUDA graphs are captured
at vLLM's default sizes, and step inputs are prepared with numpy into
persistent buffers.

The main findings:

- The engines are at parity across the curve. lean-vLLM with async
  scheduling on is within 0.3% of vLLM's goodput up to load 32, 0.6% behind at
  load 48 and 1.7% behind at load 64, down from 3.7–3.9% on 3 October.
- Load-1 TPOT is level. lean-vLLM's median is 6.7 ms against vLLM's 6.8 ms;
  on 3 October it was 0.2 ms above.
- Async scheduling is worth 2.3–7.4%. It adds 2.3% at load 24, 4.8% at
  load 32, 7.4% at load 48 and 6.1% at load 64 over the off curve, and lifts GPU
  utilization under load from 89–91% to 96–99%, the same as vLLM.
- Offline, pipelining removes most GPU idle, but vLLM's step is still shorter.
  The CUDA trace shows the GPU idle 22.5% of the step window with async
  scheduling off and 3.3% with it on. vLLM, traced on the same workload, idles
  1.4% and spends 13% less kernel time per step.
- Latency tracks vLLM. Median TPOT is 1.01–1.08× vLLM's from load 24 up,
  p99 TTFT is lower than vLLM's at loads 24–32 and 1.09–1.13× at the plateau.

## Setup

### Hardware and software

| Component | Value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3, 81,559 MiB |
| NVIDIA driver | 580.126.09 |
| Linux kernel | `6.8.12-680-6063-coreweave-amd64-f81899c8` |
| Python | 3.12.11 |
| PyTorch, both engines | `2.11.0+cu130` |
| lean-vLLM | commit `cac48a2` on `refactor/code-cleanup`; working tree adds the pinned-buffer fix below and `uv.lock` mirror URLs |
| vLLM | 0.26.0 |
| Attention | FlashAttention-3 on both engines |
| Persistence mode | Enabled |
| Application graphics clock | 1,980 MHz, equal to the maximum; not locked (the container lacks permission) |
| Power limit | 700 W (maximum 700 W) |

At `cac48a2` the lean-vLLM server did not start: `InputBuffers` allocated its
pinned host buffers while `ModelRunner.__init__` had set CUDA as the default
device, and warmup failed with "Only dense CPU tensors can be pinned". The run
used a one-line fix that allocates them on the CPU
(`lean_vllm/engine/input_buffers.py`); the patch is in `working-tree.patch`.

Before the sweep, greedy generation was compared with vLLM on 8 prompts × 64
tokens: 6 of 8 matched exactly, and both divergences were near-ties that vLLM's
own logprobs put within 0.13 nats. The notes are in `validation/`.

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

Every vLLM server log confirmed async scheduling was on
([`vllm-async-scheduling.txt`](../results/20261017T175202Z-Qwen3-8B/vllm-async-scheduling.txt)),
so the like-for-like pair is lean-vLLM's `async-scheduling=True` curve against
vLLM. vLLM also ran with prefix caching on, its default. lean-vLLM recorded no
prefix-cache hits on this trace, so prefix caching should not favour either
engine.

### How the two engines execute a step

| | lean-vLLM | vLLM 0.26.0 |
| --- | --- | --- |
| Process layout | API server and engine core in separate processes, ZMQ between them | Same |
| Compilation | Inductor piecewise: the model is traced whole, split at attention ops, and each piece is compiled by Inductor | Inductor piecewise, same approach |
| Decode batches | Full CUDA graphs, 35 sizes up to 256 | Full CUDA graphs, 35 sizes |
| Prefill and mixed steps up to 512 tokens | Inductor-compiled piecewise graphs, 51 sizes (1, 2, 4, every 8 to 248, every 16 to 512) | Same 51 sizes |
| Prefill and mixed steps above 512 tokens | Inductor-compiled, no graph | Inductor-compiled, no graph |
| Async scheduling | Launch step k, then await and reconcile step k-1 | On by default |

Both engines now capture the same graph sizes, so the 3 October explanation for
the plateau gap, 13 piecewise sizes against 51, no longer applies.

With async scheduling off, lean-vLLM awaits step k-1's tokens before scheduling
step k. With it on, scheduling and batch preparation also run ahead of the await
([`docs/online-serving.md`](online-serving.md#pipelined-steps)).
Detokenization now runs in the API server process, outside the engine-core step
loop, on both engines.

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
| 24 | 18.97 | 19.41 | 19.36 | +0.3% | +2.3% |
| 32 | 21.97 | 23.03 | 23.05 | −0.1% | +4.8% |
| 48 | 23.76 | 25.52 | **25.68** | −0.6% | +7.4% |
| 64 | 24.10 | 25.58 | **26.02** | −1.7% | +6.1% |

Goodput is in requests/s. Output token throughput follows goodput, since every
run generates the same tokens: at load 48 and 64 lean-vLLM on produces 4,512
and 4,524 tok/s, lean-vLLM off 4,203 and 4,262, and vLLM 4,541 and 4,601.

All three curves level off from load 48. lean-vLLM on plateaus at 25.5–25.6
requests/s, off at 23.8–24.1, and vLLM at 25.7–26.0, still rising by 1.3% from
48 to 64.

Compared with 3 October, lean-vLLM on gained 2.9% at load 48 (24.81 → 25.52)
and 2.1% at load 64 (25.06 → 25.58), while vLLM's numbers held within 0.3%. The
plateau gap narrowed from 3.7–3.9% to 0.6–1.7%, which is close to the
run-to-run noise of a single run per point.

## 2. Latency

| Offered load (req/s) | p50 TTFT on / vLLM (ms) | p99 TTFT on / vLLM (s) | p50 TPOT on / vLLM (ms) | p50 E2E on / vLLM (s) | p99 E2E on / vLLM (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28 / 28 | 0.10 / 0.10 | 6.7 / 6.8 | 0.88 / 0.89 | 6.22 / 6.36 |
| 24 | 77 / 77 | 0.31 / 0.40 | 12.9 / 12.8 | 1.76 / 1.75 | 11.49 / 11.54 |
| 32 | 160 / 157 | 0.79 / 0.88 | 20.6 / 19.1 | 2.92 / 2.73 | 17.06 / 16.18 |
| 48 | 1,375 / 1,232 | 5.24 / 4.65 | 39.6 / 37.4 | 7.26 / 6.76 | 25.58 / 24.68 |
| 64 | 3,634 / 3,358 | 9.93 / 9.08 | 40.6 / 39.1 | 10.09 / 9.53 | 26.42 / 25.57 |

At load 1 the engines are indistinguishable: TTFT, TPOT and E2E all match
within 0.1 ms or 3%. On 3 October lean-vLLM carried a constant 0.2 ms per token
here. Past load 1, median TPOT is 1.01–1.08× vLLM's, down from 1.05–1.12×.
lean-vLLM's p99 TTFT is lower than vLLM's at loads 24–32 (0.31 and 0.79 s
against 0.40 and 0.88 s) and 1.09–1.13× at the plateau, down from 1.17–1.27×.

Async scheduling helps latency too. Against the off curve at load 48 it cuts
p50 TTFT from 1.67 s to 1.38 s and p99 TTFT from 6.66 s to 5.24 s. At load 24
it cuts p50 TPOT by 17%, from 15.5 ms to 12.9 ms.

## 3. GPU utilization and clock conditions

Sampled once per second and aligned to each run's recorded start and finish:

| Offered load (req/s) | util off | util on | util vLLM | power off (W) | power on (W) | power vLLM (W) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65% | 69% | 70% | 332 | 347 | 344 |
| 24 | 89% | 98% | 98% | 590 | 617 | 595 |
| 32 | 89% | 97% | 98% | 607 | 622 | 616 |
| 48 | 89% | 96% | 97% | 607 | 622 | 615 |
| 64 | 91% | 99% | 99% | 604 | 628 | 622 |

With async scheduling on, lean-vLLM keeps the GPU as busy as vLLM at every load,
within one point. On 3 October it was 4–5 points below vLLM at loads 48–64.

| Offered load (req/s) | mean SM clock on (MHz) | mean SM clock vLLM (MHz) | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 1,976 | 1,976 | +0.0% |
| 24 | 1,878 | 1,839 | +2.1% |
| 32 | 1,791 | 1,809 | −1.0% |
| 48 | 1,725 | 1,721 | +0.2% |
| 64 | 1,763 | 1,709 | +3.2% |

Clocks are averaged over busy samples. The clock could not be locked this
session, but the application clock already sits at the 1,980 MHz maximum. Under
load both engines drop below it as `SwPowerCap` engages at the 700 W limit; no
other throttle reason appeared, and the GPU peaked at 66 °C. At load 64
lean-vLLM held a 3% higher clock and still trailed by 1.7%, so the gap there may
be slightly wider at equal clocks. At load 48 the clocks match within 0.2%.

## 4. Where the step time goes

### Offline: pipelining removes the GPU idle, and vLLM runs less kernel time

The offline CUDA trace records device activity, so it gives a true busy/idle
split. It is an offline `generate` on Qwen3-0.6B: 256 sequences with inputs and
outputs of 100–1,024 tokens, 200 steps captured after 200 skipped. vLLM ran the
same workload and seed through its own profiler, with the same skip and capture
counts (`bench_offline_vllm.py` in the results directory). For all three the
window runs from the first captured step's host work to the last device or host
event, and GPU-busy time is the union of all kernel and copy intervals in it.

| | lean-vLLM, async off | lean-vLLM, async on | vLLM |
| --- | ---: | ---: | ---: |
| Window, 200 steps | 2,360 ms | 1,903 ms | **1,619 ms** |
| Step time | 11.8 ms | 9.5 ms | **8.1 ms** |
| Kernel time | 1,825 ms | 1,836 ms | 1,598 ms |
| Kernels per step | 376 | 376 | 358 |
| FlashAttention share of kernel time | 79.4% | 79.2% | 81.6% |
| GPU busy | 77.5% | 96.7% | **98.6%** |
| GPU idle | 22.5% | 3.3% | 1.4% |
| Host `await_tokens` per step | 7.82 ms | 3.95 ms | — |
| Host `launch` per step | 2.66 ms | 3.67 ms | — |

lean-vLLM's device work is the same in both arms; async scheduling shortens the
step by 19% only because batch preparation and launch now overlap the previous
step's GPU work.

Against vLLM, lean-vLLM's step is 17% longer. About 2 points of that is idle
(3.3% against 1.4%); the rest is kernel time, 1,836 ms against 1,598 ms (+15%).
FlashAttention kernels account for 1,454 ms against vLLM's 1,304 ms, and the
remaining kernels for 382 ms against 294 ms. Both engines skipped 200 steps of
the same workload, but they schedule independently, so the captured steps need
not hold identical batches; read the kernel comparison as indicative. vLLM's
wall-clock offline throughput is not comparable here because its profiler runs
for the whole `generate`.

### Online at load 48: compiled prefill steps still take close to half

lean-vLLM's server counters break engine step time down by how each step ran.
Figures are for the measured run, with warmup subtracted:

| | load 24, on | load 48, off | load 48, on | load 64, on |
| --- | ---: | ---: | ---: | ---: |
| Steps | 4,512 | 2,210 | 2,246 | 2,157 |
| Compiled prefill steps (above 512 tokens) | 436 (9.7%) | 399 (18.1%) | 369 (16.4%) | 391 (18.1%) |
| Mean compiled prefill step | 23.0 ms | 52.7 ms | **47.4 ms** | 48.0 ms |
| Mean full or piecewise graph step | 10.0 ms | 11.3 ms | 11.3 ms | 11.1 ms |
| Compiled share of step time | 19.7% | 50.7% | **45.3%** | 48.9% |

These match 3 October closely (load 48 on: 369 compiled steps, 45.1 ms each,
43.3% of step time). Both engines compile these large steps with Inductor and
neither captures them, so they are now the same kind of work on both sides.
Async scheduling shortens them by 10%, from 52.7 ms to 47.4 ms, because their
host dispatch overlaps the previous step.

### Online host trace at load 48

A CPU-only step-loop trace was captured for one load-48 run per arm, 600 steps
each ([`profile-48/`](../results/20261017T175202Z-Qwen3-8B/profile-48/)). The
profiler cut goodput by 11% with async scheduling on (22.60 vs 25.52
requests/s) and by 13% with it off (20.65 vs 23.76). Only the proportions below
are meaningful, not the absolute times:

| Host phase, mean per step | Async off | Async on |
| --- | ---: | ---: |
| `schedule` | 0.49 ms | 0.57 ms |
| `launch` | 16.33 ms | 18.48 ms |
| ↳ `prepare_batch` | 1.63 ms | 1.87 ms |
| ↳ `run_model` | 14.01 ms | 15.85 ms |
| ↳ `sample` | 0.07 ms | 0.08 ms |
| `await_tokens` | 20.31 ms | 16.12 ms |
| `reconcile` | 0.51 ms | 0.52 ms |
| Traced step | 38.6 ms | 36.8 ms |

`prepare_batch` dropped from 6.73 ms to 1.87 ms per step with async scheduling
on since 3 October, now that inputs are built with numpy into persistent
buffers. `detokenize` no longer appears: it moved to the API server process
with the engine core split, where it took about 1.5 ms per step on 3 October.
The host's own work is now dominated by `run_model`, the dispatch of the forward
pass, and `await_tokens` is time the host spends waiting on the GPU.

## Measurement limits

- One run per point. No point was repeated, so differences of a few percent
  between neighbouring loads, including the 0.6–1.7% plateau gap, are within
  run-to-run noise until repeated.
- Clocks were not locked. The container could not set `-lgc`; both engines
  ran at the default application clock, which is the maximum, and busy-sample
  clocks are reported per load instead.
- Shared host. Other users' jobs ran on other GPUs of the same node during the
  session, sharing host CPU and memory bandwidth with both engines.
- The GPU busy/idle split is offline, on Qwen3-0.6B. CUPTI cannot collect
  device activity from the online engine process. The online 8B idle is
  inferred from sampled utilization, not measured.
- The offline traces do not hold identical batches. Both engines skip and
  capture the same number of steps of the same workload, but schedule
  independently.
- The online host trace distorts what it measures. In-process
  `record_function` tracing cost 11–13% of goodput.
- vLLM exposes no step counters here. The step breakdown by kind is
  lean-vLLM only.
- vLLM is not the latest release. 0.26.0 is the last release that pins
  torch 2.11, which the FlashAttention-3 wheel links against; matching torch
  was chosen over a newer vLLM.

## Next experiments, in priority order

1. Repeat every plateau point. The remaining 0.6–1.7% gap is within the
   noise of one run; three runs per point would say whether it is real.
2. Find the offline kernel-time gap. lean-vLLM spends 15% more kernel time
   than vLLM over 200 offline steps, in FlashAttention and in the other
   kernels. Capture both engines on a fixed batch, so the steps match, and
   compare kernel by kernel.
3. Profile the online loop with less overhead. Record per-phase timings
   with `perf_counter` counters instead of `record_function`, so the online
   host split can be read without the 11–13% distortion.
