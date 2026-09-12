# Online serving benchmark — 12 September 2026

First back-to-back run on an NVIDIA H100 80GB HBM3 with FlashAttention-3, using
Qwen3-8B. Both engines now run the same PyTorch build, and the GPU clock was
locked at its maximum, so this is the cleanest comparison so far. It supersedes
the [11 September report](benchmark-2026-09-11.md), which ran on an A100 with
FlashAttention-2 and mismatched torch versions. Results are in
[`results/20260912T105854Z/`](../results/20260912T105854Z/).

All 18 runs completed — nine offered loads per engine, 1,000 requests each —
with no rejections, failures, or preemptions. Each run replayed the identical
trace: 659,770 prompt tokens and 176,838 generated tokens.

lean-vLLM ran at commit `db8e091` (install a prebuilt FA3 wheel, against torch
2.11). vLLM moved back to 0.26.0, the last release pinning torch 2.11, so that
both engines could share one build. Both used `FULL_AND_PIECEWISE` graph mode
and a 16-token KV block.

A follow-up sweep at offered loads of 32, 48, and 64 extended the same session
and found both plateaus.  Its numbers are marked † throughout.

The main findings are:

- **Both engines plateau, and the gap between them is 25%.** lean-vLLM tops out
  at 19.6 requests/s, vLLM at 26.0. The A100 sweep put the same gap at 5.7–7.0%.
- **The hardware move helped vLLM more.** Plateau throughput rose 2.24× for
  lean-vLLM against the A100 and 2.77× for vLLM. Faster kernels expose
  lean-vLLM's fixed per-step cost rather than hiding it.
- **The deficit is small until saturation.** lean-vLLM trails by 0.7% or less up
  to 8 requests/s and by about 2% at 12 and 16. It reaches 7.0% at 24, 16% at
  32, and 24–25% from 48 upward.
- **The GPU sits idle inside lean-vLLM's step time.** Sampled utilization is
  67–91% under load against a flat 100% for vLLM, and lean-vLLM draws 49–65 W
  less from load 12 upward. The deficit is host-side gaps, not slower kernels.
- **Eager steps no longer dominate.** They hold 42% of step time at load 24,
  against 63–69% at saturation on the A100. The 512-token capture cap costs much
  less on this hardware.
- **vLLM caps piecewise capture at 512 tokens too.** Its logged
  `max_cudagraph_capture_size` is 512, matching lean-vLLM's
  `PIECEWISE_MAX_TOKENS`. The 11 September report flagged this as unknown.

## Setup and metric definitions

### Hardware and runtime

| Component | Value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 |
| GPU memory | 81,559 MiB |
| NVIDIA driver | 580.126.09 |
| Linux kernel | `6.8.12-680-6063-coreweave-amd64-f81899c8` |
| Python | 3.12.11 |
| PyTorch, both engines | `2.11.0+cu130` |
| vLLM | 0.26.0 |
| Attention | FlashAttention-3 on both engines |
| System time at metadata collection | 12 September 2026, 11:01:36 UTC (`+0000`) |
| Persistence mode | Enabled |
| Application graphics clock | 1,980 MHz, equal to the maximum |
| Power limit / maximum power limit | 700.00 W / 700.00 W |

The clock was locked at the maximum, which the A100 host did not allow. Both
engines ran the same torch build, removing the largest measurement limit of the
11 September batch.

### Workload and server settings

| Setting | Value |
| --- | --- |
| Requests per run | 1,000, plus 3 warmup requests |
| Workload | Lognormal lengths: input parameter 512, output parameter 128, σ = 0.8 |
| Sampling | Seed 0, greedy decoding |
| Server limits | `--max-num-batched-tokens 8192 --max-num-seqs 256` on both engines |
| Chunked prefill | Enabled on both engines |
| Graph mode | `full_and_piecewise` on both engines |
| KV block size | 16 tokens on both engines |
| KV cache capacity | 327,680 tokens |
| Maximum model length | 4,096 |

vLLM enables prefix caching by default and lean-vLLM does not, but both
measured a 0.0% hit rate on this trace, so the asymmetry had no effect.

### Reading the results

| Metric | Meaning | Better direction |
| --- | --- | --- |
| Offered load | Target request arrival rate, in requests/s | Test input |
| Goodput | Completed requests divided by total run time, including time to drain the queue; no latency cutoff is applied | Higher |
| TTFT | Time to first token: how long a request waits before output begins | Lower |
| TPOT | Time per output token after the first token | Lower |
| E2E | End-to-end time to complete a request | Lower |
| p50 / p99 | Median / 99th percentile; p99 describes the slow tail | Lower for latency |

Because goodput counts the drain after the last arrival, it stays a few percent
below the offered load even when the engine keeps up. Read it against the other
engine at the same load, not against the offered rate.

## 1. Throughput: parity below 12 req/s, a 25% gap at the plateau

| Offered load (req/s) | Goodput, lean-vLLM (req/s) | Goodput, vLLM (req/s) | Difference | Output tok/s, lean-vLLM | Output tok/s, vLLM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.01 | 1.01 | −0.0% | 179 | 179 |
| 4 | 3.97 | 3.98 | −0.2% | 702 | 704 |
| 8 | 7.57 | 7.63 | −0.7% | 1,339 | 1,349 |
| 12 | 10.77 | 11.00 | −2.1% | 1,904 | 1,946 |
| 16 | 13.77 | 14.09 | −2.3% | 2,435 | 2,492 |
| 24 | 18.04 | 19.39 | −7.0% | 3,190 | 3,429 |
| 32 † | 19.39 | 23.09 | −16.0% | 3,429 | 4,083 |
| 48 † | 19.55 | 25.72 | −24.0% | 3,458 | 4,548 |
| 64 † | 19.56 | **25.97** | −24.7% | 3,459 | **4,592** |

lean-vLLM flattens hard: 19.39, 19.55, 19.56 across the last three loads, a
spread under 1%. It was already within 9% of that ceiling at load 24, which is
why the original grid read as a modest deficit. vLLM keeps climbing to load 48
and settles at 25.97.

The deficit therefore grows with load rather than shrinking. Against the A100,
lean-vLLM's plateau improved by 2.24× and vLLM's by 2.77×. FlashAttention-3 and
Hopper made the model work cheaper for both engines, and the engine that had
less fixed overhead to begin with converted more of it.

## 2. Latency: lean-vLLM queues from 24 req/s, vLLM not until 32

### End-to-end and queue state

| Offered load (req/s) | p50 E2E, lean-vLLM (s) | p50 E2E, vLLM (s) | p99 E2E, lean-vLLM (s) | p99 E2E, vLLM (s) | Waiting peak, lean-vLLM | KV peak, lean-vLLM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.96 | 0.88 | 6.99 | 6.23 | 0 | 3% |
| 4 | 1.08 | 0.98 | 7.79 | 7.04 | 0 | 5% |
| 8 | 1.28 | 1.08 | 9.04 | 7.64 | 0 | 9% |
| 12 | 1.71 | 1.18 | 11.33 | 8.00 | 0 | 12% |
| 16 | 1.93 | 1.33 | 13.22 | 8.95 | 1 | 17% |
| 24 | **3.97** | 1.77 | **23.17** | 11.48 | 11 | 44% |

Median end-to-end latency doubles for lean-vLLM between 16 and 24 requests/s
while vLLM's rises by a third. The waiting queue is the confirmation: it is
empty through load 12 and reaches 11 at load 24. KV cache never came close to
full, so this is compute, not capacity.

### Time to first token

| Offered load (req/s) | p50 TTFT, lean-vLLM (ms) | p50 TTFT, vLLM (ms) | p99 TTFT, lean-vLLM (ms) | p99 TTFT, vLLM (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | **28** | 29 | 125 | **105** |
| 4 | **30** | 32 | 141 | **133** |
| 8 | **33** | 36 | 230 | **140** |
| 12 | 50 | **44** | 429 | **163** |
| 16 | **47** | 52 | 524 | **179** |
| 24 | 108 | **80** | 878 | **271** |

Below saturation the multi-second p99 TTFT that the A100 batch measured at the
knee is gone: lean-vLLM's worst here is 0.88 seconds against 6.27 on the A100 at
load 12. The ratio between the engines is still 2–3× from load 8 upward, but the
absolute tail is small enough that no request waits noticeably for its first
token. It returns past the plateau, below.

### Time per output token

| Offered load (req/s) | p50 TPOT, lean-vLLM (ms) | p50 TPOT, vLLM (ms) | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 7.2 | 6.7 | +8% |
| 4 | 8.2 | 7.4 | +10% |
| 8 | 9.6 | 8.1 | +19% |
| 12 | 12.2 | 8.8 | +38% |
| 16 | 14.2 | 9.9 | +43% |
| 24 | 28.7 | 12.8 | +124% |

TPOT is where the two engines separate most. At load 1, lean-vLLM's mean
graph-replaying step is 7.11 ms and its p50 TPOT is 7.2 ms, so a pure decode
step accounts for the whole figure; the 0.5 ms gap to vLLM is constant per-step
overhead outside the graph. From load 8 upward the gap widens because decode
tokens wait behind large prefill steps, and at load 24 they also wait behind the
queue.

### Past saturation

| Offered load (req/s) | p99 TTFT, lean-vLLM (s) | p99 TTFT, vLLM (s) | p50 TPOT, lean-vLLM (ms) | p50 TPOT, vLLM (ms) | p99 E2E, lean-vLLM (s) | p99 E2E, vLLM (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 † | 4.25 | **0.71** | 53.4 | **19.4** | 32.37 | **16.20** |
| 48 † | 13.85 | **4.59** | 56.7 | **38.4** | 36.43 | **24.98** |
| 64 † | 18.97 | **9.22** | 56.3 | **39.2** | 37.33 | **25.67** |

Both engines queue here, as they must past their plateaus, and both latency
curves flatten once the queue is the dominant term. lean-vLLM settles near
56 ms per output token against vLLM's 39 ms. Its p99 end-to-end latency is
twice vLLM's at load 32 and about 1.45× at loads 48 and 64. No run rejected or
failed a request at any load.

## 3. Where the time went: idle GPU, not slow kernels

lean-vLLM's server counters show the model busy 89–98% of wall time at every
load above 1. That counter times the whole step — scheduling, batch preparation,
the forward pass, sampling, and detokenization — not the forward pass alone.
Sampled GPU utilization shows how much of it reaches the device:

| Offered load (req/s) | Mean GPU utilization, lean-vLLM | Mean GPU utilization, vLLM | Mean power, lean-vLLM (W) | Mean power, vLLM (W) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 57% | 62% | 324 | 326 |
| 4 | 91% | 97% | 443 | 461 |
| 8 | 88% | **100%** | 476 | 503 |
| 12 | 67% | **100%** | 493 | 556 |
| 16 | 82% | **100%** | 539 | 588 |
| 24 | 71% | **100%** | 585 | 650 |

vLLM saturates the GPU from load 8 onward. lean-vLLM never does, and the power
draw follows: 65 W less at load 24. Taking load 24 as the example, the run lasted
55.4 seconds, the counters attribute 49.9 seconds to steps, and 71% utilization
implies roughly 39 seconds of kernel activity. About 10 seconds of counted step
time had no kernel running, and the step loop is serial, so none of that host
work overlaps the next step's GPU work.

This is a different diagnosis from the A100 batch, where the deficit was
concentrated in a few expensive eager steps. Those steps are now a minority of
the cost:

| Offered load (req/s) | Steps | Eager steps | Share of steps | Share of step time | Mean eager step (ms) | Mean graph-replaying step (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96,696 | 490 | 0.5% | 2.3% | 33.5 | 7.11 |
| 4 | 30,893 | 482 | 1.6% | 6.0% | 30.7 | 7.62 |
| 8 | 13,867 | 475 | 3.4% | 11.9% | 31.8 | 8.38 |
| 12 | 7,866 | 453 | 5.8% | 20.1% | 38.5 | 9.35 |
| 16 | 5,592 | 458 | 8.2% | 24.9% | 36.6 | 9.82 |
| 24 | 2,914 | 397 | 13.6% | 42.2% | 53.1 | 11.47 |

The eager step count is nearly constant near 480 across the whole sweep, which
is what the trace implies: roughly one prefill step above 512 tokens per two
requests. What changes is the denominator. FlashAttention-3 made every step
cheaper, so the fixed eager work occupies a growing share as load rises, but at
42% it is well below the 63–69% the A100 measured and well below the ceiling
the [piecewise CUDA graphs doc](../docs/piecewise-cuda-graphs.md) predicted.

Two conclusions follow. Capturing the large mixed steps would recover less here
than the doc's estimate assumed, and the launch gaps visible in the utilization
data are the larger target.

The counters and GPU log cover only the first six loads, so this accounting
stops short of the plateau. The trend it shows — a widening idle fraction and a
rising eager share — points the same way as the 25% plateau gap, but the
attribution at the plateau itself is not measured.

### What the idle host time is: input prep and detokenization

To name the host work behind the idle, the step loop was profiled in process
with `torch.profiler`. `record_function` ranges label six phases — `schedule`,
`prepare_batch`, `run_model`, `sample`, `postprocess`, `detokenize` — and a
window of steps is captured after warmup and the load ramp have passed. nsys is
unavailable on this box, which carries no CUDA toolkit by design, so this
replaces the Nsight Systems run the previous batch called for.

Two traces were taken. The first is the plateau itself: a load-48 online run,
600 steps of steady-state decode at 10.0 ms/step. It is CPU-side only — the
engine steps on a worker thread while CUDA was initialised on the main thread,
and kineto refuses to collect device activity across that boundary. The host
phases still read cleanly:

| Phase | Share of step | Per step (µs) |
| :--- | ---: | ---: |
| `sample` (the `.tolist()` device→host sync) | 79.8% | 7,989 |
| `prepare_batch` | 5.5% | 548 |
| `run_model` (kernel launch only) | 5.5% | 547 |
| inter-step loop overhead (untraced) | 5.5% | 545 |
| `detokenize` | 2.5% | 246 |
| `postprocess` | 0.6% | 61 |
| `schedule` | 0.3% | 34 |

The `sample` phase is misleadingly large. Drilling in, 99.3% of it is
`aten::copy_` under the `.tolist()` that pulls sampled tokens back to the host.
That call is a full device sync: `run_model` only launches the forward kernels
asynchronously and returns, so the CPU blocks in `sample` until the GPU drains.
On a CPU-only trace the entire forward-pass GPU time is therefore absorbed into
`sample`, and busy cannot be split from idle here. What the trace does measure is
the genuinely host-serial work — everything except `sample` — and it is spread
thin: `prepare_batch`, `run_model`'s launch overhead, and the async-engine loop
glue between steps each take about 0.55 ms, `detokenize` half that. No single
host phase dominates.

The second trace recovers the busy/idle split the online run could not. It is an
offline `generate` run (256 sequences, 200 steps at 13.1 ms/step), which steps on
the main thread, so CUPTI collects device activity. Taking the union of all
kernel and copy intervals as GPU-busy time:

| | Time (ms) | Share of step window |
| :--- | ---: | ---: |
| GPU busy | 1,713 | 65.6% |
| GPU idle | 898 | **34.4%** |

Even in a saturated big-batch throughput run the GPU is idle 34% of the wall
window, at the top of the 9–33% range the utilization log implied for the online
loads. The device work itself is unremarkable: FlashAttention is 79% of the busy
time, the sampler and the GEMMs the rest, nothing pathological. The idle is host
work that does not overlap the forward pass, and on this workload the two largest
host phases are `detokenize` (1.72 ms/step, 13.2% of the window) and
`prepare_batch` (1.23 ms/step, 9.4%) — both scale with batch size, which is why
they read larger here than on the smaller online decode batches. Together they
account for roughly 3 ms of the 13 ms step, serial, with the GPU stalled.

Two conclusions follow, both consistent with section 3's utilization data.
Overlapping `prepare_batch` and `detokenize` with the forward pass — preparing
the next step's batch and draining detokenization off the critical path — is the
larger lever, targeting the 34% idle directly. CUDA graphs would remove
`run_model`'s launch bubbles (about 0.55 ms/step) but is second-order. The
attention kernels are already FA3 and are not the bottleneck.

The offline split is a different workload from the online plateau, and the online
trace cannot confirm the busy fraction directly, so the 34% figure is corroborating
rather than measured at load 48. But both traces agree on the attribution: the
plateau's idle is `detokenize` and input preparation, not slow kernels.

## 4. Clock conditions: matched, and not the explanation

The GPU log this time carries UTC timestamps that agree with the recorded
metadata time, so alignment needed only the local-to-UTC offset rather than the
reconstruction the last two batches required. Mean SM clocks over each run
window:

| Offered load (req/s) | Mean SM clock, lean-vLLM (MHz) | Mean SM clock, vLLM (MHz) | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 1,973 | 1,979 | 0.3% |
| 4 | 1,969 | 1,952 | 0.9% |
| 8 | 1,964 | 1,935 | 1.5% |
| 12 | 1,962 | 1,942 | 1.0% |
| 16 | 1,938 | 1,928 | 0.5% |
| 24 | 1,833 | 1,868 | 1.9% |

Clocks agree within 2% everywhere, and at every load from 4 to 16 the slower
engine held the higher clock. `SwPowerCap` was asserted in some samples of every
window, more often for vLLM than for lean-vLLM, which is consistent with vLLM
doing more work per second rather than being held back. Clock conditions do not
explain any part of the deficit.

## Relation to the 11 September findings

- **The A100 saturation gap widened rather than closed.** There, lean-vLLM
  plateaued 5.7–7.0% below vLLM. Here it plateaus 24.7% below. Below saturation
  the two engines are closer than they were on the A100, which makes the
  saturation gap easy to miss on a grid that stops at load 24.
- **Two open questions are answered.** vLLM 0.26 caps piecewise capture at 512
  tokens, the same as lean-vLLM, so `FULL_AND_PIECEWISE` means the same capture
  scheme on both engines. And both engines now run the same torch build, so the
  remaining gap belongs to the engines.
- **The chunked-prefill experiment is still unrun.** It was the top-priority
  item on 11 September and remains untested back to back. The p99 TTFT symptom
  it was meant to address is absent below saturation here but returns past the
  plateau, so the test is still worth running, at higher loads than proposed.

## Measurement limits

- **The extension sweep is reported from a terminal capture.** Loads 32, 48,
  and 64 were read off the run's console output. The result JSON for those loads
  is not in this repository, so their per-run detail — server counters, GPU log
  windows, percentiles beyond those shown — could not be checked.
- **GPU sampling is coarse at high load.** The log samples every 10 seconds, so
  the load-16 and load-24 windows hold 5 to 7 samples each. The utilization gap
  between the engines is large and consistent across loads, but the per-load
  figures carry real sampling error.
- **vLLM exposes no server counters here.** Model-busy fraction, batch tokens,
  and preemptions are null in the vLLM results, so section 3's step accounting
  is lean-vLLM-only. Client-side metrics are comparable.
- **vLLM is two releases behind.** 0.26.0 was chosen because it pins torch 2.11,
  which the FlashAttention-3 wheel links against. Matching torch was judged the
  more important control.
- **The working tree was not clean.** `uv.lock` carried registry-URL rewrites to
  a package mirror; see [`working-tree.patch`](../results/20260912T105854Z/working-tree.patch).
  Package versions and hashes are unchanged.
- **Result JSON still records no run boundaries.** Windows were derived from
  file modification times and `duration_seconds`. This is the third batch with
  the same defect.

## Next experiments, in priority order

1. **Overlap `detokenize` and `prepare_batch` with the forward pass.** The
   host-side profiling in section 3 (done — see "What the idle host time is")
   found the 34% GPU idle is these two phases running serially, about 3 ms of a
   13 ms step with the GPU stalled, not slow kernels. Moving next-step batch
   preparation and detokenization off the critical path is the largest lever on
   the plateau gap. A confirming online GPU timeline is still owed: CUPTI cannot
   collect device activity from the engine thread, so the busy/idle split was
   measured on an offline run and only corroborated at load 48.
2. **Collect the extension sweep's result files and GPU log.** Loads 32 to 64
   are the loads that matter now, and they are the only ones with no server
   counters, no clock check, and no step accounting.
3. **Reconsider capturing large mixed steps.** Eager steps hold 42% of step
   time at load 24, up from 25% at load 16, and the trend was still rising where
   the counters stop. Measure the share at the plateau before deciding whether
   Route A is worth the effort.
4. **Run the chunked-prefill-off arm at the plateau.** Carried over from
   11 September, and now worth testing at loads 32 and 48 rather than the loads
   it was originally proposed for.
5. **Record explicit run start and end times in result JSON.** Third time this
   has been asked for.
