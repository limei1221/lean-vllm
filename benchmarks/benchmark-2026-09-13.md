# Online serving benchmark — 13 September 2026

Second H100 run of the same comparison, taken to measure the effect of the
step-loop overlap change. lean-vLLM now prepares the next step's batch and
detokenizes the last step's output while the forward pass runs, and with async
scheduling on it launches the next step before reconciling the previous one, so
the host work no longer sits serial in front of the GPU. This report supersedes
the [12 September report](benchmark-2026-09-12.md) for the throughput numbers;
the hardware, workload, and vLLM build are unchanged, so the two are directly
comparable. Results are in
[`results/20260913T101307Z/`](../results/20260913T101307Z/).

All fifteen serving runs completed — five offered loads per arm, three arms,
1,000 requests each — with no rejections, failures, or preemptions. Every run
replayed the identical trace: 659,770 prompt tokens and 176,838 generated
tokens, the same trace as 12 September.

lean-vLLM ran at commit `0af72d2` ("fix a lost EOS, a double finish and early
TTFT in pipelined steps"), which sits on top of the overlap change (`dc52319`)
and turning async scheduling on by default (`79d3835`). vLLM stayed on 0.26.0.
Both used `FULL_AND_PIECEWISE` graph mode and a 16-token KV block. This time
lean-vLLM ran two curves — async scheduling off and on — and vLLM ran its
default, which is on. The result JSON now records `started_at` and `finished_at`
per run, so the GPU log is aligned to real run boundaries rather than file
modification times.

The main findings are:

- **The plateau gap closed from 25% to 5–7%.** lean-vLLM with async scheduling
  on now reaches 24.45 requests/s at load 48 and 24.10 at load 64, against
  vLLM's 25.72 and 26.05. On 12 September the same loads trailed by 24.0% and
  24.7%. vLLM is unchanged, so the whole gain is lean-side.
- **The overlap change lifted the entire lean curve by about 25%.** lean-vLLM
  plateaued at 19.55–19.56 requests/s on 12 September and now plateaus at
  24.1–24.5, a 23–25% improvement from the same engine before the change.
- **The offline GPU idle fell from 34% to 3%.** The offline CUDA trace that
  measured 34.4% GPU idle on 12 September now measures 3.2% idle with async
  scheduling on and 22.3% with it off. The kernel work is unchanged; the idle
  the overlap targeted is gone.
- **Online GPU utilization now matches vLLM.** Sampled utilization under load
  rose from 67–91% to 92–98%, and lean-vLLM's power draw, which trailed by
  49–65 W on 12 September, is now within a few watts of vLLM's. The host gaps
  that the previous report diagnosed are largely closed.
- **Async scheduling is worth 7–9% at the plateau.** Turning it on adds 8.7% at
  load 48 and 6.9% at load 64 over the overlap-only arm, and 9.2% at load 24.
- **Below saturation the engines are at parity.** At load 24 lean-vLLM edges
  vLLM by 0.2%; at load 1 they are identical. The residual gap is confined to
  the plateau.

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
| System time at metadata collection | 13 September 2026, 10:13:48 UTC (`+0000`) |
| Persistence mode | Enabled |
| Application graphics clock | 1,980 MHz, equal to the maximum |
| Power limit / maximum power limit | 700.00 W / 700.00 W |

The clock was locked at the maximum, as on 12 September. Both engines ran the
same torch build.

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
| Offered loads | 1, 24, 32, 48, 64 requests/s |

vLLM's async scheduling was confirmed on from its server logs
([`vllm-async-scheduling.txt`](../results/20260913T101307Z/vllm-async-scheduling.txt)),
so the like-for-like pair is lean-vLLM's `async-scheduling=True` arm against
vLLM.

### Reading the results

| Metric | Meaning | Better direction |
| --- | --- | --- |
| Offered load | Target request arrival rate, in requests/s | Test input |
| Goodput | Completed requests divided by total run time, including queue-drain time; no latency cutoff | Higher |
| TTFT | Time to first token: how long a request waits before output begins | Lower |
| TPOT | Time per output token after the first token | Lower |
| E2E | End-to-end time to complete a request | Lower |
| p50 / p99 | Median / 99th percentile; p99 describes the slow tail | Lower for latency |

Because goodput counts the drain after the last arrival, it stays a few percent
below the offered load even when the engine keeps up. Read it against the other
engine at the same load, not against the offered rate.

## 1. Throughput: parity to load 24, a 5–7% gap at the plateau

| Offered load (req/s) | lean-vLLM off (req/s) | lean-vLLM on (req/s) | vLLM (req/s) | on vs vLLM | on vs off |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.01 | 1.01 | 1.01 | −0.0% | +0.0% |
| 24 | 17.79 | 19.43 | 19.39 | **+0.2%** | +9.2% |
| 32 | 21.97 | 22.54 | 23.07 | −2.3% | +2.6% |
| 48 | 22.49 | 24.45 | 25.72 | −4.9% | +8.7% |
| 64 | 22.53 | 24.10 | **26.05** | −7.5% | +6.9% |

Output token throughput tracks goodput exactly, since every run replays the same
trace: lean-vLLM on reaches 4,323 tok/s at load 48 and 4,261 at load 64 against
vLLM's 4,548 and 4,607.

The plateau gap is the headline. On 12 September lean-vLLM flattened at
19.55–19.56 requests/s and trailed vLLM by 24–25% from load 48 up. The overlap
change and async scheduling lift the lean curve to 24.1–24.5, closing that gap
to 4.9% at load 48 and 7.5% at load 64. vLLM's numbers are within noise of
12 September (25.72 vs 25.72 at load 48, 26.05 vs 25.97 at load 64), so the
improvement is entirely lean-side.

Async scheduling is the smaller of the two levers but a real one: on top of the
overlap-only arm it adds 9.2% at load 24, 8.7% at load 48, and 6.9% at load 64.
The overlap-only arm (`async-scheduling=False`) already plateaus at 22.5,
14–15% above the 12 September engine, so most of the 25% recovery is the overlap
itself and the rest is the launch-first pipelining that async scheduling adds.

## 2. Latency: the tail widened with throughput, still trails vLLM

lean-vLLM on now sustains higher load, so it queues at higher offered rates than
the 12 September engine did, but its tail is still above vLLM's at every load.

| Offered load (req/s) | p50 TTFT on / vLLM (ms) | p99 TTFT on / vLLM (s) | p50 TPOT on / vLLM (ms) | p50 E2E on / vLLM (s) | p99 E2E on / vLLM (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30 / 30 | 0.12 / 0.12 | 7.0 / 6.8 | 0.92 / 0.89 | 6.54 / 6.38 |
| 24 | 102 / 72 | 1.01 / 0.30 | 14.4 / 12.7 | 2.04 / 1.74 | 13.33 / 11.49 |
| 32 | 452 / 144 | 2.39 / 0.65 | 24.8 / 19.2 | 3.93 / 2.72 | 20.18 / 16.12 |
| 48 | 2,114 / 1,206 | 6.75 / 4.56 | 43.3 / 37.7 | 8.60 / 6.73 | 27.93 / 24.77 |
| 64 | 5,012 / 3,287 | 11.96 / 9.03 | 43.1 / 38.9 | 11.89 / 9.48 | 28.63 / 25.53 |

At load 1 the engines are indistinguishable: p50 TPOT differs by 0.2 ms, the
constant per-step overhead outside the graph. The ratio grows with load as
decode tokens wait behind prefill and the queue, but the shape now matches
vLLM's rather than diverging — p50 TPOT is 1.11–1.29× vLLM across the sweep,
against the 2.24× the 12 September engine reached at load 24. p99 end-to-end
latency is 1.13–1.25× vLLM at the plateau, down from about 1.45× on
12 September. No run rejected or failed a request at any load.

## 3. Online GPU utilization: the host gaps are largely closed

The 12 September report diagnosed the plateau deficit as host work stalling the
GPU: sampled utilization was 67–91% under load against vLLM's flat 100%, and
lean-vLLM drew 49–65 W less. Aligning this run's GPU log to the recorded run
windows shows both symptoms mostly resolved.

| Offered load (req/s) | util off | util on | util vLLM | power on (W) | power vLLM (W) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67% | 70% | 70% | 341 | 341 |
| 24 | 82% | 97% | 98% | 613 | 598 |
| 32 | 87% | 98% | 97% | 618 | 613 |
| 48 | 82% | 93% | 97% | 618 | 616 |
| 64 | 86% | 92% | 97% | 609 | 610 |

Turning async scheduling on raises lean-vLLM's sampled utilization from 82–87%
to 92–98%, and its power draw from 540–608 W to 609–618 W, level with vLLM. The
remaining 5-point utilization gap at loads 48 and 64 lines up with the 5–7%
throughput gap: the last of the host serial time is not yet hidden, but the bulk
of the 12 September idle is gone. At load 24, where lean-vLLM matches vLLM on
throughput, it also matches on utilization (97% vs 98%).

The GPU sampler runs every second here rather than every ten, so these windows
hold 39–52 samples each at the plateau loads, far more than the 5–7 the previous
report warned about.

## 4. Offline GPU idle: 34% down to 3%

The direct measurement of the overlap change is the offline CUDA trace, which
collects device activity and so gives a true GPU busy/idle split. As on
12 September this is an offline `generate` run on Qwen3-0.6B — 256 sequences,
200 captured steps — a smaller and different workload from the 8B online
serving trace, but the same measurement as the 34.4% idle the previous report
reported. Taking the union of all kernel and copy intervals as GPU-busy time:

| Trace | Step time (ms) | GPU busy | GPU idle |
| :--- | ---: | ---: | ---: |
| 12 September (pre-overlap) | 13.1 | 65.6% | **34.4%** |
| 13 September, async off | 11.6 | 77.7% | 22.3% |
| 13 September, async on | 9.4 | **96.8%** | **3.2%** |

The overlap change alone (async off) removes a third of the idle; adding
launch-first async scheduling removes almost all of the rest, leaving the GPU
idle only 3.2% of the offline window. The device work is unchanged across all
three: kernel-busy time holds near 1,810–1,840 ms and FlashAttention is 79.8%
of it in both arms, exactly the 79% share of 12 September. The step got faster
purely by removing host stalls, not by changing what runs on the device — the
window shrank from 2.33 s (async off) to 1.88 s (async on) while the kernels
occupied the same wall time.

This is the offline corroboration the runbook asks for, and it is decisive on
this workload: the 34% idle the 12 September profiling attributed to
`detokenize` and `prepare_batch` running serially is gone once those phases
overlap the forward pass. The online 8B serving path shows the same direction
in section 3's utilization data, though a few points of idle remain there that
the smaller offline model does not.

A host-side CPU-only trace of the load-48 online step loop was also captured for
each arm
([`profile-48/`](../results/20260913T101307Z/profile-48/)). It confirms the loop
is now split into distinct `schedule`, `launch`, `await_tokens`, `reconcile`,
and `detokenize` phases, with `await_tokens` isolating the device wait that used
to hide inside `sample`. Its absolute per-step figures are inflated by in-process
`record_function` overhead — the traced step reads far longer than the real
~10 ms online step — so they are not reported here; the offline CUDA trace is the
quantitative measurement.

## 5. Clock conditions: matched, not the explanation

Mean SM clocks over each run window agree between the engines within about 2%,
and both drop below the 1,980 MHz lock under load as `SwPowerCap` engages at the
700 W limit:

| Offered load (req/s) | SM clock on (MHz) | SM clock vLLM (MHz) | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 1,977 | 1,976 | 0.0% |
| 24 | 1,874 | 1,872 | 0.1% |
| 32 | 1,771 | 1,792 | 1.1% |
| 48 | 1,755 | 1,751 | 0.2% |
| 64 | 1,758 | 1,717 | 2.4% |

At load 64 lean-vLLM held the higher clock while still trailing on throughput,
which is consistent with the residual gap being host-side rather than a clock
disadvantage. Clock conditions do not explain any part of the deficit.

## Relation to the 12 September findings

- **The top-priority experiment landed.** Overlapping `detokenize` and
  `prepare_batch` with the forward pass — item 1 on 12 September — was the
  largest lever, exactly as the profiling predicted. It cut the offline idle
  from 34% to 22%, and with async scheduling to 3%, and closed three-quarters
  of the plateau throughput gap.
- **The extension sweep now has full result files.** Loads 32, 48, and 64 were
  read off console output on 12 September; here they are complete result JSON
  with server counters, so the plateau is measured rather than transcribed. This
  clears measurement limits 1 and 2 from the previous report.
- **Run boundaries are now recorded.** `started_at` and `finished_at` are in
  every result JSON, so the GPU log aligns to real windows. This clears the
  defect flagged in three consecutive batches.
- **The chunked-prefill-off arm is still unrun.** Carried over again; the p99
  TTFT symptom it targets is still present past the plateau (6.75 s at load 48).

## Measurement limits

- **The offline idle is Qwen3-0.6B, not the 8B online workload.** As on
  12 September, the busy/idle split is measured on the small offline model, which
  has cheaper kernels and so a relatively larger host share. It corroborates the
  online utilization data in section 3 but is not the same workload; the online
  8B path still shows a few points of residual idle at the plateau that the
  offline trace does not.
- **vLLM exposes no server counters here.** Model-busy fraction, batch tokens,
  and preemptions are null in the vLLM results, so the server-counter comparison
  is lean-only. Client-side metrics are comparable.
- **vLLM is two releases behind.** 0.26.0 was chosen because it pins torch 2.11,
  which the FlashAttention-3 wheel links against. Matching torch was judged the
  more important control.
- **The working tree carried a `uv.lock` change.** As on 12 September, `uv.lock`
  held registry-URL rewrites to a package mirror; see
  [`working-tree.patch`](../results/20260913T101307Z/working-tree.patch).
  Package versions and hashes are unchanged.

## Next experiments, in priority order

1. **Chase the residual 5–7% at the plateau.** The online utilization gap at
   loads 48 and 64 is now about 5 points, matching the throughput gap. Profile
   the online step loop with a lower-overhead method than in-process
   `record_function` — the host trace here was too inflated to attribute the
   remaining serial time — and confirm whether it is scheduling, batch
   preparation, or the launch path.
2. **Capture an online GPU timeline at the plateau.** The busy/idle split is
   still measured only offline on the 0.6B model. CUPTI cannot collect device
   activity from the engine worker thread; find a way to measure the 8B online
   loop directly so the offline 3.2% can be confirmed on the real workload.
3. **Run the chunked-prefill-off arm at loads 32 and 48.** Carried over from the
   last two reports. The p99 TTFT it targets is still elevated past the plateau.
4. **Revisit capturing large mixed steps.** With the host idle largely removed,
   remeasure the eager-step share at the plateau before deciding whether it is
   still worth capturing.
