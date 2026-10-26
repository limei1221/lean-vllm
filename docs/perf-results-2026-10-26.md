# Performance experiments, 25–26 October 2026

Results of executing [the optimization plan](2026-10-19-performance-optimization.md) on one H100, lean-vLLM against
vLLM 0.26.0 on DeepSeek-V2-Lite-Chat and Qwen3-8B. Raw results, per-request JSON, server logs, GPU logs and every
script are under `results/perf-2026-10-25/` (not tracked by git).

The main findings:

- **MoE routing kernels sized for decode** (kept). Routing, block alignment and the top-k sum used tiles sized for
  thousands of tokens at every batch size. Sizing them to the batch cuts one DeepSeek-V2-Lite decode step at batch 1
  from 4.10 to 3.80 ms (vLLM: 3.75 ms) and at batch 32 from 9.72 to 9.00 ms (vLLM: 9.87 ms). Serving, median TPOT at
  load 1 falls from 5.0 to 4.6 ms (vLLM 4.4 ms) and goodput at load 48 rises 2.1%, outside run-to-run variation.
- **Tuned MoE launch configs** (rejected). The tuner's configs are 5–13% faster on prefill-sized batches in isolation,
  on random and on captured real routing, but gave no repeatable serving gain and add first-use Triton compiles.
- **Qwen3-8B decode trails vLLM by 2–5%** (explained, not fixed). Prefill is equal. The decode gap is inside the CUDA
  graph: wider gaps between kernels, and vLLM's own FlashAttention build being about 4.5% faster at batch 128. Neither
  is a parameter lean-vLLM gets wrong.
- **Prefill budget and graph range** (a workload setting, defaults unchanged). On Qwen3-8B, a 2,048-token budget with
  graphs up to 2,048 tokens halves p99 inter-token latency at loads 32–48, raises SLO goodput at load 48 by 19%, and
  goodput by 0.3–1.2%. It costs 9 s of startup and 664 MiB of graph memory. vLLM does not change at that budget, so
  at equal settings lean-vLLM still trails it by 0.2–1.4%.
- **Speculative decoding** is blocked on this hardware: neither model has a draft mechanism lean-vLLM supports.

## Setup

| Component | Value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3, driver 580.126.09; clocks not lockable (the container lacks permission) |
| PyTorch, CUDA, cuBLAS, Triton | 2.11.0+cu130, 13.0, 13.1.0.3, 3.6.0 in both environments |
| lean-vLLM | `54b92b9` plus this branch's working tree (`perf/optimization-experiments`) |
| FlashMLA | `ba89a34`, built from source |
| vLLM | 0.26.0, V2 model runner, async scheduling on (logged) |
| Workload | Lognormal trace, medians 512 / 128 tokens, σ 0.8, greedy, seed 0, as in the [runbook](benchmark-runbook.md) |
| Limits | 4,096 context, 327,680 KV tokens on both engines, 256 sequences, 8,192-token budget unless stated |
| Requests per point | 1,000; 200 at load 1 (it measures per-token latency, and 1,000 at 1/s take 17 minutes) |
| Repetitions | 3 per engine per point, engine order alternating between repetitions |
| SLO goodput limits | TTFT ≤ 1.0 s and per-request TPOT ≤ 50 ms, chosen before any run |

The host is shared: its load average was 14–19 on 16 cores throughout, from other users' jobs, and is logged beside
every run (`loadavg.log`). Busy SM clocks matched between engines within 0.2% (DeepSeek-V2-Lite 1,963 against
1,962 MHz; Qwen3-8B 1,871 against 1,868 MHz; `gpu_conditions.py`).

Before any timing, the GPU gates of the runbook ran and none skipped: MLA backends (14), the CUDA MoE (3) and
DeepSeek-V2 against transformers (70). They were re-run after every kernel change.

### New measurement tools

- `bench_serving.py --slo-ttft --slo-tpot`: SLO goodput, the requests meeting both limits per second of the whole run,
  queue drain included. A failed request misses; a one-token answer has no inter-token interval and is judged on TTFT
  alone. The existing goodput (completions per second, no latency cutoff) is unchanged.
- `--max-cudagraph-capture-size`: graphs past vLLM's default 512 tokens.

One-off scripts, kept out of the repository under `results/perf-2026-10-25/scripts/` and in commit `b052aa5`:

- `repeat_sweep.sh`: repeated, alternating-order curves with environment, GPU and host-load logs.
- `aggregate_reps.py`: mean and range per engine and load across repetitions.
- `gpu_conditions.py`: busy SM clock, temperature and power per engine from the sweep's nvidia-smi log.
- `bench_fixed_batch.py`: identical batches through either engine's offline `LLM` (cold prefill, cached prefix, pure
  decode), with an optional per-kernel breakdown.

## 1. Baseline

Three repetitions each; mean (min–max). Goodput is completed requests per second; SLO goodput applies the limits above.

### DeepSeek-V2-Lite-Chat

| Load | lean-vLLM goodput | vLLM goodput | lean-vLLM SLO goodput | vLLM SLO goodput | lean-vLLM TPOT p50 | vLLM TPOT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.96 | 0.96 | 0.95 | 0.96 | 5.0 ms (5.0–5.0) | 4.4 ms (4.3–4.5) |
| 48 | 30.03 (29.92–30.14) | 30.19 (29.98–30.37) | 29.48 | 28.39 | 22.1 ms | 22.3 ms |
| 64 | 31.45 (30.86–31.91) | 30.77 (30.66–30.96) | 14.01 | 13.43 | 27.5 ms | 29.0 ms |
| 80 | **32.42 (32.09–32.61)** | 30.53 (30.28–30.69) | 10.51 | 9.17 | 28.3 ms | 31.8 ms |

lean-vLLM's lead at load 80 (+6.2%) holds across repetitions; loads 48 and 64 are within variation. At load 1
lean-vLLM was 11.8% slower per token, every repetition.

### Qwen3-8B

| Load | lean-vLLM goodput | vLLM goodput | lean-vLLM SLO goodput | vLLM SLO goodput | lean-vLLM TPOT p50 | vLLM TPOT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.96 | 0.96 | 0.96 | 0.96 | 6.9 ms | 6.8 ms |
| 32 | 23.04 (23.02–23.06) | 23.14 (23.06–23.29) | 22.96 | 23.10 | 20.8 ms | 19.4 ms |
| 48 | 25.25 (25.18–25.31) | **25.60 (25.55–25.63)** | 7.58 | 9.58 | 40.1 ms | 38.2 ms |
| 64 | 25.38 (25.30–25.54) | **25.94 (25.93–25.95)** | 1.84 | 3.48 | 40.8 ms | 39.2 ms |

lean-vLLM trails by 1.4% at load 48 and 2.2% at load 64, outside variation, with 4–7% higher TPOT under load.

### Cold start

lean-vLLM's first server start after a code change shows 0.4–1.3 s TTFT on a handful of early requests (p99 TTFT
0.86 s at load 1 against 0.09 s on later starts). Later starts of the same code do not, and neither did vLLM beyond
0.4 s. These are Triton kernel variants compiled on first use and then served from the disk cache; the 17 October
DeepSeek runs show the same pattern. The tuned MoE configs make it worse (more distinct variants). Warming up each
MoE config bucket at startup would move it out of live traffic.

## 2. DeepSeek MoE

### Tuning the launch configs (rejected)

`tune_moe.py --tune` over vLLM's 1,920-config space took 3 hours. Re-timed at 100 iterations against the defaults:

| Tokens | Random routing | Real routing (captured from the model on this repository's docs) |
| ---: | ---: | ---: |
| 1–32 | within ±1.2% | within ±3% |
| 48–512 | −1% to −4% | −3% to +2% |
| 1,536–4,096 | −5.5% to −8.4% | −11% to −13% |

Served with the kernel fix below, the tuned file gave no repeatable gain: goodput at load 48 was 30.34–30.47 against
30.62–30.72 without it, and loads 64 and 80 overlapped. Its first repetition also paid first-use compiles in live
traffic (TPOT 27 ms at load 48). It is not shipped, so vLLM was not tuned for a tuned-against-tuned comparison.

Captured routing from the engine's warmup prompt (all token ID 0) is extremely skewed, 10.7× the mean load on one
expert, and under it the tuned configs are 54–150% slower at 64–256 tokens: a reason to rank configs on real routing.

### Routing kernels sized for the batch (kept)

At batch 1 lean-vLLM's DeepSeek-V2-Lite step was 9.5% slower than vLLM's offline (4.10 against 3.75 ms), entirely in
GPU kernel time inside the full graph. Profiling one eager step against vLLM's attributed it to three small MoE kernels
that kept their large-batch tiles at every batch size:

- `topk_softmax` routed 64 tokens per program (`4096 // experts`) even for one token: 10 µs per layer. Tiles of
  512 scores (8 tokens for 64 experts) take 2.0–2.4 µs from 1 to 8,192 tokens. Grouped routing (DeepSeek-V3's 256
  experts in 8 groups) is fastest at one token and one warp per program: 4 µs against 34 µs at one token, 28 against
  77 µs at 8,192.
- Block alignment took three launches over 128-row one-hot tiles: 7.4–9.8 µs. One program now aligns a decode-sized
  batch (at most 4,096 one-hot elements and 128 experts), and the three-kernel path picks its tile from the pair count.
- The top-k sum was `torch.sum` over `[T, 6, 2048]`; a Triton kernel halves it.

Per MoE layer, in CUDA graphs (µs):

| Tokens | Routing before → after | Alignment | Sum | Whole `fused_experts` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 9.65 → 1.95 | 7.37 → 2.88 | 2.16 → 1.27 | 58.8 → 50.1 |
| 16 | 9.96 → 2.19 | 7.98 → 5.28 | 2.63 → 1.52 | 298.3 → 292.9 |
| 256 | 9.98 → 2.19 | 9.80 → 7.62 | 4.27 → 2.24 | 421.3 → 415.9 |

Decode step, fixed batch, DeepSeek-V2-Lite:

| Batch | lean-vLLM before | after | vLLM |
| ---: | ---: | ---: | ---: |
| 1 | 4.103 ms | 3.796 ms | 3.746 ms |
| 8 | 7.040 ms | 6.709 ms | 6.990 ms |
| 32 | 9.723 ms | 9.004 ms | 9.871 ms |

Serving, three repetitions against the baseline:

| Load | Goodput before | after | vLLM | TPOT p50 before | after | vLLM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.96 | 0.96 | 0.96 | 5.0 ms | **4.6 ms** | 4.4 ms |
| 48 | 30.03 (29.92–30.14) | **30.67 (30.62–30.72)** | 30.19 | 22.1 ms | 21.3 ms | 22.3 ms |
| 64 | 31.45 (30.86–31.91) | 31.44 (30.46–32.30) | 30.77 | 27.5 ms | 27.2 ms | 29.0 ms |
| 80 | 32.42 (32.09–32.61) | 32.56 (32.29–32.80) | 30.53 | 28.3 ms | 28.4 ms | 31.8 ms |

Load-1 TPOT now trails vLLM by 4.5% rather than 11.8%. The remaining batch-1 difference sits in MLA attention and
the dense GEMMs, which this change does not touch.

Correctness: `tests/test_fused_moe.py` (186 pass) adds the one-program alignment's boundary (5, 10, 11 tokens), a
128-row block, routing at 1 and 7 tokens for every scoring mode including V3's, and the Triton sum; the DeepSeek-V2
tests (70) pass. The V3 routing speedup is measured on the kernel only: no V3 checkpoint fits this GPU.

## 3. The Qwen3-8B decode gap

`bench_fixed_batch.py`, the same token IDs through both engines' offline `LLM`, median of five:

| Case | lean-vLLM | vLLM | Difference |
| --- | ---: | ---: | ---: |
| Prefill, 1 × 512 | 13.97 ms | 13.97 ms | 0.0% |
| Prefill, 16 × 512 | 191.0 ms | 189.7 ms | +0.7% |
| Decode step, batch 1 | 6.729 ms | 6.587 ms | +2.2% |
| Decode step, batch 64 | 12.008 ms | 11.496 ms | +4.5% |
| Decode step, batch 128 | 11.431 ms | 10.931 ms | +4.6% |
| Decode step, batch 256 | 17.387 ms | 16.823 ms | +3.4% |

The decode difference matches the serving TPOT gap. Splitting each decode step at its graph replay:

| Batch 1 / 128 | lean-vLLM | vLLM |
| --- | ---: | ---: |
| Between graph replays (LM head, sampling, input copies) | 0.507 / 0.575 ms | 0.518 / 0.588 ms |
| GPU idle between replays | 0.039 / 0.047 ms | 0.036 / 0.038 ms |
| Graph span | **6.307 / 10.499 ms** | 6.146 / 10.104 ms |
| Kernel time inside the graph | 6.051 / 10.309 ms | 6.132 / 10.118 ms |
| Gaps between kernels inside the graph | **0.256 / 0.191 ms** | 0.045 / 0.015 ms |

Outside the graph the engines are equal, so the host step loop is not the cause. Inside it:

- **Spacing between kernels**: about 0.45 µs per kernel boundary in lean-vLLM's graph against about 0.1 µs in vLLM's,
  for the same cuBLAS kernel pairs. It persists when lean-vLLM's graph replays alone, on the default or a dedicated
  stream (6.17 ms either way), and the two environments have identical torch, CUDA, cuBLAS, CUPTI and Triton. Its
  cause is not established; Nsight Systems on both engines (not installed here) is the next step.
- **FlashAttention decode at batch 128**: 4.12 ms against 3.94 ms per graph. A sweep of lean-vLLM's FA3 call over
  page-table width, forced split counts and GQA packing found nothing faster than its current call at batch 128
  (≤ 4% at 256 with one split), so the difference is vLLM's own FlashAttention build.
- **The FA output copy** in `flash_backend.py` (`write_into`): 35–52 µs per step, about 0.5%. FA3's custom op takes an
  `out_` but does not declare it mutated, so removing the copy would rely on an undeclared mutation through a private
  API. Not changed.
- Fused QK-norm and RoPE: about 50 µs per step slower than vLLM's two kernels at batch 128 (0.5%). Not pursued.

## 4. Prefill budget and graph range (Qwen3-8B)

One run per point first, then three repetitions of the candidates. All runs at loads 32, 48 and 64 with the baseline
trace; KV capacity pinned, so graph memory does not shrink the cache. Busy SM clocks per load matched the baseline
within about 1% (for lean-vLLM at load 48: 1,739, 1,758, 1,734 and 1,739 MHz).

### Budget sweep, current graphs (single runs)

| Budget | Goodput at 32 / 48 | SLO goodput at 32 / 48 | TTFT p50 at 48 | p99 ITL at 32 / 48 |
| ---: | --- | --- | ---: | --- |
| 512 | 22.72 / 23.10 | 18.74 / 3.95 | 4.86 s | 45 / 52 ms |
| 1,024 | 23.24 / 25.36 | 23.24 / 7.25 | 1.96 s | 58 / 101 ms |
| 2,048 | 23.12 / 25.44 | 23.08 / 7.63 | 1.74 s | 71 / 140 ms |
| 4,096 | 23.04 / 25.16 | 23.00 / 7.77 | 1.66 s | 107 / 127 ms |
| 8,192 | 23.02 / 25.10 | 22.98 / 7.43 | 1.70 s | 138 / 200 ms |

512 cannot keep up with prefill. From 1,024 up, goodput moves by about 1%, inside run-to-run variation, while tail
inter-token latency grows with the budget: a long prefill chunk stalls every decode row sharing its step.

### Graph range

Graphs past 512 tokens cost little: capturing to 1,024 adds 4 s of startup and 228 MiB, to 2,048 adds 9 s and 664 MiB
(about 4,700 KV tokens of Qwen3-8B, under 2% of this run's cache).

### Repeated candidates (three runs each, lean-vLLM)

| Budget, graphs to | Goodput at 32 / 48 / 64 | p99 ITL at 32 / 48 / 64 | TTFT p50 at 48 | SLO goodput at 48 |
| --- | --- | --- | ---: | ---: |
| 8,192, 512 (baseline) | 23.04 / 25.25 / 25.38 | 144 / 195 / 208 ms | 1.60 s | 7.58 |
| 2,048, 512 | 23.18 / 25.20 / 25.43 | 73 / 134 / 168 ms | 1.71 s | 7.07 |
| 8,192, 2,048 | 23.07 / 25.34 / 25.48 | 149 / 194 / 219 ms | 1.38 s | 9.33 |
| **2,048, 2,048** | 23.11 / 25.35 / **25.68 (25.58–25.80)** | **77 / 120 / 196 ms** | **1.33 s** | **8.99** |

The two settings act on different things. The smaller budget halves tail inter-token latency; larger graphs cover
the 512–2,048-token mixed steps, which lowers TTFT and raises SLO goodput at load 48. Together, goodput rises
0.3–1.2% (outside variation only at load 64), SLO goodput at load 48 by 19%, and p99 ITL falls 47% and 38% at loads 32
and 48.

vLLM at the same 2,048 budget, three runs: 23.16 / 25.64 / 26.05, within 0.4% of its 8,192 baseline. At equal
budgets lean-vLLM trails by 0.2%, 1.1% and 1.4%, against 0.4%, 1.4% and 2.2% at 8,192: the remaining gap is the decode
step of section 3.

No adaptive budget was built: a fixed 2,048 met the latency targets as well as any setting tried, and the plan reserves
an adaptive budget for when no fixed one does. The defaults stay at vLLM's (8,192, graphs to 512) so the two engines
remain comparable out of the box; for latency-sensitive Qwen3-8B serving on an H100, use
`--max-num-batched-tokens 2048 --max-cudagraph-capture-size 2048`.

## 5. Cache-aware scheduling

Not run. The plan conditions it on shared-prefix workloads being the target deployment. The traces and the hit-rate
counter it needs were drafted in `b052aa5` and removed until the experiment runs.

## 6. Speculative decoding

Blocked on this hardware. lean-vLLM's only drafter is DeepSeek-V3's MTP; neither Qwen3-8B nor DeepSeek-V2-Lite has
MTP layers (no `num_nextn_predict_layers`), and V3 does not fit on one H100.

## Decisions

| Experiment | Decision |
| --- | --- |
| MoE routing, alignment and sum kernels sized for the batch | Keep |
| Tuned MoE launch configs for V2-Lite on H100 | Reject: no repeatable serving gain |
| FA3 output copy removal | Reject: private API, ~0.5% |
| Qwen3-8B kernel spacing inside graphs | Inconclusive: needs Nsight Systems |
| Budget 2,048 with graphs to 2,048 for Qwen3-8B | Keep as a documented setting; defaults unchanged |
| Graphs past 512 tokens at budget 8,192 alone | No repeatable gain |
| Cache-aware scheduling | Not run: deployment target not shared-prefix |
| Speculative decoding | Blocked: no compatible draft model on one H100 |
