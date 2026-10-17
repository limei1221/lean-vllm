# DeepSeek-V2-Lite serving benchmark, 17 October 2026

lean-vLLM against vLLM 0.26.0 on DeepSeek-V2-Lite-Chat (16B total, 2.4B active,
MLA + MoE + YaRN), one H100, with chunked prefill, full + piecewise CUDA
graphs, and Inductor compilation on both engines. Both engines ran with async
scheduling on. Results are in
[`results/20261017T191302Z-DeepSeek-V2-Lite/`](../results/20261017T191302Z-DeepSeek-V2-Lite/).

All fourteen serving runs completed: seven offered loads per engine, 1,000
requests each, with no rejections, failures or preemptions. Every run replayed
the same lognormal trace of 659,770 prompt tokens and 176,838 generated tokens.

This run follows the 3 October report, which is retired and in the git history.
lean-vLLM moved from `06e2725` to `cac48a2`. The changes that bear on this model
include fused Triton kernels for MoE routing and SiLU-and-mul, shared experts
on a side stream, CUDA graphs at vLLM's default capture sizes, an engine core
in its own process, and numpy-built step inputs. The load bracket gained 40
and 80 requests/s to place the knee.

The main findings:

- lean-vLLM now matches vLLM. Goodput is within 1.1% of vLLM's from load 1
  to 48, and 1.4–2.8% above it at loads 64–80. On 3 October lean-vLLM trailed by
  23% at load 48 and 32% at load 64.
- The plateau rose from ~23 to ~32 requests/s. At load 48 goodput is 29.87
  requests/s, up 29% from 23.18; at load 64 it is 31.50, up 49% from 21.14. The
  load-64 decline seen on 3 October is gone.
- Load-1 TPOT is 4.7 ms against vLLM's 4.3 ms (1.09×), down from 6.5 ms
  (1.42×). Under load, lean-vLLM's median TPOT is 0.91–1.00× vLLM's.
- Compiled prefill steps got cheaper and stopped growing with load. Steps
  above the 512-token graph range take 41–44 ms at loads 48–80, against 56–70 ms
  on 3 October, and 30–46% of step time, against 55–63%.
- Offline, kernel time is equal. Over 200 traced steps on V2-Lite-Chat both
  engines spend 3.4 s in kernels, 62% of it in the MoE; vLLM's step is 4.5%
  shorter because more of its kernels overlap and its GPU idles 2.4% against
  lean-vLLM's 3.7%.

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
| FlashMLA | commit `ba89a34` (deepseek-ai HEAD), built from source against the pinned torch |
| vLLM | 0.26.0 |
| MLA kernels | lean-vLLM: FlashMLA decode (`flashmla`), FA3 prefill; vLLM: `FLASH_ATTN_MLA` decode, `FLASH_ATTN` prefill |
| MoE | lean-vLLM: Triton fused MoE; neither engine tuned for this GPU, both use vLLM's default tile sizes |
| Persistence mode | Enabled |
| Application graphics clock | 1,980 MHz, equal to the maximum; not locked (the container lacks permission) |
| Power limit | 700 W (maximum 700 W) |

At `cac48a2` the lean-vLLM server did not start: `InputBuffers` allocated its
pinned host buffers while `ModelRunner.__init__` had set CUDA as the default
device, and warmup failed with "Only dense CPU tensors can be pinned". The run
used a one-line fix that allocates them on the CPU
(`lean_vllm/engine/input_buffers.py`); the patch is in `working-tree.patch`.

### Correctness gate

Section 0 of the [runbook](benchmark-runbook.md) passed before any timing:

| Check | Result |
| --- | --- |
| `test_attention_backends.py -k mla` (FlashMLA vs fp32 oracle) | 14 passed, none skipped |
| `test_fused_moe.py -k cuda` (Triton MoE vs `grouped_mm`) | 3 passed, none skipped |
| `test_deepseek_v2.py` (MLA + MoE + YaRN vs transformers) | 70 passed |
| Greedy generation vs vLLM, 8 prompts × 64 tokens | 4 of 8 identical; 72% of tokens agree before the first divergence |

At each of the three real divergences vLLM's own top-5 logprobs put the two
engines' tokens within 0.25 nats, and in two of them vLLM ranks lean-vLLM's
token higher. These are bf16 near-ties, not a numerics fault. A fourth, apparent
1.1-nat divergence was a re-tokenization offset in the comparison: given vLLM's
exact prefix, lean-vLLM picks vLLM's token. The notes are in `validation/`.

### Workload and server settings

| Setting | Both engines |
| --- | --- |
| Model | DeepSeek-V2-Lite-Chat, bf16, 27 layers |
| Requests per run | 1,000, plus 3 warmup requests |
| Workload | Lognormal lengths: input 512, output 128, σ = 0.8 |
| Sampling | Greedy, seed 0 |
| Batch token budget / maximum sequences | 8,192 / 256 |
| Chunked prefill | Enabled |
| Graph mode | Full + piecewise |
| KV cache | 327,680 tokens; lean-vLLM uses 64-token blocks, FlashMLA's page |
| Maximum model length | 4,096 |
| Offered loads | 1, 24, 32, 40, 48, 64, 80 requests/s |

vLLM's async scheduling was confirmed on in every server log
([`vllm-async-scheduling.txt`](../results/20261017T191302Z-DeepSeek-V2-Lite/vllm-async-scheduling.txt)).
Both engines decoded with an MLA kernel, not an expanded fallback
([`vllm-attention-backend.txt`](../results/20261017T191302Z-DeepSeek-V2-Lite/vllm-attention-backend.txt);
lean-vLLM's offline trace shows `flash_fwd_splitkv_mla_kernel`). vLLM ran with
prefix caching on, its default; lean-vLLM recorded no prefix-cache hits on this
trace, so prefix caching should not favour either engine.

### How the two engines execute a step

Both engines capture MLA decode inside the full graph, compile the model
piecewise with Inductor, and now capture the same graph sizes.

| | lean-vLLM | vLLM 0.26.0 |
| --- | --- | --- |
| Process layout | API server and engine core in separate processes, ZMQ between them | Same |
| Compilation | Inductor piecewise: the model is traced whole, split at attention ops, and each piece is compiled by Inductor | Inductor piecewise, same approach |
| MLA decode (batches up to `max_num_seqs`) | Inside the full CUDA graph (FlashMLA), 35 sizes | Inside the full CUDA graph, 35 sizes |
| Prefill / mixed steps up to 512 tokens | Inductor-compiled piecewise graphs, 51 sizes | Inductor-compiled piecewise graphs, 51 sizes |
| Prefill / mixed over 512 tokens | Inductor-compiled, no graph | Inductor-compiled, no graph |
| Shared experts | Side stream beside the routed experts | Side stream (`SharedFusedMoE`) |
| Async scheduling | On | On by default |

lean-vLLM runs `full_and_piecewise` with the decode step captured. The FlashMLA
HEAD here fuses the decode schedule into the kernel; lean-vLLM warms up and
captures at worst-case sequence lengths (`context_lens = max_model_len`), so the
tile schedule and split-KV workspace baked into the graph are sized for the
longest sequence, and each replay refreshes `context_lens`/`block_tables` while
the kernel gates its KV loop on them. Pure decode batches therefore replay the
whole model, attention included. See the
[runbook](benchmark-runbook.md) for the mechanism.

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

| Offered load (req/s) | lean-vLLM | vLLM | lean vs vLLM |
| ---: | ---: | ---: | ---: |
| 1 | 1.01 | 1.01 | −0.0% |
| 24 | 19.92 | 19.95 | −0.2% |
| 32 | 24.17 | 24.44 | −1.1% |
| 40 | 27.64 | 27.80 | −0.6% |
| 48 | 29.87 | 29.80 | +0.2% |
| 64 | 31.50 | 31.06 | +1.4% |
| 80 | **32.02** | 31.14 | +2.8% |

Goodput is in requests/s. Output token throughput follows goodput: at load 48
both engines produce about 5,280 tok/s (5,282 and 5,269); at load 80 lean-vLLM
produces 5,662 tok/s against vLLM's 5,506.

Both engines saturate around load 64. From 48 to 64 lean-vLLM gains 5.5% and
vLLM 4.2%; from 64 to 80 they gain 1.6% and 0.3%, while p99 TTFT climbs from
1.3–1.7 s at load 48 to 5.2–6.6 s at load 80. The new loads 40 and 80 confirm the
curves straddle the knee. Past it lean-vLLM is slightly ahead; with one run per
point, a 1–3% lead is within noise.

Against 3 October, vLLM's curve is unchanged (29.80 vs 30.13 at load 48, 31.06
vs 31.21 at load 64), so the shift is lean-vLLM's: +4.6% at load 24, +8.1% at
32, +29% at 48 and +49% at 64.

## 2. Latency

| Offered load (req/s) | p50 TTFT lean / vLLM (ms) | p99 TTFT lean / vLLM (s) | p50 TPOT lean / vLLM (ms) | p50 E2E lean / vLLM (s) | p99 E2E lean / vLLM (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37 / 34 | 0.10 / 0.09 | **4.7 / 4.3** | 0.66 / 0.61 | 4.59 / 4.18 |
| 24 | 78 / 92 | 0.24 / 0.23 | 15.9 / 17.1 | 2.12 / 2.25 | 14.26 / 15.02 |
| 32 | 108 / 124 | 0.64 / 0.32 | 18.1 / 19.9 | 2.45 / 2.61 | 15.13 / 15.85 |
| 40 | 187 / 190 | 0.73 / 0.64 | 19.8 / 21.1 | 2.75 / 2.87 | 15.68 / 16.30 |
| 48 | 314 / 367 | 1.27 / 1.67 | 21.6 / 22.5 | 3.12 / 3.37 | 15.98 / 16.17 |
| 64 | 1,434 / 1,145 | 3.04 / 3.57 | 27.2 / 27.2 | 5.02 / 5.23 | 18.47 / 19.00 |
| 80 | 1,632 / 2,518 | 5.18 / 6.58 | 28.3 / 29.2 | 5.99 / 7.22 | 19.49 / 20.04 |

Load 1 isolates per-step overhead: lean-vLLM spends 4.7 ms per generated token
to vLLM's 4.3 ms (1.09×), down from 6.5 ms (1.42×) on 3 October. Both decode
steps replay one captured graph, so the remaining 0.4 ms is in the kernels
inside it or in the per-step host work around it.

Under load, lean-vLLM's median TPOT is 0.91–1.00× vLLM's and its p99 E2E is
lower at every load from 24 up. The p99 TTFT tail is mixed: lean-vLLM is lower
at loads 48–80, but at load 32 it is 0.64 s against vLLM's 0.32 s. On 3 October
vLLM showed the opposite outlier at load 32, so this point looks noisy.

## 3. GPU utilization and clock conditions

Sampled once per second and aligned to each run's recorded start and finish:

| Offered load (req/s) | util lean | util vLLM | power lean (W) | power vLLM (W) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 56% | 52% | 235 | 240 |
| 24 | 95% | 93% | 504 | 508 |
| 32 | 92% | 90% | 508 | 506 |
| 40 | 96% | 95% | 530 | 532 |
| 48 | 97% | 93% | 534 | 533 |
| 64 | 94% | 90% | 530 | 528 |
| 80 | 90% | 88% | 528 | 524 |

lean-vLLM now keeps the GPU at least as busy as vLLM at every load and draws the
same power. On 3 October its utilization fell from 78% to 68% at loads 48–64
while vLLM held 91–92%.

| Offered load (req/s) | mean SM clock lean (MHz) | mean SM clock vLLM (MHz) | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 1,980 | 1,980 | +0.0% |
| 24 | 1,966 | 1,964 | +0.1% |
| 32 | 1,959 | 1,962 | −0.2% |
| 40 | 1,936 | 1,961 | −1.3% |
| 48 | 1,950 | 1,930 | +1.0% |
| 64 | 1,944 | 1,948 | −0.2% |
| 80 | 1,946 | 1,954 | −0.4% |

Clocks are averaged over busy samples and match within about 1% at every load
except 40, where lean-vLLM ran 1.3% slower and still trailed by only 0.6%. The
average draw stays near 530 W, but `SwPowerCap` appears in 30–60% of samples
under load for both engines, so short peaks reach the 700 W cap; no other
throttle reason appeared, and the GPU peaked at 62 °C.

## 4. Where the step time goes

### Online: server counters

lean-vLLM's server counters split each step by how it ran. `graph` is a
pure-decode batch replaying the whole model from one captured graph, attention
included; `piecewise` is a prefill or mixed step of up to 512 tokens replaying
per-piece graphs with attention running between them; `prefill` is a prefill or
mixed step past the 512-token grid, running compiled through Inductor but
without a graph capture. Figures are for the measured run, warmup subtracted:

| | load 24 | load 32 | load 48 | load 64 | load 80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Steps | 3,938 | 3,150 | 2,520 | 2,291 | 2,213 |
| Captured (decode + piecewise) steps | 3,500 (88.9%) | 2,769 (87.9%) | 2,292 (91.0%) | 1,955 (85.3%) | 1,864 (84.2%) |
| Compiled prefill steps | 438 (11.1%) | 381 (12.1%) | 228 (9.0%) | 336 (14.7%) | 349 (15.8%) |
| Full-graph decode step time | 31.2 s (62.9%) | 24.8 s (60.8%) | 22.3 s (67.7%) | 16.8 s (53.8%) | 16.0 s (52.0%) |
| Piecewise step time | 3.3 s (6.7%) | 1.9 s (4.7%) | 0.6 s (1.8%) | 0.6 s (2.0%) | 0.5 s (1.6%) |
| Compiled prefill step time | 15.0 s (30.3%) | 14.1 s (34.5%) | 10.0 s (30.5%) | 13.8 s (44.2%) | 14.3 s (46.4%) |
| Mean compiled prefill step | 34.3 ms | 36.9 ms | **44.0 ms** | **41.1 ms** | 40.9 ms |
| Mean captured step | 9.9 ms | 9.6 ms | 10.0 ms | 8.9 ms | 8.8 ms |

On 3 October the compiled prefill steps cost 50–70 ms and grew with load, taking
55–63% of step time at loads 48–64; that growth drove the load-64 goodput
decline. They now cost 34–44 ms and level off at about 41 ms past load 48, and
their share stays below half even at load 80. The captured steps also got
slightly faster, 8.8–10.0 ms against 9.3–11.0 ms. The MoE routing kernels and
the shared-expert side stream shorten both kinds of step; the engine-core split
takes detokenization and the HTTP front end off the step loop's host thread.

### Offline: equal kernel time, vLLM overlaps more

The offline CUDA trace records device activity, so it gives a true busy/idle
split. It is an offline `generate` on DeepSeek-V2-Lite-Chat: 256 sequences
with inputs and outputs of 100–1,024 tokens, 200 steps captured after 200
skipped. vLLM ran the same workload and seed through its own profiler, with the
same skip and capture counts (`bench_offline_vllm.py` in the results
directory). The window runs from the first captured step's host work to the
last device or host event, and GPU-busy time is the union of all kernel and
copy intervals in it.

| | lean-vLLM, async off | lean-vLLM, async on | vLLM |
| --- | ---: | ---: | ---: |
| Window, 200 steps | 3,992 ms | 3,471 ms | **3,324 ms** |
| Step time | 20.0 ms | 17.4 ms | **16.6 ms** |
| Kernel time | 3,413 ms | 3,412 ms | 3,395 ms |
| GPU-busy time | 3,341 ms | 3,343 ms | 3,244 ms |
| Kernels per step | 758 | 758 | 786 |
| MoE share of kernel time | 61.5% | 61.5% | 62.5% |
| Attention share of kernel time | 14.2% | 14.2% | 16.6% |
| GPU busy | 83.7% | 96.3% | **97.6%** |
| GPU idle | 16.3% | 3.7% | 2.4% |
| Host `await_tokens` per step | 15.30 ms | 6.03 ms | — |

The captured window is decode-heavy: lean-vLLM's attention time is almost all
FlashMLA's `flash_fwd_splitkv_mla_kernel`. Async scheduling shortens lean-vLLM's
step by 13% with the same kernel work, cutting idle from 16.3% to 3.7%.

Against vLLM the kernel time agrees within 0.5%, but vLLM's step is 4.5%
shorter. Two things account for it: vLLM idles 1.3 points less, and more of
its kernels overlap. Its kernel time exceeds its busy time by at least 151 ms,
against at least 69 ms for lean-vLLM. As with Qwen, the two engines schedule
independently, so the captured steps need not hold identical batches.

### Online host trace at load 48

A CPU-only step-loop trace of one load-48 run, 600 steps
([`profile-48/`](../results/20261017T191302Z-DeepSeek-V2-Lite/profile-48/)),
cut goodput by 10% (26.94 vs 29.87 requests/s). Only the proportions are
meaningful:

| Host phase, mean per step | Async on |
| --- | ---: |
| `schedule` | 0.27 ms |
| `launch` | 12.75 ms |
| ↳ `prepare_batch` | 0.93 ms |
| ↳ `run_model` | 11.39 ms |
| ↳ `sample` | 0.05 ms |
| `await_tokens` | 9.36 ms |
| `reconcile` | 0.21 ms |
| Traced step | 23.2 ms |

The host's own work is dominated by `run_model`, the dispatch of the forward
pass; scheduling, batch preparation and reconciliation together take under
1.5 ms per step.

## Measurement limits

- One run per point. No point was repeated, so differences of a few percent
  between neighbouring loads, including lean-vLLM's 1–3% lead past the knee,
  are within run-to-run noise until repeated.
- Clocks were not locked. The container could not set `-lgc`; both engines
  ran at the default application clock, which is the maximum, and busy-sample
  clocks are reported per load instead.
- Shared host. Other users' jobs ran on other GPUs of the same node during the
  session, sharing host CPU and memory bandwidth with both engines.
- Neither engine tuned its MoE. Both use default tile sizes for the routed
  experts; tuning could change the absolute numbers.
- The offline traces do not hold identical batches. Both engines skip and
  capture the same number of steps of the same workload, but schedule
  independently.
- No async-off serving curve. Only async scheduling on was run online; the
  pipelining gain is measured offline only.
- vLLM exposes no step counters here. The step breakdown by kind is
  lean-vLLM only.
- vLLM is not the latest release. 0.26.0 is the last release that pins
  torch 2.11, which the FlashAttention-3 wheel links against; matching torch
  was chosen over a newer vLLM.

## Next experiments, in priority order

1. Repeat the points around the knee. Three runs each at loads 48, 64 and
   80 would say whether lean-vLLM's 1–3% lead past the knee is real.
2. Tune the MoE kernels. The MoE is 62% of kernel time on both engines and
   neither ran with tuned tile sizes for E=64, N=1408 on this H100. Tune both and
   rerun.
3. Close the remaining load-1 gap. Decode is 4.7 ms per token against vLLM's
   4.3 ms with both replaying a full graph; compare the two decode graphs kernel
   by kernel on a fixed batch, including how much of each overlaps the side
   stream.
