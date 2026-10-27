# lean-vLLM Performance Optimization Plan

> **For agentic workers:** Use `superpowers:executing-plans` to execute each experiment and review its evidence before proceeding. Use `superpowers:subagent-driven-development` if parallel agent execution is requested. Track progress with the checkboxes below.

**Goal:** Establish repeatable performance wins over vLLM on specified models and workloads, while preserving generation correctness and lean-vLLM's small implementation.

**Architecture:** Optimize the existing execution path first: MoE kernels, attention calls, prefill scheduling, and CUDA graph coverage. Evaluate cache-aware scheduling and speculative decoding as separate workload-specific experiments after establishing reliable baselines.

**Tech stack:** Python, PyTorch/Inductor, Triton, CUDA graphs, FlashAttention-3, FlashMLA, and the existing serving benchmark harness; initially one NVIDIA H100 80 GB.

**Spec:** The optimization recommendations discussed on 19 October 2026, captured below. Supporting evidence: the [Qwen3-8B report](benchmark-2026-10-17-Qwen3-8B.md), [DeepSeek-V2-Lite report](benchmark-2026-10-17-DeepSeek-V2-Lite.md), and [results table](benchmark-results-2026-10-17.md).

**Status, 26 October:** Steps 1–4 have run; see the [25–26 October results](perf-results-2026-10-26.md). Step 5 was not run and step 6 is blocked. The open work is under [Next steps](#next-steps).

| Step | Outcome |
| --- | --- |
| 1. Baseline | Done. DeepSeek-V2-Lite leads by 6.2% at load 80; Qwen3-8B trails by 1.4–2.2% at loads 48–64. |
| 2. DeepSeek MoE | Tuned configs rejected. Routing, alignment and sum kernels sized to the batch: kept. |
| 3. Qwen GPU gap | Explained, not fixed: kernel spacing inside the decode graph and vLLM's FlashAttention build. |
| 4. Prefill and graphs | Budget 2,048 with graphs to 2,048 documented for Qwen3-8B; defaults unchanged. |
| 5. Cache-aware scheduling | Not run: shared-prefix serving is not the deployment target. |
| 6. Speculative decoding | Blocked: no supported draft mechanism for either model on one H100. |

## Constraints and success criteria

- Compare identical model weights, tokenizer, precision, sampling settings, workload traces, and resource limits. Record all differences in attention backends and software versions.
- Keep the historical vLLM 0.26.0 comparison for continuity. Before claiming a win over current vLLM, run a separate comparison against a current stable release and document dependency differences.
- Repeat key measurements at least three times, alternating engine order. Record GPU clocks, power, temperature, and competing host activity; lock clocks when available.
- Preserve the existing correctness checks. GPU-specific checks must execute rather than silently skip or use a fallback.
- Change one factor at a time. Retain an optimization only when its end-to-end benefit exceeds measured variation and its latency, memory, and correctness costs are acceptable for the target workload.
- This is an experiment plan. Kernel and scheduler changes should follow evidence from their preceding measurements; no speedup is assumed in advance.

## Starting evidence

| Observation from 17 October | Implication |
| --- | --- |
| Qwen3-8B trails vLLM by 0.6–1.7% at high offered load. | Repeat measurements before treating the gap as real. |
| DeepSeek-V2-Lite leads by 1.4–2.8% at loads 64–80. | Promising, but single runs do not establish a reliable lead. |
| DeepSeek MoE accounts for 62% of offline kernel time; both engines use default tiles. | MoE tuning is the first concrete optimization experiment. |
| Qwen3-0.6B offline traces show about 15% more aggregate kernel time than vLLM. | Compare identical batches before attributing the gap to kernels. This is not an online Qwen3-8B measurement. |
| Qwen3-8B steps above the 512-token graph range account for about 45% of recorded step time at load 48. | Investigate prefill budgets and graph coverage together. |
| The current Qwen trace recorded no prefix-cache hits. | It cannot evaluate improvements to cache reuse. |

Async scheduling, prefix caching, chunked prefill, compiled execution, and CUDA graphs already exist. Their presence alone is not a new optimization opportunity.

## 1. Establish a repeatable baseline

**Relevant files:** `benchmarks/sweep.py`, `benchmarks/bench_serving.py`, `tests/test_bench_serving.py`, `tests/test_sweep.py`, and `docs/benchmark-runbook.md`.

- [x] Reproduce the existing workload and correctness gates from the runbook, using the intended checkout and recording its commit and working-tree patch. Treat historical branch names and dependency instructions as recorded setup, not an instruction to reset the workspace.
- [x] Run DeepSeek-V2-Lite at offered loads 1, 48, 64, and 80; run Qwen3-8B at 1, 32, 48, and 64. Use at least three runs per engine per point and the same trace within each comparison.
- [x] Record completed requests/s, output tokens/s, TTFT, TPOT, inter-token latency, end-to-end latency, failures, preemptions, KV usage, and GPU conditions. Preserve raw per-request results. Clocks could not be locked (the container lacks permission); busy SM clocks matched between engines within 0.2%.
- [x] Choose and record workload-specific TTFT and inter-token latency limits before evaluating changes. Add SLO goodput: requests satisfying both limits divided by total measured duration, including queue drain. Preserve the existing completion-rate metric separately; it currently has no latency cutoff. Limits chosen: TTFT ≤ 1.0 s and per-request TPOT ≤ 50 ms (`bench_serving.py --slo-ttft --slo-tpot`).
- [x] If extending benchmark metrics, add deterministic cases covering a request that passes both limits, fails either limit, fails execution, or has insufficient output tokens to measure an inter-token interval. Define the short-output policy explicitly in the report. A one-token answer is judged on TTFT alone.
- [x] Save a baseline report with per-run values and variability, not only an average.

**Deliverable:** A reproducible baseline and declared latency objectives. Small differences that overlap run-to-run variation remain inconclusive.

**Outcome:** DeepSeek-V2-Lite's lead at load 80 (+6.2%) holds across repetitions; loads 48 and 64 are within variation, and load-1 TPOT was 11.8% slower. Qwen3-8B trails by 1.4% at load 48 and 2.2% at load 64, outside variation.

## 2. Tune DeepSeek MoE kernels

**Relevant files:** `benchmarks/tune_moe.py`, `lean_vllm/layers/fused_moe.py`, `lean_vllm/layers/moe.py`, `tests/test_fused_moe.py`, and `tests/test_deepseek_v2.py`.

- [x] Measure the current MoE configuration across small decode and large prefill batches. Use the existing tuner on the target H100, initially covering batch sizes 1–4096.

```bash
uv run python benchmarks/tune_moe.py \
  --model ~/workspace/huggingface/DeepSeek-V2-Lite-Chat \
  --tune --save-dir results/moe-tuning-h100
```

- [x] Load the resulting configurations through `LEAN_VLLM_TUNED_CONFIG_FOLDER` and verify the server log confirms the selected file.
- [x] Check promising configurations against routing distributions captured from the real model. The current tuner uses random routing, which may rank configurations differently. Capture routing from real text: the warmup prompt's routing is 10.7× skewed onto one expert and ranks configs wrongly.
- [ ] Tune vLLM independently on the same GPU. Report untuned-to-tuned gains for each engine and tuned-to-tuned serving comparisons. Skipped: lean-vLLM's tuned configs were not shipped.
- [x] Run the relevant correctness checks before timing:

```bash
uv run pytest tests/test_fused_moe.py -k cuda -v
uv run pytest tests/test_deepseek_v2.py -v
```

- [x] Repeat the DeepSeek serving points from step 1, including load 1 to detect small-batch regressions.

**Decision:** Keep tuned configurations when they improve end-to-end performance repeatably. Only investigate routing, dispatch, expert GEMM, or overlap changes after the tuned profile identifies the remaining cost.

**Outcome:** Tuned configs rejected: 5–13% faster on prefill-sized batches in isolation, but no repeatable serving gain, and more first-use Triton compiles. Profiling batch 1 against vLLM found routing, block alignment and the top-k sum using large-batch tiles at every size. Sizing them to the batch (kept) cut the batch-1 decode step from 4.10 to 3.80 ms (vLLM 3.75 ms) and raised goodput at load 48 by 2.1%. The remaining batch-1 gap is in MLA attention and the dense GEMMs.

## 3. Isolate the Qwen GPU execution gap

**Relevant files:** `benchmarks/bench_offline.py`, `bench_fixed_batch.py` (kept out of the repository in `results/perf-2026-10-25/scripts/` and commit `b052aa5`), `lean_vllm/attention/flash_backend.py`, `lean_vllm/attention/triton_cache.py`, `lean_vllm/engine/model_runner.py`, and `lean_vllm/engine/compilation.py`.

- [x] Build a fixed-batch measurement harness. Match token IDs, batch size, per-sequence query/context lengths, precision, cache layout where supported, and graph padding across engines. Record differences that cannot be matched.
- [ ] Measure pure decode, cold prefill, cached-prefix prefill, and mixed batches separately. Start with Qwen3-0.6B to explain the existing trace, then validate conclusions on Qwen3-8B. Done for pure decode, cold prefill and cached prefix, directly on Qwen3-8B; mixed batches remain.
- [x] Compare attention, GEMMs, KV writes, normalization/activation kernels, output copies, metadata preparation, and graph padding. Report GPU elapsed time and overlap as well as aggregate kernel time.
- [x] Inspect FA3 call parameters and scheduling metadata. Measure whether the output copies in `flash_backend.py` materially contribute before changing buffer handling. The copy is about 0.5% of a step; removing it would rely on an undeclared mutation through a private FA3 API, so it stays.
- [ ] Select one measured bottleneck, implement a targeted change, and run the corresponding attention/model/compilation correctness checks. Re-run the fixed-batch case and the Qwen serving points. Blocked on the cause of the kernel spacing below.

**Decision:** Do not infer a kernel regression from independently scheduled traces or transfer the 0.6B result directly to 8B. A kernel improvement must survive serving validation.

**Outcome:** Prefill is equal; decode is 2.2–4.6% slower, all inside the CUDA graph. Kernels are spaced about 0.45 µs apart in lean-vLLM's graph against about 0.1 µs in vLLM's, with identical torch, CUDA, cuBLAS and Triton; the cause is not established. vLLM's own FlashAttention build is about 4.5% faster at batch 128, and no FA3 call parameter closes it. Host-side work between replays is equal.

## 4. Tune prefill scheduling and graph coverage

**Relevant files:** `lean_vllm/engine/scheduler.py`, `lean_vllm/engine/model_runner.py`, `lean_vllm/engine/compilation.py`, `lean_vllm/config.py`, `tests/test_scheduler.py`, `tests/test_model_runner.py`, and `tests/test_compilation.py`.

- [x] Sweep batch token budgets of 512, 1024, 2048, 4096, and 8192 with the current graph policy. Record prefill/mixed batch sizes, step times, TTFT, inter-token latency, throughput, and KV usage.
- [x] Independently test additional piecewise graph buckets at 768, 1024, and 2048 tokens where feasible. Measure capture time, graph memory, padding cost, and resulting KV capacity. Tested graph ranges to 1,024 and 2,048 tokens through `--max-cudagraph-capture-size`.
- [x] Combine promising budgets and graph ranges. Keep model weights and available memory budgets comparable across engines.
- [ ] If no fixed setting meets the chosen latency targets across load levels, design an adaptive prefill budget based on active decode work and measured step duration. Bound its range and test that waiting prefills continue to make progress. Not needed: a fixed 2,048 met the targets as well as any setting tried.
- [x] Validate token-budget accounting, graph bucket boundaries, padded rows, mixed-batch outputs, and unchanged pure-decode behavior.

**Decision:** Choose the throughput/latency tradeoff for the target workload. Smaller chunks can protect decode latency while reducing prefill efficiency; larger graphs can consume memory without improving execution.

**Outcome:** On Qwen3-8B, a 2,048-token budget with graphs to 2,048 tokens halves p99 inter-token latency at loads 32–48 and raises SLO goodput at load 48 by 19%. It costs 9 s of startup and 664 MiB. Defaults stay at vLLM's (8,192, graphs to 512) so the engines remain comparable; use `--max-num-batched-tokens 2048 --max-cudagraph-capture-size 2048` for latency-sensitive Qwen3-8B serving on an H100.

## 5. Evaluate SGLang-style cache-aware scheduling

**Relevant files:** `lean_vllm/engine/policy.py`, `lean_vllm/engine/scheduler.py`, `lean_vllm/engine/block_manager.py`, `benchmarks/bench_serving.py`, `tests/test_scheduler.py`, and `tests/test_block_manager.py`.

- [ ] Add deterministic shared-system-prompt, multi-turn, and branching-prompt workloads. Include unrelated prompts and cache-pressure cases. Measure cold-cache and warm-cache behavior separately.
- [ ] Establish prefix-hit rates, reused tokens, actual prefill work, and latency under the existing hash-based cache and FCFS policy.
- [ ] Prototype a bounded scan of waiting requests, preferring reusable prefixes while preserving explicit request priority and enforcing a maximum waiting age. Record scan overhead and cache-state changes between scheduling decisions.
- [ ] Test starvation prevention, priority ordering, prefix eviction, cache reference counts, and cancellation under asynchronous execution.
- [ ] Compare against FCFS and vLLM with prefix caching enabled on the same traces.
- [ ] Consider a radix-tree cache only if measurements identify a limitation in the hash-based design that the tree would address.

**Decision:** Retain cache-aware scheduling only when saved prefill work outweighs scheduling overhead and waiting-time costs. A radix tree alone does not establish an advantage over hash-based prefix caching.

**Status:** Not run, because shared-prefix serving is not the deployment target. Multi-turn and branching traces, unrelated prompts, and a prefix-cache hit-rate counter were drafted in `b052aa5` and removed in `2a5a0a6`; restore them from there if this step runs.

## 6. Evaluate speculative decoding for low-concurrency latency

**Relevant files:** `lean_vllm/spec_decode/`, `lean_vllm/models/deepseek_mtp.py`, `lean_vllm/engine/model_runner.py`, `tests/test_mtp.py`, `tests/test_spec_decode.py`, and `tests/test_pipeline.py`.

- [ ] Select a model with a compatible draft mechanism. Existing DeepSeek-V3 MTP support does not automatically apply to Qwen3-8B or DeepSeek-V2-Lite.
- [ ] Complete GPU correctness validation for draft generation, verification, rejection handling, KV state, and asynchronous execution before measuring speed.
- [ ] Measure accepted tokens per verification step, draft cost, verification cost, memory consumption, and end-to-end latency at low concurrency and near saturation.
- [ ] Sweep supported draft lengths. Compare with speculation disabled and, where supported, the same draft method in vLLM.
- [ ] Consider reducing draft length or disabling speculation under load only after identifying its measured break-even point.

**Decision:** Adopt speculation for workloads where reduced target iterations outweigh draft and verification costs. Treat quality-changing approximations as a separate comparison.

**Status:** Blocked. lean-vLLM's only drafter is DeepSeek-V3 MTP. Neither Qwen3-8B nor DeepSeek-V2-Lite has MTP layers, and V3 does not fit on one H100.

## Next steps

In order:

1. **Qwen3-8B kernel spacing.** Profile both engines' decode graphs with Nsight Systems to find why lean-vLLM's kernels are spaced about 0.45 µs apart against vLLM's 0.1 µs. This is the largest known part of the 2–5% decode gap.
2. **First-use Triton compiles.** Warm up each MoE config bucket at startup. The first server start after a code change shows 0.4–1.3 s TTFT on early requests.
3. **Mixed batches.** Finish the step 3 fixed-batch measurement for mixed prefill and decode batches.
4. **DeepSeek-V2-Lite at batch 1.** Profile MLA attention and the dense GEMMs, where the remaining 1.3% decode-step gap sits.
5. **Current vLLM.** Repeat the DeepSeek-V2-Lite load-80 comparison against a current stable vLLM before claiming the lead, as the constraints require.

Plan sections 5 and 6 stay conditional: run cache-aware scheduling if shared-prefix workloads become the deployment target, and speculation once a model with a supported draft mechanism fits the hardware.

## Reporting

For each experiment, save the hypothesis, commit/configuration changes, exact commands, software versions, correctness results, raw timings, repeated serving results, and a keep/reject/inconclusive decision. Re-profile after a successful change because the dominant bottleneck may move. Avoid adding distributed-serving machinery to address these single-GPU measurements.

## References

- [Benchmark runbook](benchmark-runbook.md): existing workload and GPU correctness gates.
- [Mini-SGLang](https://github.com/sgl-project/mini-sglang): compact reference for overlap scheduling, radix caching, and chunked prefill.
- [SGLang overlap scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/): scheduling ahead of GPU completion and resolving future-token dependencies.
- [SGLang paper](https://arxiv.org/abs/2312.07104): RadixAttention and cache-aware scheduling.
- [vLLM optimization guidance](https://docs.vllm.ai/en/latest/configuration/optimization/): prefill token-budget tradeoffs and performance tuning.
- [SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding): reference implementation options and model requirements.
