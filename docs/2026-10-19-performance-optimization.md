# lean-vLLM Performance Optimization Plan

> **For agentic workers:** Use `superpowers:executing-plans` to execute each experiment and review its evidence before proceeding. Use `superpowers:subagent-driven-development` if parallel agent execution is requested. Track progress with the checkboxes below.

**Goal:** Establish repeatable performance wins over vLLM on specified models and workloads, while preserving generation correctness and lean-vLLM's small implementation.

**Architecture:** Optimize the existing execution path first: MoE kernels, attention calls, prefill scheduling, and CUDA graph coverage. Evaluate cache-aware scheduling and speculative decoding as separate workload-specific experiments after establishing reliable baselines.

**Tech stack:** Python, PyTorch/Inductor, Triton, CUDA graphs, FlashAttention-3, FlashMLA, and the existing serving benchmark harness; initially one NVIDIA H100 80 GB.

**Spec:** The optimization recommendations discussed on 19 October 2026, captured below. Supporting evidence: the [Qwen3-8B report](benchmark-2026-10-17-Qwen3-8B.md), [DeepSeek-V2-Lite report](benchmark-2026-10-17-DeepSeek-V2-Lite.md), and [results table](benchmark-results-2026-10-17.md).

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

- [ ] Reproduce the existing workload and correctness gates from the runbook, using the intended checkout and recording its commit and working-tree patch. Treat historical branch names and dependency instructions as recorded setup, not an instruction to reset the workspace.
- [ ] Run DeepSeek-V2-Lite at offered loads 1, 48, 64, and 80; run Qwen3-8B at 1, 32, 48, and 64. Use at least three runs per engine per point and the same trace within each comparison.
- [ ] Record completed requests/s, output tokens/s, TTFT, TPOT, inter-token latency, end-to-end latency, failures, preemptions, KV usage, and GPU conditions. Preserve raw per-request results.
- [ ] Choose and record workload-specific TTFT and inter-token latency limits before evaluating changes. Add SLO goodput: requests satisfying both limits divided by total measured duration, including queue drain. Preserve the existing completion-rate metric separately; it currently has no latency cutoff.
- [ ] If extending benchmark metrics, add deterministic cases covering a request that passes both limits, fails either limit, fails execution, or has insufficient output tokens to measure an inter-token interval. Define the short-output policy explicitly in the report.
- [ ] Save a baseline report with per-run values and variability, not only an average.

**Deliverable:** A reproducible baseline and declared latency objectives. Small differences that overlap run-to-run variation remain inconclusive.

## 2. Tune DeepSeek MoE kernels

**Relevant files:** `benchmarks/tune_moe.py`, `lean_vllm/layers/fused_moe.py`, `lean_vllm/layers/moe.py`, `tests/test_fused_moe.py`, and `tests/test_deepseek_v2.py`.

- [ ] Measure the current MoE configuration across small decode and large prefill batches. Use the existing tuner on the target H100, initially covering batch sizes 1–4096.

```bash
uv run python benchmarks/tune_moe.py \
  --model ~/workspace/huggingface/DeepSeek-V2-Lite-Chat \
  --tune --save-dir results/moe-tuning-h100
```

- [ ] Load the resulting configurations through `LEAN_VLLM_TUNED_CONFIG_FOLDER` and verify the server log confirms the selected file.
- [ ] Check promising configurations against routing distributions captured from the real model. The current tuner uses random routing, which may rank configurations differently.
- [ ] Tune vLLM independently on the same GPU. Report untuned-to-tuned gains for each engine and tuned-to-tuned serving comparisons.
- [ ] Run the relevant correctness checks before timing:

```bash
uv run pytest tests/test_fused_moe.py -k cuda -v
uv run pytest tests/test_deepseek_v2.py -v
```

- [ ] Repeat the DeepSeek serving points from step 1, including load 1 to detect small-batch regressions.

**Decision:** Keep tuned configurations when they improve end-to-end performance repeatably. Only investigate routing, dispatch, expert GEMM, or overlap changes after the tuned profile identifies the remaining cost.

## 3. Isolate the Qwen GPU execution gap

**Relevant files:** `benchmarks/bench_offline.py`, proposed `benchmarks/bench_fixed_batch.py`, `lean_vllm/attention/flash_backend.py`, `lean_vllm/attention/triton_cache.py`, `lean_vllm/engine/model_runner.py`, and `lean_vllm/engine/compilation.py`.

- [ ] Build a fixed-batch measurement harness. Match token IDs, batch size, per-sequence query/context lengths, precision, cache layout where supported, and graph padding across engines. Record differences that cannot be matched.
- [ ] Measure pure decode, cold prefill, cached-prefix prefill, and mixed batches separately. Start with Qwen3-0.6B to explain the existing trace, then validate conclusions on Qwen3-8B.
- [ ] Compare attention, GEMMs, KV writes, normalization/activation kernels, output copies, metadata preparation, and graph padding. Report GPU elapsed time and overlap as well as aggregate kernel time.
- [ ] Inspect FA3 call parameters and scheduling metadata. Measure whether the output copies in `flash_backend.py` materially contribute before changing buffer handling.
- [ ] Select one measured bottleneck, implement a targeted change, and run the corresponding attention/model/compilation correctness checks. Re-run the fixed-batch case and the Qwen serving points.

**Decision:** Do not infer a kernel regression from independently scheduled traces or transfer the 0.6B result directly to 8B. A kernel improvement must survive serving validation.

## 4. Tune prefill scheduling and graph coverage

**Relevant files:** `lean_vllm/engine/scheduler.py`, `lean_vllm/engine/model_runner.py`, `lean_vllm/engine/compilation.py`, `lean_vllm/config.py`, `tests/test_scheduler.py`, `tests/test_model_runner.py`, and `tests/test_compilation.py`.

- [ ] Sweep batch token budgets of 512, 1024, 2048, 4096, and 8192 with the current graph policy. Record prefill/mixed batch sizes, step times, TTFT, inter-token latency, throughput, and KV usage.
- [ ] Independently test additional piecewise graph buckets at 768, 1024, and 2048 tokens where feasible. Measure capture time, graph memory, padding cost, and resulting KV capacity.
- [ ] Combine promising budgets and graph ranges. Keep model weights and available memory budgets comparable across engines.
- [ ] If no fixed setting meets the chosen latency targets across load levels, design an adaptive prefill budget based on active decode work and measured step duration. Bound its range and test that waiting prefills continue to make progress.
- [ ] Validate token-budget accounting, graph bucket boundaries, padded rows, mixed-batch outputs, and unchanged pure-decode behavior.

**Decision:** Choose the throughput/latency tradeoff for the target workload. Smaller chunks can protect decode latency while reducing prefill efficiency; larger graphs can consume memory without improving execution.

## 5. Evaluate SGLang-style cache-aware scheduling

**Relevant files:** `lean_vllm/engine/policy.py`, `lean_vllm/engine/scheduler.py`, `lean_vllm/engine/block_manager.py`, `benchmarks/bench_serving.py`, `tests/test_scheduler.py`, and `tests/test_block_manager.py`.

- [ ] Add deterministic shared-system-prompt, multi-turn, and branching-prompt workloads. Include unrelated prompts and cache-pressure cases. Measure cold-cache and warm-cache behavior separately.
- [ ] Establish prefix-hit rates, reused tokens, actual prefill work, and latency under the existing hash-based cache and FCFS policy.
- [ ] Prototype a bounded scan of waiting requests, preferring reusable prefixes while preserving explicit request priority and enforcing a maximum waiting age. Record scan overhead and cache-state changes between scheduling decisions.
- [ ] Test starvation prevention, priority ordering, prefix eviction, cache reference counts, and cancellation under asynchronous execution.
- [ ] Compare against FCFS and vLLM with prefix caching enabled on the same traces.
- [ ] Consider a radix-tree cache only if measurements identify a limitation in the hash-based design that the tree would address.

**Decision:** Retain cache-aware scheduling only when saved prefill work outweighs scheduling overhead and waiting-time costs. A radix tree alone does not establish an advantage over hash-based prefix caching.

## 6. Evaluate speculative decoding for low-concurrency latency

**Relevant files:** `lean_vllm/spec_decode/`, `lean_vllm/models/deepseek_mtp.py`, `lean_vllm/engine/model_runner.py`, `tests/test_mtp.py`, `tests/test_spec_decode.py`, and `tests/test_pipeline.py`.

- [ ] Select a model with a compatible draft mechanism. Existing DeepSeek-V3 MTP support does not automatically apply to Qwen3-8B or DeepSeek-V2-Lite.
- [ ] Complete GPU correctness validation for draft generation, verification, rejection handling, KV state, and asynchronous execution before measuring speed.
- [ ] Measure accepted tokens per verification step, draft cost, verification cost, memory consumption, and end-to-end latency at low concurrency and near saturation.
- [ ] Sweep supported draft lengths. Compare with speculation disabled and, where supported, the same draft method in vLLM.
- [ ] Consider reducing draft length or disabling speculation under load only after identifying its measured break-even point.

**Decision:** Adopt speculation for workloads where reduced target iterations outweigh draft and verification costs. Treat quality-changing approximations as a separate comparison.

## Execution order and reporting

Start with baseline repetition and DeepSeek MoE tuning, then the fixed-batch Qwen investigation. Run prefill/graph experiments next. Prioritize cache-aware scheduling if shared-prefix workloads are the intended deployment; prioritize speculation if low-concurrency generation latency is the goal.

For each experiment, save the hypothesis, commit/configuration changes, exact commands, software versions, correctness results, raw timings, repeated serving results, and a keep/reject/inconclusive decision. Re-profile after a successful change because the dominant bottleneck may move. Avoid adding distributed-serving machinery to address these single-GPU measurements.

## References

- [Benchmark runbook](benchmark-runbook.md): existing workload and GPU correctness gates.
- [Mini-SGLang](https://github.com/sgl-project/mini-sglang): compact reference for overlap scheduling, radix caching, and chunked prefill.
- [SGLang overlap scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/): scheduling ahead of GPU completion and resolving future-token dependencies.
- [SGLang paper](https://arxiv.org/abs/2312.07104): RadixAttention and cache-aware scheduling.
- [vLLM optimization guidance](https://docs.vllm.ai/en/latest/configuration/optimization/): prefill token-budget tradeoffs and performance tuning.
- [SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding): reference implementation options and model requirements.
