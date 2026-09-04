# Online Serving + Advanced Scheduler

Status: M0 and M1 landed. M2-M6 below are plan; this becomes the results
document as they land.

## Goal

Turn InferWeave from an offline batch generator into an online serving engine:
requests arrive at arbitrary times over HTTP, tokens stream back as they are
produced, and one token-budget scheduler decides every step what to run —
mixing chunked prefill with decode, with admission control, preemption, and a
pluggable fairness policy.

Done when `inferweave serve` speaks OpenAI-compatible SSE and frees a
disconnected client's KV blocks mid-generation; when one step can hold prefill
chunks and decode rows together with the backends still matching the dense
oracle; and when the same Poisson-arrival benchmark script runs against both
InferWeave and vLLM and this document publishes the resulting
goodput-versus-p99 curves.

Not in scope: multi-node, KV offload, disaggregated prefill (Project 4),
speculative decoding (Project 3), LoRA, beam search, `n > 1`.

## Problem

| current behaviour | why it blocks the goal |
|---|---|
| `generate()` adds every prompt up front, then loops `step()` to completion | no arrival process, no way to add a request mid-flight |
| `step()` returns only *finished* sequences | nothing to stream |
| `schedule()` returns `(seqs, is_prefill)`: a batch is all-prefill **or** all-decode, and prefill returns first unconditionally | one arriving prompt stalls every decoding request; the TTFT/TPOT trade-off cannot even be expressed |
| chunked prefill only for the first sequence, and `max_num_batched_tokens = 16384` | one long prompt blows the budget, and chunking never engages in practice |
| `assert scheduled_seqs` in the decode branch (`scheduler.py:71`) | crashes when memory pressure preempts every running sequence |
| `SamplingParams` forbids `temperature = 0` | OpenAI clients send it, and benchmarks want a deterministic decode |
| no request ids, cancellation, metrics, or incremental detokenization | nothing to build a server on |

Iteration-level scheduling already works: `schedule()` runs every step, so a
finished sequence leaves the batch and a waiting one takes its slot without
draining. What is missing for continuous batching is arrival mid-flight (M1,
M4) and mixing prefill with decode inside one step (M2, M3).

## Design

```text
                HTTP (FastAPI / uvicorn)          <- tokenize, detokenize, SSE
                          |
                   AsyncLLMEngine                 <- per-request asyncio.Queue
                          |    (thread boundary)
                   engine thread: step()
                          |
                      Scheduler                   <- one token budget per step
              +-----------+-----------+
           waiting                 running
              +-----------+-----------+
                          |
                   SchedulerOutput                <- [(seq, num_scheduled_tokens)]
                          |
                     ModelRunner                  <- one mixed batch
```

The engine stays synchronous and single-threaded. Everything asynchronous —
tokenization, detokenization, SSE — lives above the thread boundary, and new
requests and aborts cross it through queues drained at the top of `step()`,
never by mutating `Scheduler.waiting` from another thread. This is vLLM V1's
frontend / EngineCore split without the process boundary.

## Decisions

Where the design forks, what vLLM does and what this project does.

| question | vLLM | here |
|---|---|---|
| Split the token budget between prefill and decode? | V1 has one budget and schedules running before waiting, so decode is never starved. V0 split them. | Follow V1. The long-prompt knobs are the better lever. |
| Is chunked prefill optional? | V1 removed the flag; always on. | Keep `enable_chunked_prefill` anyway. It is the only honest way to get the "before" number. |
| Preemption by swap or recompute? | V1 dropped swapping; recompute, back to the head of the queue. | Recompute. Prefix caching means recompute often re-hits the blocks it just dropped — worth measuring rather than assuming. |
| Where does priority come from? | Client-supplied `priority` field, lower means earlier, arrival breaks ties. | Same. Deriving it server-side from prompt length was rejected: it turns the policy comparison into a comparison of two heuristics. |
| Backpressure | Unbounded queueing, no rejection. | Queue by default, but `max_waiting_requests` returns 429. Deliberately unlike vLLM — unbounded queueing converts overload into unbounded TTFT, and the benchmark can show it. |
| Engine loop: thread or process? | V1 uses a separate process with ZMQ to keep GIL contention off the engine. | A thread first: no serialization, no IPC, far less code. The process boundary is the known next move if Python work shows up in step time. |
| Metric names | `vllm:time_to_first_token_seconds`, `vllm:num_requests_running`, ... | Mirrored under `inferweave:`, so one dashboard reads both engines. |

## Milestones

Each is independently shippable and leaves the suite green.

### M0 — a GPU-free test harness — *done*

`tests/conftest.py` holds a `FakeModelRunner` with `ModelRunner`'s `call()`
surface, deterministic tokens and a record of every batch it was handed, plus a
config stub. `Scheduler` already reads plain attributes off its config, so this
needed no production change.

`tests/test_scheduler.py` pins 14 behaviours in 0.06s, three of them
limitations that name the milestone which changes them: prefill returns before
decode unconditionally, only the leading sequence may be chunked (so leftover
budget is wasted), and preempting the last running sequence trips
`assert scheduled_seqs`.

### M1 — request plumbing: ids, streaming, cancellation — *done*

- `engine/sequence.py`: `request_id`, arrival / first-scheduled / first-token /
  finish timestamps, `finish_reason`, `num_preemptions`, `stop_token_ids`.
- `engine/output.py` (new): `RequestOutput` — tokens new since the last step,
  delta text, finished flag, per-request metrics.
- `step()` returns a `RequestOutput` for every sequence that produced a token,
  not only finished ones; `generate()` keeps its signature by collecting them.
- `utils/detokenizer.py` (new): incremental detokenization with prefix/read
  offsets, as vLLM's `IncrementalDetokenizer` does. Re-decoding the whole id
  list per token is quadratic and splits multi-byte characters mid-token — a
  correctness fix, not an optimization.
- `Scheduler.abort(request_id)`: drop from waiting or running, deallocate, mark
  aborted. Applied between steps, never during one.
- `Sampler`: `temperature == 0` means argmax; drop the assert in
  `SamplingParams`.

The detokenizer is tested against a byte-level fake tokenizer, which reproduces
the split-character problem without a download, and additionally against the
real Qwen3 tokenizer when a model directory is present. `stop_token_ids` are
honoured even under `ignore_eos`, which covers the eos token only — vLLM's
semantics.

### M2 — unified token-budget scheduler

`SchedulerOutput` carrying `[(seq, num_scheduled_tokens)]` and the preemption
list replaces `(seqs, is_prefill)`. One pass per step: schedule running
sequences first (one token for decode, the next chunk for an unfinished
prefill), then admit from the waiting queue into the budget that is left.

- chunked prefill for any sequence, not only the first.
- fix the all-preempted crash; make victim selection a policy rather than
  `running.pop()`.
- policy object: FCFS (a deque, as today) or a heap on
  `(priority, arrival_time)`. A client can starve other clients; aging the
  effective priority by queue time is the fix, left out until the benchmark
  shows the starvation it would solve.
- `max_num_partial_prefills` and `long_prefill_token_threshold` for long-prompt
  fairness.
- admission control: `max_waiting_requests`, so the server can return 429.
- prefix-cache-aware admission: `block_manager.can_allocate` already returns the
  cached-block count; prefer high-hit prompts when the queue is deep.
- replace `running.remove(seq)` in `postprocess`, a linear scan per finish.

**Bridge to M3.** `SchedulerOutput` can describe a mixed batch before the runner
can execute one, so the engine splits a mixed output into two runner calls per
step — prefill chunks, then decode rows — and M3 collapses them into one call
and deletes the split. That costs an extra forward pass per step, so M2
publishes no numbers; in exchange every policy, preemption and admission
decision is testable against the *existing* runner, leaving M3 as "one batch
instead of two" gated by the differential tests. The split also forces two
invariants M3 needs anyway: a sequence appears in at most one sub-batch per
step, and only rows whose prefill completed may sample.

### M3 — mixed batches through the runner and backends

- `Context` always carries `query_start_loc`, `seq_lens` and `block_tables`, and
  gains `logits_indices`. `is_prefill` stops being the branch; the pure-decode
  fast path is selected by "every query length is 1".
- `ModelRunner.prepare_batch()` replaces `prepare_prefill` / `prepare_decode`.
- `ParallelLMHead` gathers logits from `context.logits_indices` rather than
  `cu_seqlens_q[1:] - 1`, since an unfinished chunk must not sample. The
  scheduler output therefore also carries the sampler-row-to-sequence map.
  Getting this wrong corrupts output silently, so it gets its own test.
- Backends gain a unified mixed-length path. FlashAttention needs only
  `flash_attn_varlen_func(..., block_table=...)` with per-sequence context
  lengths, a decode row being just `q_len == 1`; whether it wants `cu_seqlens_k`
  or `seqused_k` is version-dependent and gets verified against the pinned 2.8.3
  rather than assumed. `TorchAttention`'s loop already handles arbitrary `q_len`
  with bottom-right masking and simply always takes the paged path. `decode()`
  stays as the pure-decode fast path, keeping `flash_attn_with_kvcache` and
  CUDA graphs.
- New differential test: a mixed batch against the dense oracle, with the
  top-left-masking mutation asserted to fail it.
- Behind `enable_chunked_prefill`, so the old path survives for the A/B.
- `docs/attention-backends.md` updated in the same commit as the interface.

### M4 — async engine and HTTP server

- `engine/async_engine.py` (new): the sync engine on a dedicated thread; a
  per-request `asyncio.Queue` fed through `loop.call_soon_threadsafe`;
  `add_request()` returns an async generator whose `finally` calls `abort()`,
  which is what makes disconnect free KV blocks; the loop idles on an `Event`
  rather than spinning. A thread rather than the event loop because `step()`
  blocks in C for a whole forward pass and would starve the HTTP handlers,
  showing up as TTFT jitter.
- `entrypoints/api_server.py` (new): FastAPI and uvicorn, OpenAI SSE deltas
  terminated by `data: [DONE]`, chat template from the tokenizer, `/health`,
  `/v1/models`, `/metrics`. Prompts over `max_model_len` are refused with 400
  rather than asserting deep in the runner.
- Accepted: `model`, `prompt`/`messages`, `max_tokens`, `temperature` (0 is
  greedy), `stream`, `stream_options.include_usage`, `stop`, and the extras
  `ignore_eos` and `priority` — the latter rides in as an extra body field
  since the OpenAI schema has none. Everything else is refused with 400:
  `top_p`, `top_k`, `n > 1`, `best_of`, `logprobs`, penalties, `seed`,
  `logit_bias`, `tools`, `echo`, `suffix`. Silently ignoring `top_p` returns
  wrong output with no signal, which is worse than refusing it.
- `stop` as strings lives in the frontend: the engine stops on `stop_token_ids`
  and `max_tokens`, while string matching needs detokenized text. On a match the
  delta is truncated, the request aborted, `finish_reason` set to `stop`. A stop
  string can span token boundaries, so the tail is buffered up to the longest
  stop string — a deliberate small addition to streaming latency.
- New optional dependency group `serve` and a console script. Any flag that must
  reach the engine has to be a `Config` field: `LLMEngine.__init__` filters
  kwargs against `fields(Config)` and silently drops the rest.
- Server tests run against an in-process fake engine over httpx's ASGI
  transport, so no model is needed.

**When the engine thread dies.** The loop is wrapped; on an unhandled exception
it records the error, marks itself dead, and pushes that exception into every
outstanding request queue. Streams that already sent a 200 get an SSE error
event and a close, since a status code cannot be retracted. `/health` flips to
503, new requests are refused, and the process exits for a supervisor to
restart. No in-process restart: the CUDA context, KV cache and tensor-parallel
children do not survive, and rank 0 reaps those children through the existing
`atexit` path. vLLM takes the same posture.

### M5 — metrics

`engine/metrics.py` (new): per-request TTFT, TPOT, queue delay and E2E; gauges
for running and waiting depth, KV utilization, preemption count, prefix-cache
hit rate, per-step batch composition, graph-covered step fraction, and step
time. Prometheus at `/metrics`, plus a JSON summary for the benchmark.

On GPU utilization: the honest metric is the model-busy fraction of wall clock
from step timing. The `nvidia-smi` number is reported beside it, with a note on
why they differ.

### M6 — benchmarks and the write-up

- `benchmarks/bench_serving.py` (new): async client, trace generators (fixed,
  sampled ShareGPT-like, and a mixed short/long blend for the fairness story),
  Poisson arrivals at `--request-rate`, per-request percentiles, JSON output.
  The same script points at vLLM.
- `benchmarks/sweep.py` (new): rate sweep for the curve, plus the policy and
  budget comparisons.
- This document and the README roadmap row. `bench.py` stays as the offline
  throughput number, unchanged.

Accounting rules, so the numbers cannot flatter the engine: arrivals are
**open-loop** — request *i* is sent on schedule whatever is outstanding — and a
429 is **never retried**, because retrying converts a rejection into an
invisible queue and the offered load stops being the λ it claims to be.
Rejections are a reported outcome: goodput sits beside throughput, and any table
reporting percentiles also reports rejection rate, since percentiles cover
completed requests only and an engine shedding 90% of its load would otherwise
show an excellent p99. Non-429 failures are counted separately and abort the run
above a threshold; they are bugs, not backpressure.

## Blast radius

Most of the work is new files: `engine/output.py`, `engine/async_engine.py`,
`engine/metrics.py`, `engine/policy.py`, `utils/detokenizer.py`,
`entrypoints/api_server.py`, `benchmarks/*`, tests. Existing files change only
where the goal cannot be reached otherwise:

| file | change | why unavoidable |
|---|---|---|
| `engine/scheduler.py` | rewritten | it *is* the project |
| `engine/sequence.py` | added fields | request ids and timestamps have nowhere else to live |
| `engine/llm_engine.py` | `step()` return type; request/abort intake | streaming and cancellation |
| `engine/model_runner.py` | `prepare_batch` replaces the two prepares | mixed batches |
| `utils/context.py` | fields added, `is_prefill` demoted | mixed batches |
| `layers/embed_head.py` | logits from `logits_indices` | chunked prefill must not sample |
| `attention/*` | mixed-length path added, `decode()` kept | mixed batches |
| `layers/sampler.py` | greedy branch | `temperature = 0` |
| `sampling_params.py` | drop the assert, add `stop_token_ids`, priority | server semantics |
| `config.py` | new fields | the only channel into the engine |
| `pyproject.toml` | `serve` extra, console script | packaging |

`models/qwen3.py`, the remaining layers, `engine/block_manager.py` and
`utils/loader.py` should not change at all. If `block_manager.py` starts needing
edits, the scheduler is reaching through an abstraction.

## Hardware

Every number comes from one **A100 80GB SXM**, single GPU, running **Qwen3-8B**
for headline results and Qwen3-0.6B for laptop development and CI.

The pinned `flash-attn==2.8.3` wheel is a `cu12torch2.9` build targeting
sm80-sm90. On Blackwell the import succeeds and `torch.cuda.is_available()` is
true, so `FlashAttentionBackend.is_available()` returns True and the kernels
fail below that check — the selector's refusal to downgrade silently does not
protect against a backend that is importable but unsupported. Ampere also keeps
one hardware footnote in the repo, since `docs/attention-backends.md` validated
numerics on an A100. 8B of bf16 weights leaves ~60GB of KV while being large
enough that a step is not dominated by scheduler Python: at 0.6B a decode step
is 1-3ms and a vLLM comparison would largely measure interpreter speed against
vLLM's multi-process frontend.

L40S 48GB is the budget substitute at half the price, though ~864 GB/s against
the A100's ~2 TB/s makes decode-bound TPOT incomparable to published numbers.
H100 costs 2-3x and at this model size only makes the engine more CPU-bound.
24GB consumer cards and ROCm are out.

Tensor parallel is exercised on a 2xA100 node for the smoke test only; sweeps
run single-GPU. The results table records driver, torch and **vLLM version** —
vLLM's scheduler changed substantially between V0 and V1, so a bare "vs vLLM"
number ages badly.

## Measurement plan

Baseline first: before M2 lands, record the current engine under the new harness
with `enable_chunked_prefill=False` — the "before" column the README promises.
Axes to sweep: request rate, `max_num_batched_tokens`, scheduling policy,
chunked prefill on/off, and prompt/output length mix.

KV pressure is set deliberately, not hoped for. Qwen3-0.6B is 112 KiB per token,
so 40GB of cache holds ~374k tokens — far beyond any sane arrival rate on one
card. Results are published at both a comfortable and a cache-thrashing
`num_kvcache_blocks`, so preemption and admission control are exercised rather
than merely implemented.

The chunked-prefill A/B runs four times, not twice. Chunking on produces mixed
steps that run eager, while chunking off leaves pure-decode steps that capture
CUDA graphs, so a two-run A/B fuses the scheduling change with lost graph
coverage. `enforce_eager=True` on both arms isolates scheduling and is the
primary result; the default configuration on both arms is the
deployment-realistic result, and the delta between the pairs is the graph
effect, reported on its own line.

Two confounds to control for:

- `warmup_model` sizes its warmup batch from `max_num_batched_tokens`, and on
  CUDA `kvcache_bytes()` derives the cache size by subtracting peak allocated
  memory — so changing the token budget silently changes the number of KV
  blocks. Pin `num_kvcache_blocks` for every run.
- vLLM queues without bound, so its rejection rate is zero by construction and
  overload lands in its p99; InferWeave sheds load and moves the same pressure
  into its rejection rate. Compare on goodput-versus-p99, never p99 alone.

## Risks

1. M3 touches the attention backends, the LM head and CUDA-graph capture at
   once. Mitigation: config flag, old path retained, differential tests as gate.
2. Real numbers need a GPU; laptop runs prove correctness only.
3. `torch.compile` on `Sampler` with a per-step-varying batch size may recompile
   constantly. Check, and mark dynamic if so.
4. Tensor parallel is exercised by none of this, and rank 0 is the only rank
   that samples. Keep a TP smoke test in the loop or the path will rot.
