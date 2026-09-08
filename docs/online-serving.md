# Online Serving + Advanced Scheduler

Status: M0-M6 have landed, except for two M2 items (see M2). The M6 scripts
run, but no numbers are published: every result in the measurement plan needs
the A100, and the engine has not been on one since the fork.

## Goal

Turn lean-vLLM from an offline batch generator into an online serving engine:
requests arrive at arbitrary times over HTTP, tokens stream back as they are
produced, and one token-budget scheduler decides every step what to run —
mixing chunked prefill with decode, with admission control, preemption, and a
pluggable fairness policy.

Done when three things are true:

1. `lean-vllm serve` speaks OpenAI-compatible SSE, and a disconnected client's
   KV blocks are freed mid-generation.
2. One step can hold prefill chunks and decode rows together, with the backends
   still matching the dense oracle.
3. One Poisson-arrival benchmark script runs against both lean-vLLM and vLLM,
   and this document publishes the resulting curves: goodput (completed
   requests per second, not counting rejections) against p99 latency.

Not in scope: multi-node, KV offload, disaggregated prefill (Project 4),
speculative decoding (Project 3), LoRA, beam search, `n > 1`.

## Where this started

| behaviour before M0 | why it blocked the goal |
|---|---|
| `generate()` adds every prompt up front, then loops `step()` to completion | no arrival process, no way to add a request mid-flight |
| `step()` returns only *finished* sequences | nothing to stream |
| `schedule()` returns `(seqs, is_prefill)`: a batch is all-prefill **or** all-decode, and prefill returns first unconditionally | one arriving prompt stalls every decoding request; the TTFT/TPOT trade-off cannot even be expressed |
| chunked prefill only for the first sequence, and `max_num_batched_tokens = 16384` | one long prompt blows the budget, and chunking never engages in practice |
| `assert scheduled_seqs` in the decode branch | crashes when memory pressure preempts every running sequence |

Iteration-level scheduling already worked: `schedule()` runs every step, so a
finished sequence leaves the batch and a waiting one takes its slot without
draining. What was missing for continuous batching was arrival mid-flight (M1,
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
| Admission control | Unbounded queueing, no rejection. | Queue by default, matching vLLM, so the M6 comparison runs the same discipline on both sides. `max_waiting_requests` (429 at arrival) and `request_timeout` (504 after waiting too long) are opt-in, because unbounded queueing converts overload into unbounded TTFT and the benchmark can show it. |
| Engine loop: thread or process? | V1 uses a separate process with ZMQ to keep GIL contention off the engine. | A thread first: no serialization, no IPC, far less code. The process boundary is the known next move if Python work shows up in step time. |
| Metric names | `vllm:time_to_first_token_seconds`, `vllm:num_requests_running`, ... | Mirrored under `lean_vllm:`, so one dashboard reads both engines. |

## Milestones

Each is independently shippable and leaves the suite green.

### M0 — a GPU-free test harness — *done*

`tests/conftest.py` holds a `FakeModelRunner` with `ModelRunner`'s `call()`
surface, deterministic tokens and a record of every batch it was handed.
`Scheduler` already reads plain attributes off its config, so this needed no
production change. Three of the pinned behaviours were limitations naming the
milestone that would change them.

### M1 — request plumbing: ids, streaming, cancellation — *done*

- `engine/output.py` (new): `RequestOutput` — tokens new since the last step,
  delta text, finished flag, per-request metrics. `step()` returns one for every
  sequence that produced a token, not only finished ones.
- `utils/detokenizer.py` (new): incremental detokenization with prefix/read
  offsets. Re-decoding the whole id list per token is quadratic *and* splits
  multi-byte characters mid-token — a correctness fix, not an optimization. It
  is tested against a byte-level fake tokenizer, which reproduces the split
  without a download.
- `Scheduler.abort(request_id)` applies between steps, never during one.
- `temperature == 0` means argmax; the assert forbidding it is gone.
- `stop_token_ids` are honoured even under `ignore_eos`, which covers the eos
  token only — vLLM's semantics.

### M2 — unified token-budget scheduler — *done*

`SchedulerOutput` carrying `[(seq, num_scheduled_tokens)]` and the preemption
list replaces `(seqs, is_prefill)`. One pass per step: schedule running
sequences first (one token for decode, the next chunk for an unfinished
prefill), then admit from the waiting queue into the budget that is left.
Chunked prefill applies to any sequence; victim selection is a policy object
(FCFS deque, or a heap on `(priority, arrival_time)`) rather than
`running.pop()`.

`long_prefill_token_threshold` caps how many tokens one prompt may take from a
single step's budget, which is what actually stops a long prompt from starving
short ones; `max_num_partial_prefills` caps how many prompts may be mid-chunk at
once. vLLM splits the same concern differently, marking prompts over a threshold
as "long" and capping those separately.

A sequence that is alone in the cache and still cannot grow is dropped with
`finish_reason="capacity"` rather than preempted forever. The old code asserted
here instead. Prompts that exceed total cache capacity are also dropped at
scheduling time, so smaller requests behind them can proceed even with no
request timeout configured.

One client can still starve others. The fix is to age a request's effective
priority by how long it has queued — left out until the benchmark shows the
starvation it would solve.

Two bullets **not done**: prefix-cache-aware admission (it reorders the waiting
queue, so it belongs with the policy object rather than the admission loop), and
replacing the linear `running.remove(seq)` in `postprocess` (not worth doing
until a profile says the deque scan costs something).

### M3 — mixed batches through the runner and backends — *done*

- `ModelRunner.prepare_batch()` replaces `prepare_prefill` / `prepare_decode`.
  `Context` gains `logits_indices`, and `cu_seqlens_q` / `cu_seqlens_k` /
  `context_lens` / `block_tables` are now filled on every step.
- `is_prefill` survives as a field name but means "take the varlen path", set
  from `any(seq.is_prefill)` — **not** from "every query length is 1", which
  would wrongly send a one-token final prompt chunk down the decode path.
- `ParallelLMHead` gathers logits from `context.logits_indices` rather than
  `cu_seqlens_q[1:] - 1`, since an unfinished chunk must not sample. Getting
  this wrong corrupts output silently, so it has its own test.
- Backends needed **no change**: `prefill` was already the unified varlen path
  with bottom-right masking, and a decode row is just a row of query length 1.
  This was the milestone's main risk and it evaporated on contact. `decode()`
  stays as the pure-decode fast path, keeping `flash_attn_with_kvcache` and
  CUDA graphs.
- `enable_chunked_prefill=False` restores the pre-M2 shape — whole prompts only,
  never mixed with decode — so the A/B has a "before" arm.
- Verified on Qwen3-0.6B: greedy output is identical token-for-token with
  `max_num_batched_tokens=16` (a ~150-token prompt in ~10 chunks) and with 4096
  (no chunking at all).

### M4 — async engine and HTTP server — *done*

`engine/async_engine.py`, `entrypoints/{api_server,protocol,stop_checker,server,cli}.py`.

- A thread rather than the event loop, because `step()` blocks in C for a whole
  forward pass and would starve the HTTP handlers, showing up as TTFT jitter.
  The loop idles on an `Event` rather than spinning.
- Traffic in crosses the boundary the same way traffic out does: adds and aborts
  go through one `SimpleQueue` drained at the top of each step, so nothing locks
  the scheduler and the two stay ordered. `add_request()` is a coroutine
  *returning* the generator, not a generator itself, and awaits an acceptance
  future the engine thread settles — so a 429 is decided in microseconds, before
  any status code is sent, rather than at the first token. The generator's
  `finally` aborts, which is what makes a disconnect free KV blocks.
- Unsupported sampling parameters are refused with 400 — `top_p`, `top_k`,
  `n > 1`, `best_of`, `logprobs`, penalties, `seed`, `logit_bias`, `tools`,
  `echo`, `suffix`. Silently ignoring `top_p` returns wrong output with no
  signal, which is worse than refusing it. Refused *by value*, though, not by
  presence: `"top_p": 1.0` asks for nothing, and a client that fills in OpenAI's
  defaults should not be punished for a field it never set.
- `stop` as strings lives in the frontend: the engine stops on `stop_token_ids`
  and `max_tokens`, while string matching needs detokenized text. A stop string
  can span token boundaries, so the tail is buffered up to the longest stop
  string — a deliberate small addition to streaming latency.
- Engine flags are *generated* from `fields(Config)` rather than listed, since
  `LLMEngine.__init__` filters kwargs against exactly that set and silently
  drops the rest. A field added to `Config` is a flag on the next run.
- `/metrics` is **deferred to M5**: nothing aggregates counters until
  `engine/metrics.py` exists, and an endpoint serving zeroes is worse than none.

**Two forms of admission control, both off by default.** `max_waiting_requests`
refuses a request on arrival with a 429. `request_timeout` drops one that has
waited that long without ever being scheduled, and the server answers 504. The
timeout only touches sequences that have never run: a preempted sequence has
tokens to show for itself, and `max_tokens` already bounds a request that is
running. Both stay off for the M6 comparison, so neither engine sheds what the
other queues.

**A request could finish without ever producing an output.** The scheduler can
drop a request that will never fit (`finish_reason="capacity"`) without it ever
reaching the sampler, so nothing was pushed to the stream and the client hung
forever. Offline `generate()` never noticed, because it polls `is_finished()`.
`SchedulerOutput` now carries `dropped` and `LLMEngine` emits a final output for
each. The server treats `capacity` as an error rather than an OpenAI finish
reason: a 503 before the stream opens, an SSE error event after.

**When the engine thread dies.** The loop is wrapped; on an unhandled exception
it records the error, marks itself dead, and pushes that exception into every
outstanding request queue. Streams that already sent a 200 get an SSE error
event and a close, since a status code cannot be retracted. `/health` flips to
503, new requests are refused, and the process exits for a supervisor to
restart. No in-process restart: the CUDA context, KV cache and tensor-parallel
children do not survive. vLLM takes the same posture. This path was exercised
for real before it was tested — a malformed prompt reached the runner and the
server behaved exactly as described.

**Two bugs the end-to-end run found.** `apply_chat_template(tokenize=True)`
returns a `BatchEncoding` on current transformers, not a list of ids, so the
call now passes `return_dict=False`. And `LLMEngine.add_request` registered the
detokenizer after `scheduler.add`, so a request the scheduler refused left an
orphan behind; the scheduler is now the last thing touched.

Verified on Qwen3-0.6B: greedy completions and chat, SSE streaming with
`include_usage`, four concurrent requests, a stop string truncating mid-delta, a
client hanging up mid-stream with the server healthy after, and each 400/429/503
above.

### M5 — metrics — *done*

`engine/metrics.py` (new), with `/metrics` (Prometheus text) and `/metrics.json`
(the summary the benchmark records beside its own client-side numbers).

- Hand-rolled rather than `prometheus_client`: three metric types and a renderer
  is less code than the dependency, and the same registry produces the JSON
  summary. Names mirror vLLM's under `lean_vllm:`.
- The engine thread records, the HTTP handler renders, and one lock covers both
  — otherwise a scrape can catch a histogram between its bucket and its sum.
- Per request, at finish: TTFT, TPOT, queue delay, E2E, prompt and completion
  length. Counters for received / rejected / aborted / finished-by-reason, so
  M6's goodput and rejection rate come off the server as well as the client.
- Per step: duration, batch size, prefill-vs-decode token split, and whether the
  step replayed a CUDA graph. A step that scheduled nothing is not counted as a
  forward pass; counting the idle poll would inflate the step count and deflate
  the busy fraction.
- Prefix-cache hit rate is counted in *blocks* at admission, where
  `can_allocate`'s return value already says how many the cache supplied.
  Extracting `Scheduler._admit()` for that also removed the duplicate admission
  body the two schedule paths were carrying.
- The scheduler does not know about metrics. It reports what it did on
  `SchedulerOutput` — now including `num_queried_blocks` / `num_cached_blocks` —
  and `LLMEngine.step()` records. Keeping the recording out of the scheduler is
  what lets the test harness exercise it with no model.

The summary reports counts, sums and means, and no percentiles: the histogram
buckets are too coarse to interpolate a percentile from without lying about it,
and M6's client measures the real ones. Rates are `None` rather than `0.0`
before anything has happened, since a hit rate of zero and no queries at all are
different claims.

On GPU utilization: the honest metric is `model_busy_fraction`, the share of
wall clock spent inside a step. `nvidia-smi` is reported beside it as
`gpu_utilization_percent_nvidia_smi` and counts any live kernel as busy, so it
reads high even when the batch is one row wide. It is there to be compared, not
believed.

Verified on Qwen3-0.6B, one 48-token chat completion, scraped from both
endpoints: `num_requests_received_total 1`, `request_success_total{finish_reason="length"} 1`,
`prompt_tokens_total 13`, `generation_tokens_total 48`, `num_steps_total 48`.
The split is `prefill: 13, decode: 47` — the prefill step samples the first
token, so decodes run one behind generations, which is the arithmetic the unit
tests assert.

That run also shows what M6 is for: `mean_step_seconds` 0.99 and
`mean_batch_tokens` 1.25, a full forward pass per token with a single-row batch.
`model_busy_fraction` was 0.81 over a 58s uptime that included idle time before
the request arrived. `graph_step_fraction` is 0.0 and the `nvidia-smi` reading
`null`, both correct on a machine with no CUDA.

Still open: `init_process_group` binds a hardcoded `localhost:2333` even at
`tensor_parallel_size=1`, so only one engine can exist per machine.

### M6 — benchmarks and the write-up — *scripts done, numbers pending*

`benchmarks/bench_serving.py` (new) is the client: Poisson arrivals at
`--request-rate`, four traces, per-request percentiles, JSON out. It points at
vLLM unchanged.

```bash
uv run python benchmarks/bench_serving.py --dataset lognormal --request-rate 8
uv run python benchmarks/sweep.py --model ~/huggingface/Qwen3-8B \
    --suite rate --rates 1,2,4,8,16 --kvcache-tokens 330000 --out results/8b
```

`benchmarks/sweep.py` (new) drives it. An arm is one server configuration, and
every run — one arm at one rate — gets a freshly started server, so the
`/metrics.json` captured beside it describes that run rather than the one
before, and no run inherits the block pool the last one left. Runs go one at a
time, because `init_process_group` binds a fixed port and only one engine fits
on a machine. Five suites: `rate` (the curve), `chunked` (the four-run A/B),
`budget`, `policy`, and `starvation`.

Accounting rules, so the numbers cannot flatter the engine: arrivals are
**open-loop** — request *i* is sent on schedule whatever is outstanding — and a
429 is **never retried**, because retrying converts a rejection into an
invisible queue and the offered load stops being the λ it claims to be.
Rejections are a reported outcome: goodput sits beside throughput, and any table
reporting percentiles also reports rejection rate, since percentiles cover
completed requests only and an engine shedding 90% of its load would otherwise
show an excellent p99. Non-429 failures are counted separately and abort the run
above a threshold; they are bugs, not admission control.

- **Prompts are random token ids over `/v1/completions`.** Lengths are then
  exactly what the trace says, with no chat template varying by model, and no
  two prompts share a prefix — so the prefix cache cannot quietly supply half
  the blocks in a run that was never about caching. `--dataset sharegpt` reads a
  real ShareGPT file when the question *is* about real length distributions;
  `lognormal` gives that shape without the download.
- **A chunk is not a token.** Incremental detokenization holds bytes back, so
  the SSE deltas undercount. TPOT is computed from `usage.completion_tokens`,
  and the per-chunk gaps are reported separately as ITL.
- **The client is the official OpenAI SDK, with `max_retries=0`.** Driving the
  client a user would drive means the benchmark exercises the protocol the
  engine actually serves, and it deletes the hand-rolled SSE parsing. The retry
  default is the trap: the SDK retries a 429 twice, which would turn every
  rejection into exactly the invisible queue the accounting rules forbid.
  Verified against a server with `max_waiting_requests=1` — 10 offered, 2
  completed, 8 rejected, and the engine's own counters read 2 received and 8
  rejected, so nothing was retried. The cost is a parsed model per chunk rather
  than a `json.loads`, client-side work the old path did not do.
- **The connection pool is raised to 8192.** The SDK defaults to 1000 and httpx
  to 100; either would queue arrivals inside the client at a high rate and turn
  the open loop into a closed one — a measurement that looks fine and is wrong.
- **`--dataset mixed` labels each request `short` or `long`, and the summary
  reports each class on its own.** Starvation does not show up in an aggregate:
  mean TTFT can sit still while every short request waits behind a long prefill.
  The same breakdown carries the policy comparison, since `--long-priority 1`
  is the only thing `--scheduling-policy priority` has to act on.
- **`num_kvcache_blocks` is now a `serve` flag.** It was internal, on the
  grounds that profiling derives it. But profiling derives it from what the
  weights and the warmup batch left over, and the warmup batch is sized from
  `max_num_batched_tokens` — so the budget sweep would have changed the cache
  size underneath itself. The measurement plan says pin it; now it can be.
- **The sweep pins that cache in tokens, not blocks.** A lean-vLLM block holds
  256 tokens and a vLLM block holds 16, so handing both engines the same block
  count would have given vLLM a sixteenth of the cache — an unfair comparison
  that no output would have flagged. `--kvcache-tokens` converts, and pins each
  engine's block size so the arithmetic cannot drift.
- **The client reads the model id off `/v1/models`.** vLLM answers 404 to a
  request naming a model it does not serve, so a hardcoded default breaks that
  arm at the first request. lean-vLLM answers 404 now too: ignoring the field
  made it easy to point a benchmark at the wrong server and never find out.

Verified on Qwen3-0.6B on MPS: a `rate` sweep and a two-arm `starvation` sweep
end to end, server started and stopped per arm, every request completed, and
`/metrics.json` captured either side of each run. `tests/test_bench_serving.py`
pins the accounting against the fake engine — a 429 counted as a rejection and
never retried, a 503 counted as a failure that aborts the run, percentiles taken
over completed requests only.

What is not done is the point of the milestone: the numbers. Every row of the
measurement plan below needs the A100, and this document publishes no curve
until it runs on one.

## Blast radius

Most of the work is new files under `engine/`, `entrypoints/`, `utils/` and
`benchmarks/`. `models/qwen3.py`, the remaining layers, `engine/block_manager.py`
and `utils/loader.py` should not change at all. If `block_manager.py` starts
needing edits, the scheduler is reaching through an abstraction.

## Hardware

Every number comes from one **A100 80GB SXM**, single GPU, running **Qwen3-8B**
for headline results and Qwen3-0.6B for laptop development and CI. 8B of bf16
weights leaves ~60GB of KV while being large enough that a step is not dominated
by scheduler Python: at 0.6B a decode step is 1-3ms and a vLLM comparison would
largely measure interpreter speed against vLLM's multi-process frontend.

Not Blackwell. The pinned `flash-attn==2.8.3` wheel is a `cu12torch2.9` build
targeting sm80-sm90. On Blackwell it still imports, and `torch.cuda.is_available()`
is still true — so the backend selector's own `is_available()` says yes and picks
flash-attn. The kernels then fail at launch, below the point where anything
checked. Refusing to downgrade silently does not protect against a backend that
imports but cannot run. Ampere also keeps `docs/attention-backends.md` honest,
since its numerics were validated on an A100.

L40S 48GB is the budget substitute at half the price, though ~864 GB/s against
the A100's ~2 TB/s makes decode-bound TPOT incomparable to published numbers.
H100 costs 2-3x and at this model size only makes the engine more CPU-bound.
24GB consumer cards and ROCm are out.

Tensor parallel is exercised on a 2xA100 node for the smoke test only; sweeps
run single-GPU. The results table records driver, torch and **vLLM version** —
vLLM's scheduler changed substantially between V0 and V1, so a bare "vs vLLM"
number ages badly.

## Measurement plan

[benchmark-runbook.md](benchmark-runbook.md) turns this section into commands.

Axes to sweep: request rate, `max_num_batched_tokens`, scheduling policy,
chunked prefill on/off, and prompt/output length mix — one `sweep.py` suite
each. The "before" column is `enable_chunked_prefill=False`, which M3 kept alive
for exactly this.

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
effect, reported on its own line. That is `--suite chunked`.

Two confounds to control for:

- `warmup_model` sizes its warmup batch from `max_num_batched_tokens`, and on
  CUDA `kvcache_bytes()` derives the cache size by subtracting peak allocated
  memory — so changing the token budget silently changes the number of KV
  blocks. Pin it with `sweep.py --kvcache-tokens` on every run.
- vLLM queues without bound, so its rejection rate is zero by construction and
  overload lands in its p99. lean-vLLM does the same by default, but with
  admission control switched on it moves that pressure into its rejection rate
  instead. Compare on goodput-versus-p99, never p99 alone.

## Risks

1. Real numbers need a GPU; laptop runs prove correctness only.
2. `torch.compile` on `Sampler` with a per-step-varying batch size may recompile
   constantly. Check, and mark dynamic if so.
3. Tensor parallel is exercised by none of this, and rank 0 is the only rank
   that samples. Keep a TP smoke test in the loop or the path will rot.
