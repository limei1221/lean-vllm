# Online serving

`lean-vllm serve` puts the engine behind an OpenAI-compatible HTTP API.
Requests arrive at any time, tokens stream back as they are produced, and one
token budget per step decides what runs — prefill chunks and decode rows
together, with preemption and an optional fairness policy.

## Start a server

```bash
uv sync --extra serve
uv run lean-vllm serve ~/workspace/huggingface/Qwen3-8B --port 8000 --served-model-name qwen
```

`--served-model-name` is the id the server answers to. Without it, the model
path is the id. Every `Config` field is also a flag; `lean-vllm serve --help`
lists them.

## Endpoints

| | |
| --- | --- |
| `POST /v1/completions` | prompt as a string or a list of token ids |
| `POST /v1/chat/completions` | messages, through the model's chat template |
| `GET /v1/models` | the id this server serves |
| `GET /health` | 503 once the engine thread dies |
| `GET /metrics` | Prometheus text |
| `GET /metrics.json` | the same numbers as a JSON summary |

Both completion endpoints stream with `"stream": true`, and
`"stream_options": {"include_usage": true}` adds a final usage chunk.

## Use it

Any OpenAI client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
completion = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "introduce yourself"}],
    max_tokens=256,
    temperature=0.6,
)
print(completion.choices[0].message.content)
```

No key is checked, so `api_key` can be anything. `example_serving.py` streams
instead: two prompts concurrently over the async client, reporting each one's
time-to-first-token.

## Request fields

Supported: `model`, `prompt` / `messages`, `max_tokens` (default 64),
`temperature` (default 1.0, where 0 is greedy), `stream`, `stream_options` and
`stop`. `n`, the number of completions to return for one prompt, is accepted
only as 1: the engine samples one sequence per request.

Two extras ride in the body, since the OpenAI schema has no field for either:

| | |
| --- | --- |
| `ignore_eos` | generate the full `max_tokens`, for benchmarks |
| `priority` | lower runs earlier under `--scheduling-policy priority` |

Everything else is **refused with a 400, not ignored** — `top_p`, `top_k`,
`min_p`, `seed`, `logprobs`, penalties, `logit_bias`, `tools`, `echo`,
`suffix`, `best_of`, `n > 1`. Silently ignoring `top_p` returns wrong output
with no signal, which is worse than refusing it.

Refusal is by value, not by presence: `"top_p": 1.0` asks for nothing, so a
client filling in OpenAI's defaults is not punished for a field it never set.

## Errors

| status | when |
| ---: | --- |
| 400 | an unsupported field, or prompt + `max_tokens` over the context |
| 404 | a `model` this server does not serve |
| 429 | the waiting queue is full (`--max-waiting-requests`) |
| 503 | the engine died, or the request can never fit in the cache |
| 504 | the request waited past `--request-timeout` without ever running |

Once a stream has sent its 200 the status code cannot be retracted, so errors
after that point arrive as an SSE error event before the close.

## Scheduling

One pass per step: running sequences first (a token for each decode, the next
chunk for each unfinished prefill), then admissions from the waiting queue into
whatever budget is left. A sequence that cannot grow even alone in the cache is
dropped rather than preempted forever.

| flag | default | |
| --- | ---: | --- |
| `--max-num-batched-tokens` | 8192 | tokens one step may schedule |
| `--max-num-seqs` | 1024 | sequences that may run at once |
| `--enable-chunked-prefill` | on | off is whole prompts only, never mixed with decode |
| `--scheduling-policy` | `fcfs` | or `priority`, which reads the request's `priority` |
| `--enable-prefix-caching` | on | off recomputes every prompt |
| `--prefix-caching-hash-algo` | `sha256` | or `xxhash` |
| `--long-prefill-token-threshold` | 0 | cap on one prompt's share of a step; 0 is none |
| `--num-kvcache-blocks` | profiled | pin it to hold cache capacity still across runs |
| `--kvcache-block-size` | 16 | tokens per block |
| `--enforce-eager` | off | on disables CUDA graphs |
| `--async-scheduling` | on | overlaps scheduling and batch prep with the forward pass; off under tensor parallelism; see [Pipelined steps](#pipelined-steps) |

Preemption recomputes rather than swaps, and a preempted sequence goes back to
the head of the queue.

`--long-prefill-token-threshold` follows vLLM's V1 meaning: a per-step token cap
on one prompt, applied to every prefill.

## Pipelined steps

`step()` launches one step and drains the one launched before it, so the
drain's host work runs while the GPU works on the step just queued.
`--async-scheduling` picks which phases land in that overlap:

```text
off                              on
---                              --
await tokens of step k-1         schedule step k
reconcile step k-1               launch step k
schedule step k                  await tokens of step k-1
launch step k                    reconcile step k-1
detokenize step k-1              detokenize step k-1
```

Off, only detokenization overlaps. The drain still runs before scheduling, so a
stop condition is always seen before the next step is built. On, scheduling and
batch preparation for step k run ahead of the await too.

**A stop is seen one step late.** With the flag on, a request that has just hit
EOS or a stop sequence already has one more step launched. That token is
discarded on reconcile and never emitted, but its compute is spent. A request
ending at `max_tokens` does not pay it: the scheduler stops scheduling it once
its reserved tokens reach the limit.

**Awaiting a step does not wait for the next one.** `SampledTokens`
(`lean_vllm/engine/sampled_tokens.py`) copies the sampled tensor to pinned host
memory on its own CUDA stream and records an event at launch. The await waits
on that event, not on the default stream, which may already hold the next
step's kernels. A plain `.tolist()` on the device tensor would serialize the
pipeline.

**All other device work stays on one stream.** A later step's kernels run
strictly after this step's writes, which is what makes it safe to free a KV
block while a step is in flight, or to publish a prefix-cache block a running
kernel is still writing. A pending token is a count, never a placeholder value,
and it is discarded if its sequence was preempted or aborted since launch.

The flag is on by default, as in vLLM, and turns itself off with a warning
under tensor parallelism: ranks above zero never see the sampled tokens, so
they could not follow. On an H100
([benchmark-2026-09-13.md](../benchmarks/benchmark-2026-09-13.md)) turning it
on cuts offline GPU idle from 22.4% to 3.2% and adds 7–9% serving goodput on
Qwen3-8B at loads 24, 48 and 64.

## Prefix caching

A prompt is cut into blocks, each block hashed together with the chain of
blocks before it. A request whose leading blocks are already in the cache
takes them by reference and prefills only the rest, so the token budget goes
to work that has not been done.

Three properties make that safe and cheap:

**Blocks are hashed as they arrive.** A full block never takes another token,
so its hash is final and is computed once, on the request, as the tokens come
in. Nothing is hashed inside a step. This matters because the cache is queried
again on every step a request spends at the head of the waiting queue.

**The tail is always recomputed.** Attention needs at least one query token, so
the last block is never a cache candidate even when the whole prompt is
resident.

**Eviction spares the prefix.** Freed blocks re-enter the queue deepest first,
so a shared head outlives the suffixes built on it. A block keeps its contents
until something claims it.

The queue order itself is untouched: prefix caching changes what a request
costs, never when it runs.

`--no-enable-prefix-caching` turns all of it off: no hashing, no lookups, every
prompt computed in full. It is the A/B arm, not an optimisation.

`--prefix-caching-hash-algo` picks how a block is keyed, and follows vLLM:
`sha256` by default, `xxhash` for speed where every request is trusted. A
collision serves one tenant another tenant's tokens, which is why the secure
one is the default and the fast one is opt-in. Both pickle the block before
hashing, so both survive a restart and reach the tensor-parallel workers
unchanged, and the tokens behind a hit are compared before a block changes
hands either way.

vLLM also offers `sha256_cbor` and `xxhash_cbor`, which serialize with
canonical CBOR instead of pickle so a hash reproduces across languages and
Python versions. Nothing here reads the cache from another process, so they are
not implemented.

Hit rate is blocks supplied over blocks looked up, at `/metrics.json`. It reads
0 with caching off, which is the point of having the arm.

## Admission control

Both off by default, which matches vLLM and keeps a comparison honest:

- `--max-waiting-requests` refuses on arrival with a 429, decided before any
  status code is sent rather than at the first token.
- `--request-timeout` drops a request that has waited that long without ever
  being scheduled, with a 504. It never touches a sequence that has already
  run — `max_tokens` bounds those.

Left off, both engines queue without bound, and overload lands in p99 latency
instead of a rejection rate. Compare goodput against p99, never p99 alone.

## Metrics

`/metrics.json` carries what the engine saw: step count, mean step time, mean
batch tokens, the prefill/decode token split, graph coverage, preemptions,
prefix-cache hit rate, and peak KV usage, plus counts and means for TTFT, TPOT,
queue delay and end-to-end latency. Names mirror vLLM's under `lean_vllm:`, so
one dashboard reads both engines.

Percentiles are the client's job: the histogram buckets here are too coarse to
interpolate one from without lying about it.

`model_busy_fraction` is the honest utilization number — the share of wall
clock spent inside a forward pass. `gpu_utilization_percent_nvidia_smi` sits
beside it and counts any live kernel as busy, so it reads high even when the
batch is one row wide. It is there to be compared, not believed.

## Benchmarks

```bash
uv run python benchmarks/bench_serving.py --dataset lognormal --request-rate 8
uv run python benchmarks/sweep.py --model ~/workspace/huggingface/Qwen3-8B \
    --suite rate --rates 1,2,4,8,16 --kvcache-tokens 327680 --out results/8b
```

`--suite prefix` runs caching off against each hash algorithm, over
`--dataset prefix`, where a few shared system prompts sit in front of unique
questions. Every other dataset draws token ids at random, so no two prompts
share a block and the cache has nothing to find.

`bench_serving.py` sends Poisson arrivals through the OpenAI SDK and points at
vLLM unchanged. `sweep.py` runs it across arms, restarting the server per run.
Arrivals are open-loop and a 429 is never retried, so the offered load is the
rate it claims to be.

[benchmark-runbook.md](benchmark-runbook.md) is the step-by-step for a fresh
GPU box. [benchmarks/benchmark-2026-09-13.md](../benchmarks/benchmark-2026-09-13.md)
has the latest numbers.

## How it fits together

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
tokenization, detokenization, SSE — lives above the thread boundary, and
requests and aborts cross it through a queue drained at the top of `step()`,
never by mutating the scheduler from another thread. It runs in a thread rather
than the event loop because `step()` blocks in C for a whole forward pass and
would otherwise starve the HTTP handlers.

There is no in-process restart. The CUDA context, KV cache and tensor-parallel
children do not survive the engine thread, so an unhandled exception fails every
outstanding request, flips `/health` to 503, and exits for a supervisor to
bring the process back.
