# Disaggregated prefill / decode

Prefill and decode run on separate engines. A prefill instance computes a
prompt's KV cache and samples one token. A decode instance pulls that KV and
generates the rest. A proxy in front splits each request between them. Long
prompts then stop stalling other requests' decode steps, and each side can be
sized and batched for its own phase.

The design follows vLLM V1: a KV connector with a scheduler half and a worker
half (`KVConnectorBase_V1`), the request flow and `kv_transfer_params` of its
`NixlConnector`, and its `toy_proxy_server.py`. lean-vLLM's `NixlConnector`
reads blocks GPU to GPU over RDMA with NIXL, as vLLM's does, and lets the two
sides run different tensor-parallel sizes.

## What is supported

| | Support |
| --- | --- |
| Connector | `NixlConnector`: the decode instance reads GPU to GPU over RDMA |
| Roles | `kv_producer` (prefill), `kv_consumer` (decode), `kv_both` (either, per request) |
| Models | Every model the engine loads: Qwen3's key/value pages, DeepSeek-V2's MLA latents |
| Tensor parallelism | Any sizes on the two sides; each decode rank reads its heads from the prefill ranks that hold them |
| Prefix caching | On both sides. The decode instance reads only the blocks it does not already hold |
| Front door | `lean-vllm proxy`, round-robin over any number of prefill and decode servers |
| Failure | A load that fails, for any reason, falls back to prefilling on the decode instance |
| Devices | Verified on CPU against a fake NIXL agent only. No real agent and no GPU yet |
| Metrics | Per-load time and bytes, failures and expiries on `/metrics` |

Not supported: layer-by-layer streaming during the prefill (vLLM's
`save_kv_layer`), and a streamed prefill request.

## Running it

Install the `nixl` extra (Linux only). Then two servers and a proxy. Here they
share one host; the prefill server runs on two GPUs and the decode server on
one:

```bash
uv sync --extra cuda --extra nixl
MODEL=~/workspace/huggingface/Qwen3-8B
CUDA_VISIBLE_DEVICES=0,1 uv run lean-vllm serve $MODEL --port 8100 --served-model-name qwen \
    --tensor-parallel-size 2 --kv-transfer-config '{"kv_role": "kv_producer", "kv_port": 14579}'
CUDA_VISIBLE_DEVICES=2 uv run lean-vllm serve $MODEL --port 8200 --served-model-name qwen \
    --kv-transfer-config '{"kv_role": "kv_consumer"}'
uv run lean-vllm proxy --prefill http://127.0.0.1:8100 --decode http://127.0.0.1:8200 --port 8000
```

Clients talk to the proxy on port 8000, exactly as they would to one server.
`--prefill` and `--decode` each take several URLs. Across hosts, set `kv_ip` on
each prefill server to an address its decode servers can reach. `kv_ip` and
`kv_port` serve only the handshake. UCX picks the RDMA device for the reads
itself; its `UCX_*` variables steer that.

### `--kv-transfer-config`

JSON, as vLLM's flag. Every key is optional.

| Key | Default | What it sets |
| --- | --- | --- |
| `kv_connector` | `NixlConnector` | The only one |
| `kv_role` | `kv_both` | `kv_producer` serves KV, `kv_consumer` loads it, `kv_both` does either |
| `kv_ip` | `127.0.0.1` | Where a producer answers handshakes, and the address it tells consumers to dial |
| `kv_port` | 14579 | Rank *r* of a producer answers on `kv_port + r` |
| `engine_id` | random | Names this instance; a consumer refuses blocks from a producer that has restarted since |

`LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT` (default 480 seconds, as vLLM's
`VLLM_NIXL_ABORT_REQUEST_TIMEOUT`) is how long a producer holds blocks that no
consumer has read before freeing them.

### Without the proxy

The proxy only sets `kv_transfer_params`, a request field both endpoints accept:

1. Send the request to a prefill server with `"max_tokens": 1`, not streamed,
   and `"kv_transfer_params": {"do_remote_decode": true}`.
2. Its reply carries `kv_transfer_params`: the blocks, and where to pull them.
3. Send the original request to a decode server with those
   `kv_transfer_params`.

A server started without `--kv-transfer-config`, or in the wrong role, answers
a request carrying `kv_transfer_params` with a 400.

## What has been verified

- `tests/test_nixl_connector.py`: prefill and decode ranks in one process on
  CPU caches, over a fake NIXL agent that reads by address with `memmove`. So
  every descriptor the workers build is checked against real memory, and a read
  outside a registered tensor fails. For keys and values and for MLA latents,
  at 11 pairs of tensor-parallel sizes, including 4 and 6, where neither
  divides the other, every decode rank ends up with exactly its heads of the
  right blocks. Each prefill rank frees its blocks once every decode rank
  reading it has notified it. Also covered: prefix-cache hits, a read finishing
  before the hand-over, and each failure below.
- `tests/test_disagg_e2e.py` runs Qwen3-0.6B on CPU with HND caches, the layout
  every disaggregated engine uses, and its greedy tokens match NHD's.
- Before NIXL, a TCP connector carried the same scheduler half, servers and
  proxy. With it, on CPU: two engines in two processes, and the HTTP servers
  behind the proxy, streamed and not, matched a single engine run with its
  prompt split the same way, and the prefill instance's KV usage returned to
  zero once read.
- Not run yet: `tests/test_disagg_e2e.py`'s two engines over a real NIXL agent,
  and `tests/test_kv_transfer.py`'s read between two GPUs. Both skip without
  NIXL. No GPU run and no benchmark yet.

## How it works

### A request's path

```
client ──▶ proxy ──(max_tokens=1, do_remote_decode)──▶ prefill server
                                                         prefills, samples 1 token, finishes "length" or "stop";
                                                         holds the prompt's blocks; replies with
                                                         kv_transfer_params {remote_block_ids, host, port, tp_size, ...}
             proxy ──(original request + kv_transfer_params)──▶ decode server
                                                         allocates blocks, WAITING_FOR_REMOTE_KVS,
                                                         reads its heads ◀──RDMA── the prefill ranks holding them;
                                                         each read notifies ──▶ prefill frees its blocks
                                                         computes the last prompt token, samples, decodes
client ◀── proxy ◀── tokens, streamed or not
```

### Scheduler half

It lives in `Scheduler`, with vLLM's method names:

- `get_num_new_matched_tokens` is asked at admission, after the local prefix
  cache. With `do_remote_prefill`, every prompt token past the local hits comes
  from the producer. If the producer's block count does not fit this prompt,
  the request prefills here instead.
- `update_state_after_alloc` queues the load. Local block ids pair with the
  producer's, skipping blocks already cached here. It also clears
  `do_remote_prefill`, so a request preempted later recomputes rather than
  reading blocks the producer has since freed.
- `build_connector_meta` collects each step's loads to start and blocks to
  expose, and goes to every rank with the step.
- `request_finished` decides, on the producer, whether to hold a finished
  request's blocks. A request that ran to its `max_tokens` or hit a stop token
  is handed over, as in vLLM; an abort ends it there. Held blocks stay in the
  prefix cache once freed.

A loading request sits in `Scheduler.recving` in `WAITING_FOR_REMOTE_KVS`. It
counts toward `num_requests_waiting`, as in vLLM, and takes no step budget.
Once loaded, it joins `running` with all but its last prompt token computed.
That token is recomputed against the pulled KV, so the decode instance samples
the first token itself, as vLLM's does. The cost is that its output matches a
local run split the same way, not one that prefills the prompt whole. In bf16
the two can pick different tokens at a near-tie.

The engine calls `kv_connector_step` every step, even one that schedules
nothing. Loads then start, and finished transfers come back, while the engine
is otherwise idle. `is_finished` stays false while a load or a held send is
open, so the server keeps polling.

### Worker half

One per rank, in `ModelRunner`. It registers the rank's KV cache with a NIXL
agent over UCX. Each step's `KVConnectorOutput` is gathered from all ranks. As
in vLLM's `KVOutputAggregator`, a transfer is over once every rank reports it,
and a load failed if any rank's did.

- **Handshake**: a producer rank answers on `kv_ip:kv_port + rank` with its
  agent's metadata and its cache's addresses. A consumer rank asks each
  producer rank it reads once, and checks the same engine, tensor-parallel size,
  dtype, block size and total head count.
- **Which heads**: caches are HND, so within a block each head's tokens are
  contiguous. A consumer rank's heads overlap those of one or more producer
  ranks; from each it reads the overlap of every block, straight into the right
  heads of its own block. One read per producer rank, one descriptor per layer,
  key or value, and block. MLA latents are the same on every rank, so a consumer
  rank reads whole blocks from the first producer rank that overlaps it.
- **Release**: each read carries a notification `request:readers`, where
  readers is how many consumer ranks overlap that producer rank. A producer
  rank frees its blocks once it has that many. An MLA consumer rank only
  notifies the other producer ranks it overlaps. A read is one-sided, so it can
  finish before the producer's engine hands the request over; the count waits
  for it.
- **Threads**: every NIXL call runs on one thread per rank, which answers
  handshakes, posts reads, polls them and takes notifications.

This differs from vLLM's `NixlConnector` in a few places. It needs no
divisibility between the two sizes, only that each divides the head count. It
builds descriptor lists per request rather than preparing them at handshake.
It has no heartbeats, push mode or host-buffer staging.

### Ordering on a GPU

On the producer, a request's KV is complete before it is handed over, because
its token was synchronised back to the host. On the consumer, freshly allocated
blocks may still be written by a step in flight under async scheduling: their
previous owner's next token. So each load records an event when it is queued,
and its reads are posted only once that event has passed. NIXL reports a read
done once its data has landed.

### Metrics

On `/metrics`, beside the engine's own, named after vLLM's NIXL metrics:

| Metric | What it counts |
| --- | --- |
| `lean_vllm:kv_transfers_total` | Loads finished |
| `lean_vllm:kv_transfer_time_seconds` | Per load, from posting its reads to the last landing, on the slowest rank |
| `lean_vllm:kv_transfer_bytes` | Per load, across all ranks |
| `lean_vllm:kv_transfer_failures_total` | Loads that failed, so the prompt prefilled on the decode instance |
| `lean_vllm:kv_transfer_expired_total` | On the prefill instance, held blocks freed with no consumer reading them |

The first four are on the decode instance; throughput is
`kv_transfer_bytes_sum / kv_transfer_time_seconds_sum`. A load's time leaves out
the first handshake with each producer rank, as vLLM's does.

### Aborts and failures

| Event | What happens |
| --- | --- |
| Client aborts during the load | The request finishes at once; its blocks are freed when the load ends, since its reads are writing them |
| Producer unreachable, restarted, or laid out differently | The load fails; the request prefills on the decode instance from its local prefix hits |
| NIXL refuses a read when it is posted | The load fails, and the consumer notifies that producer rank, which frees its blocks now |
| A read fails in flight | The load fails once every read of it has ended. Its notification may yet arrive, so the producer frees its blocks at the timeout. The timeout cannot tell a read in flight, which is why it is long |
| No consumer reads the blocks | The producer frees them after `LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT` |
| Prefill stops on a stop token | Handed over as usual, as in vLLM; the decode instance samples its first token itself and may stop there too |
| Prefill is aborted | No `kv_transfer_params`; the proxy sends the request on as is and the decode instance prefills it |

## Next steps

This connector aims to show what P/D is, not to match vLLM's feature for
feature. P/D trades the time to move a prompt's KV for decode steps that
prefills no longer stall. Seeing that trade needs a run on RDMA hardware, at
equal and at different tensor-parallel sizes, benchmarked against a single
instance: TTFT and TPOT under prefill-heavy load, set against
`kv_transfer_time_seconds`.

Left out as production concerns, all in vLLM's NIXL connector: leases renewed
by heartbeats, a load-failure policy, push mode, host-buffer staging, and
layer-wise hooks.
