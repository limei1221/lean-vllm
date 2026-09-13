# Benchmark runbook: lean-vLLM versus vLLM

Compare Qwen3-8B on one NVIDIA H100-SXM5-80GB using the same workload and resource
limits, with **chunked prefill enabled** and **full + piecewise CUDA graphs** on
both engines. lean-vLLM runs **two curves, async scheduling off and on**. vLLM
runs **one curve at its default, which turns async scheduling on**.

The five offered loads—**1, 24, 32, 48, and 64 requests/s**—cover the plateau,
where the gap opens. Load 1 stays because its TPOT isolates per-step overhead.
Each point uses 1,000 requests, for 15 runs total. Run the engines sequentially
on the same machine and compare these fresh results.

Use the same Bash session for the commands below.

## 1. Install both engines

Clone once; for an existing checkout, start with `cd`:

```bash
git clone git@github.com:limei1221/lean-vllm.git ~/workspace/lean-vllm
cd ~/workspace/lean-vllm
git checkout feature/online-serving
uv sync --extra cuda
uv run hf download Qwen/Qwen3-8B --local-dir ~/workspace/huggingface/Qwen3-8B
uv run hf download Qwen/Qwen3-0.6B --local-dir ~/workspace/huggingface/Qwen3-0.6B
uv run python -c 'from lean_vllm.attention import get_attention_backend; print(get_attention_backend().get_name())'
```

The sync pulls a prebuilt FlashAttention-3, about 400 MB, pinned by URL and
hash in `pyproject.toml`. Nothing is compiled, so the box needs no CUDA
toolkit.

The check must print `flash_attn_3`. Anything else means the sweep would measure
the Torch backend, which is about fifty times slower. The 0.6B model is only for
the offline trace in section 5.

Keep vLLM in a separate environment because it manages its own PyTorch
dependencies. Version `0.26.0` is the last one that pins torch `2.11.0`, what
the `cuda` extra pins, so the curves compare engines rather than PyTorch
releases. That is two releases behind the `0.28.0` of the previous comparison,
so read the two together only for the shape of the curve:

```bash
uv venv ~/workspace/vllm-env --python 3.12
uv pip install --python ~/workspace/vllm-env/bin/python vllm==0.26.0
```

## 2. Set the shared workload and server limits

```bash
export MODEL=~/workspace/huggingface/Qwen3-8B
export RUN_RESULTS="results/$(date -u +%Y%m%dT%H%M%SZ)"
export RATES="1,24,32,48,64"
export KVTOKENS=327680
export PINNED="--max-num-batched-tokens 8192 --max-num-seqs 256 --enable-chunked-prefill"
export TRACE_ARGS="--dataset lognormal --input-len 512 --output-len 128 --sigma 0.8 --temperature 0 --warmup 3 --timeout 1200"
export LEAN_SERVER_ARGS="$PINNED --cudagraph-mode full_and_piecewise"
export VLLM_SERVER_ARGS="$PINNED --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'"
mkdir -p "$RUN_RESULTS"
```

| Setting | Both engines |
| --- | --- |
| Model | Same local Qwen3-8B weights |
| Input / output lengths | Lognormal distribution, medians 512 / 128 tokens, σ = 0.8 |
| Sampling | Greedy, seed 0 |
| Requests per point | 1,000 measured requests, plus 3 warmup requests |
| Maximum context | 4,096 tokens |
| KV cache capacity | 327,680 tokens |
| Batch token budget / maximum sequences | 8,192 / 256 |
| Chunked prefill | Enabled |
| CUDA graphs | Full + piecewise |
| Async scheduling | lean-vLLM off and on; vLLM at its default, on |

The graph-mode option differs between engines: lean-vLLM uses
`--cudagraph-mode full_and_piecewise`; vLLM uses the equivalent
`FULL_AND_PIECEWISE` setting in `--compilation-config`.

`sweep.py` converts KV capacity to each engine's block count, so both get the
same number of cache tokens. Each point starts a fresh server. The 1,200-second
client timeout allows queued requests to finish under heavy load.

## 3. Save metadata and monitor the GPU

Record the build and environment with the results:

```bash
{
  git rev-parse HEAD
  git status --short
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  uname -r
  uv run python --version
  uv run python -c 'import torch; print("lean-vLLM torch:", torch.__version__)'
  ~/workspace/vllm-env/bin/vllm --version
  ~/workspace/vllm-env/bin/python -c 'import torch; print("vLLM torch:", torch.__version__)'
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  date '+%Y-%m-%dT%H:%M:%S%z'
} | tee "$RUN_RESULTS/environment.txt"
git diff HEAD > "$RUN_RESULTS/working-tree.patch"
```

Save any untracked source files separately; they are not included in the patch.

If the host allows it, enable persistence mode and lock the clock. Both need
root, which the shell in a rented GPU container usually already has:

```bash
nvidia-smi -pm 1
nvidia-smi -lgc "$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)"
```

If the container refuses, continue with monitoring. Record settings and keep a
GPU log running through every run:

```bash
nvidia-smi --query-gpu=name,persistence_mode,clocks.applications.graphics,clocks.max.graphics,power.limit,power.max_limit \
  --format=csv | tee "$RUN_RESULTS/gpu-settings.csv"

nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active \
  --format=csv -l 1 > "$RUN_RESULTS/nvidia-smi.log" &
GPU_LOG_PID=$!
```

## 4. Run the curves

Make sure no other model server is running. The sweep handles startup, health
checks, warmup, and shutdown automatically.

### lean-vLLM

```bash
uv run python benchmarks/sweep.py \
  --model "$MODEL" --engine lean-vllm --suite async \
  --rates "$RATES" --num-requests 1000 --seed 0 \
  --max-model-len 4096 --kvcache-tokens "$KVTOKENS" \
  --server-args "$LEAN_SERVER_ARGS" \
  --client-args "$TRACE_ARGS" --out "$RUN_RESULTS/lean-8b"
```

### vLLM

```bash
PATH="$HOME/workspace/vllm-env/bin:$PATH" uv run python benchmarks/sweep.py \
  --model "$MODEL" --engine vllm --suite rate \
  --rates "$RATES" --num-requests 1000 --seed 0 \
  --max-model-len 4096 --kvcache-tokens "$KVTOKENS" \
  --server-args "$VLLM_SERVER_ARGS" \
  --client-args "$TRACE_ARGS" --out "$RUN_RESULTS/vllm-8b"
```

Confirm vLLM ran with async scheduling on, and record it:

```bash
grep -h "Asynchronous scheduling is" "$RUN_RESULTS"/vllm-8b/rate/*.server.log | sort | uniq -c \
  | tee "$RUN_RESULTS/vllm-async-scheduling.txt"
```

If startup fails, read the corresponding `.server.log` and confirm the process
has stopped before retrying. The default startup timeout is 900 seconds.
Use a new output directory for a rerun to preserve the previous results.

## 5. Profile the step loop

Run one load-48 point per arm with the step-loop profiler on. The server
inherits these variables from the sweep, and each arm's server log names its
trace file. This trace is CPU-only; `await_tokens` shows the host waiting on
the device.

```bash
LEAN_PROFILE_DIR="$RUN_RESULTS/profile-48" LEAN_PROFILE_STEPS=600 \
  uv run python benchmarks/sweep.py \
  --model "$MODEL" --engine lean-vllm --suite async \
  --rates 48 --num-requests 1000 --seed 0 \
  --max-model-len 4096 --kvcache-tokens "$KVTOKENS" \
  --server-args "$LEAN_SERVER_ARGS" \
  --client-args "$TRACE_ARGS" --out "$RUN_RESULTS/profile-48"
```

The offline pair adds device activity, which gives the GPU idle fraction to
compare against the 34.4% of 12 September:

```bash
LEAN_PROFILE_DIR="$RUN_RESULTS/offline-off" LEAN_PROFILE_CUDA=1 \
  uv run python benchmarks/bench_offline.py
LEAN_PROFILE_DIR="$RUN_RESULTS/offline-on" LEAN_PROFILE_CUDA=1 LEAN_ASYNC_SCHEDULING=1 \
  uv run python benchmarks/bench_offline.py
```

## 6. Compare and archive

Print the same metrics for both engines:

```bash
jq -r '.engine as $engine | .rows[] | [$engine, .arm, .request_rate, .completed, .rejection_rate, .failure_rate, .goodput, .output_tok_s, .ttft_p99, .tpot_p50, .e2e_p99] | @tsv' \
  "$RUN_RESULTS/lean-8b/async/summary.json" \
  "$RUN_RESULTS/vllm-8b/rate/summary.json"
```

Columns are engine, arm, offered load, completed requests, rejection rate, failure
rate, goodput, output tokens/s, p99 TTFT, median TPOT, and p99 E2E. Latency
values are in seconds.

| Metric | What to compare |
| --- | --- |
| Goodput | Completed requests per second, including queue drain time; higher is better |
| Output tokens/s | Generation throughput; higher is better |
| p99 TTFT | Time until the first token for slow requests; lower is better |
| Median TPOT | Time per output token after the first; use load 1 for low-load latency |
| p99 E2E | Time to finish slow requests; lower is better |

Before drawing conclusions:

- Check that each run completed all 1,000 requests without failures or rejections.
  Goodput has no latency cutoff, so read it alongside latency.
- Compare matching offered loads and workload settings. The like-for-like pair
  is lean-vLLM's `async-scheduling=True` arm against vLLM. Find where throughput
  levels off and tail latency rises sharply—the saturation knee.
- Check GPU-busy clocks and throttling for each pair. Aim for mean clocks within
  about 1%; report differences that could affect the comparison.

Per-run JSON contains client summaries and server snapshots. Server counters
include warmup; use `server.after - server.before` for cumulative counters when
needed. Each run records `started_at` and `finished_at` in UTC around the client
run, so align the GPU log to those rather than to file modification times.

Stop the logger and archive the session:

```bash
kill "$GPU_LOG_PID"
wait "$GPU_LOG_PID" || true
tar czf "${RUN_RESULTS}.tar.gz" "$RUN_RESULTS"
```

Write the report around the matched throughput and latency curves, with the
commit, engine versions, hardware, and clock conditions alongside them. Report
the async scheduling mode of every curve, and the offline GPU idle fraction for
both arms.
