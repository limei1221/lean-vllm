# Benchmark runbook

How to produce the serving numbers on a A100 80GB SXM against vLLM.

Budget 4-5 hours of GPU time. Do steps 0-4 and check the numbers look sane
before committing to the long sweeps.

## Why SXM, not PCIe

Same GA100 die; what differs is the envelope it runs in.

| | A100 80GB SXM4 | A100 80GB PCIe |
|---|---|---|
| Memory bandwidth | ~2039 GB/s | ~1935 GB/s |
| TDP | 400W | 300W |
| Cooling | baseboard, active | passive card, chassis airflow |
| GPU-to-GPU | NVLink 3 / NVSwitch, ~600 GB/s | PCIe 4.0 x16, ~64 GB/s |

Sustained clocks are the reason. A sweep runs arms sequentially for an hour or
more, and a 300W passively cooled card throttles partway through — later arms
then run slower than earlier ones, which is indistinguishable from a scheduling
regression. Decode is also bandwidth-bound at 8B, the tensor-parallel smoke test
is an order of magnitude off over PCIe, and published vLLM numbers come from
HGX/DGX SXM nodes.

On PCIe anyway: lock clocks, watch `clocks_throttle_reasons.active`, record the
SKU, and treat only the within-sweep A/B as meaningful.

## 0. Record the box

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uname -r; python3 --version
```

This is the only free moment to get the driver version the results table
promises.

## 1. Setup

lean-vLLM:

```bash
git clone <remote> ~/lean-vllm && cd ~/lean-vllm
uv sync --extra cuda      # the dev group brings the server, the OpenAI SDK and the test deps
uv run python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.get_device_name(0))"
uv run hf download Qwen/Qwen3-8B --local-dir /workspace/huggingface/Qwen3-8B
```

The flash-attn wheel is pinned to `cu12torch2.9` on Python 3.12, which is what
`.python-version` already selects.

vLLM goes in a **separate** venv — it pins its own torch and will break the
flash-attn pin if it shares one:

```bash
uv venv ~/vllm-env --python 3.12
VIRTUAL_ENV=~/vllm-env uv pip install vllm
~/vllm-env/bin/vllm --version    # record it; vLLM's scheduler changed a lot between V0 and V1
```

## 2. Pin the clocks

```bash
sudo nvidia-smi -pm 1
nvidia-smi -q -d SUPPORTED_CLOCKS | head -20
sudo nvidia-smi -lgc 1410        # A100 boost
```

Leave a throttle log running for the whole session:

```bash
mkdir -p ~/lean-vllm/results
nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active \
  --format=csv -l 10 > ~/lean-vllm/results/nvidia-smi.log &
```

If `clocks_throttle_reasons.active` ever leaves `Not Active` during a sweep, that
sweep's arms are no longer comparable to each other. Without `sudo` you cannot
pin clocks, so watch the log twice as closely.

## 3. Sanity check

```bash
cd ~/lean-vllm
uv run pytest tests/ -q

uv run lean-vllm serve /workspace/huggingface/Qwen3-8B --port 8000 --max-model-len 4096 \
  --served-model-name qwen &
sleep 90
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","prompt":"The capital of France is","max_tokens":16,"temperature":0}' | jq .
curl -s localhost:8000/metrics.json | jq '.mean_step_seconds, .mean_batch_tokens, .graph_step_fraction'
kill %1
```

`graph_step_fraction` must be non-zero. If it is not, CUDA graphs are not
capturing and the `chunked` suite's graph-effect line means nothing.

## 4. Pin the KV cache size

Choose the cache size; do not let profiling choose it. `warmup_model` sizes its
warmup batch from `max_num_batched_tokens` and the cache is whatever is left
over, so an unpinned budget sweep moves the cache underneath itself.

`sweep.py` pins it in **tokens** and converts to each engine's block count, so
lean-vLLM and vLLM get the same capacity despite 256-token and 16-token blocks.

```bash
uv run python - <<'EOF'
from transformers import AutoConfig
c = AutoConfig.from_pretrained("/workspace/huggingface/Qwen3-8B")   # adjust
head_dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
per_token = 2 * c.num_hidden_layers * c.num_key_value_heads * head_dim * 2   # bf16
print(f"{per_token/2**10:.0f} KiB per token")
for gb in (6, 20, 45):
    print(f"  {gb} GB -> {int(gb * 2**30 // per_token):,} tokens")
EOF
```

Qwen3-8B (36 layers, 8 KV heads, head_dim 128) lands near 144 KiB per token, so
roughly 330k tokens for 45 GB. Take two values: **comfortable** (~45 GB) and
**cache-thrashing** (about an eighth of it), so preemption is exercised rather
than merely implemented.

```bash
export KVTOKENS=327680     # whatever the script printed
export MODEL=/workspace/huggingface/Qwen3-8B
```

## 5. lean-vLLM rate curve

```bash
uv run python benchmarks/sweep.py \
  --model $MODEL --engine lean-vllm --suite rate \
  --rates 1,2,4,8,12,16,24 --num-requests 1000 \
  --max-model-len 4096 --kvcache-tokens $KVTOKENS \
  --server-args "--max-num-batched-tokens 8192 --max-num-seqs 256" \
  --client-args "--dataset lognormal --input-len 512 --output-len 128 --timeout 1200" \
  --out results/lean-8b
```

- **1000 requests.** p99 over 300 completed requests is three samples.
- **`--timeout 1200`.** Past saturation a request takes minutes. A client timeout
  counts as a failure, and failures past 5% abort the run — so a short timeout
  turns the overload point into a dead sweep. Read overload off the exploding
  p99, not off an abort.
- **Batched tokens and seqs set explicitly.** lean-vLLM and vLLM ship different
  defaults; pin them or the comparison is between two configurations rather than
  two schedulers.

Goodput should climb, flatten, and then stall while `ttft_p99` runs away. That
knee is the result.

## 6. vLLM rate curve

Same trace, same seed, same token capacity. `sweep.py` converts
`--kvcache-tokens` into vLLM's 16-token blocks, and `bench_serving.py` reads the
model id off `/v1/models`, so nothing needs adjusting by hand.

```bash
PATH=~/vllm-env/bin:$PATH uv run python benchmarks/sweep.py \
  --model $MODEL --engine vllm --suite rate \
  --rates 1,2,4,8,12,16,24 --num-requests 1000 \
  --max-model-len 4096 --kvcache-tokens $KVTOKENS \
  --server-args "--served-model-name qwen --max-num-batched-tokens 8192 --max-num-seqs 256" \
  --client-args "--dataset lognormal --input-len 512 --output-len 128 --timeout 1200" \
  --out results/vllm-8b
```

The `PATH=` prefix puts the vLLM venv's binary in reach while the script itself
still runs under lean-vLLM's interpreter.

vLLM 0.11 dropped `--disable-log-requests`; per-request logging is off by
default now. On an older vLLM, add it back or the log drowns the run.

Run only the `rate` suite against vLLM. The others drive lean-vLLM's own flags.

## 7. The lean-vLLM suites

Separate invocations, after both curves exist.

```bash
# four-run A/B: scheduling isolated from the CUDA-graph effect
uv run python benchmarks/sweep.py --model $MODEL --suite chunked \
  --rates 4,8,16 --num-requests 1000 --kvcache-tokens $KVTOKENS \
  --client-args "--dataset lognormal --timeout 1200" --out results/lean-8b

# token budget
uv run python benchmarks/sweep.py --model $MODEL --suite budget \
  --budgets 512,2048,8192 --rates 8 --num-requests 1000 --kvcache-tokens $KVTOKENS \
  --client-args "--dataset lognormal --timeout 1200" --out results/lean-8b

# does capping one prompt's share of a step protect short requests?
uv run python benchmarks/sweep.py --model $MODEL --suite starvation \
  --rates 8 --num-requests 1000 --kvcache-tokens $KVTOKENS \
  --client-args "--long-fraction 0.2 --long-input-len 3072 --timeout 1200" \
  --out results/lean-8b

# fcfs vs priority, long prompts arriving at priority 1
uv run python benchmarks/sweep.py --model $MODEL --suite policy \
  --rates 8 --num-requests 1000 --kvcache-tokens $KVTOKENS \
  --client-args "--long-fraction 0.2 --long-input-len 3072 --timeout 1200" \
  --out results/lean-8b
```

The last two are answered by the per-label breakdown, not the table:

```bash
jq '.summary.by_label | to_entries[] | {label: .key, ttft_p99: .value.ttft_seconds.p99, e2e_p99: .value.e2e_seconds.p99}' \
   results/lean-8b/starvation/*.json
```

## 8. Cache pressure

Repeat the rate curve with the small cache:

```bash
uv run python benchmarks/sweep.py --model $MODEL --suite rate \
  --rates 4,8,16 --num-requests 1000 --kvcache-tokens $((KVTOKENS / 8)) \
  --client-args "--dataset lognormal --timeout 1200" --out results/lean-8b-tight
```

The `preempt` column should stop being zero. If it does not, shrink further.

## 9. Collect

```bash
tar czf results-$(date +%F).tar.gz results/
```

Every `summary.json` carries the full argument set, so runs reproduce from the
archive alone. Add the version block from step 0 and `vllm --version` by hand —
nothing captures those automatically.

## Optional: tensor-parallel smoke test

Needs a 2xA100 node. Not a sweep — it exists so the TP path does not rot, since
rank 0 is the only rank that samples.

```bash
uv run lean-vllm serve $MODEL --tensor-parallel-size 2 --port 8000 &
sleep 120
uv run python benchmarks/bench_serving.py --num-requests 50 --request-rate 4 \
  --dataset fixed --input-len 512 --output-len 128
```
Outputs:
```
50/50 done, 0 failed
--- fixed @ 4.0/s ---
50 completed, 0 rejected, 0 failed in 13.8s
goodput 3.63 req/s, offered 3.63 req/s, output 465 tok/s, rejected 0.0%
ttft         mean    223.0 p50     92.3 p99   1572.3   (ms)
tpot         mean     12.4 p50     12.0 p99     20.3   (ms)
itl          mean     12.5 p50      9.7 p99     82.0   (ms)
e2e          mean   1803.4 p50   1630.8 p99   3381.4   (ms)
```

## Known limits

- `init_process_group` binds a hardcoded `localhost:2333`, so only one engine
  exists per machine. Never run lean-vLLM and vLLM at once; if a sweep crashes,
  confirm the old process is gone before starting the next.
- `/metrics.json` counters cover the server's whole lifetime. `sweep.py` starts a
  fresh server per run, so each run's snapshot describes that run — plus its
  three warmup requests.
