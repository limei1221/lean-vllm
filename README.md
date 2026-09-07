# InferWeave

A lightweight vLLM implementation used as a testbed for production inference
engineering: scheduling, attention kernels, model architectures, speculative
decoding, and distributed serving.

The goal is not feature parity with vLLM, but comparable numbers on the parts
that are implemented. Each project ships a design document, a correctness check
against a reference implementation, and a benchmark against vLLM on the same
hardware.

## Roadmap

| | Project | Status |
|---|---|---|
| 0 | Attention backend abstraction | interface + Torch/FlashAttention backends done |
| 1 | Online serving + advanced scheduler | scheduler, async engine, OpenAI server and metrics done |
| 2 | DeepSeek-style model support: MLA + MoE + YaRN | |
| 3 | Speculative decoding | |
| 4 | Disaggregated prefill / decode | |

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # deps, dev tools and the package, into .venv
uv sync --extra cuda     # add FlashAttention and Triton (NVIDIA only)
uv sync --extra serve    # add FastAPI and uvicorn for the HTTP server
```

FlashAttention and Triton are optional. Without them the engine runs on CPU and
Apple Silicon via the `torch` attention backend, at laptop speed — enough to
develop and test the scheduler and cache against a small model.

The device is picked automatically (cuda, then mps, then cpu) and can be forced
with `INFERWEAVE_DEVICE`. Off CUDA there is no `mem_get_info` to size the KV
cache from, so it comes from `kvcache_memory_gb` (default 2.0) instead of
`gpu_memory_utilization`.

## Quick start

```bash
uv run hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
uv run python example.py
```

```python
from inferweave import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, InferWeave."], sampling_params)
outputs[0]["text"]
```

The attention backend is picked automatically and can be forced with
`INFERWEAVE_ATTENTION_BACKEND`.

## Serving

```bash
uv run inferweave serve ~/huggingface/Qwen3-0.6B --port 8000
```

An OpenAI-compatible server: `/v1/completions`, `/v1/chat/completions` (both
with SSE streaming), `/v1/models`, `/health`, and `/metrics` in Prometheus
format (plus `/metrics.json` for the same numbers as a summary). Requests arrive at any time and
share one token budget per step, so a prompt being prefilled in chunks and a
batch of decoding requests run together; a client that hangs up frees its KV
blocks straight away.

```bash
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "prompt": "Hello, InferWeave.", "max_tokens": 32, "temperature": 0}'
```

Sampling parameters the engine does not implement (`top_p`, `seed`, penalties,
`n > 1`, and the rest) are refused with a 400 rather than ignored. Every engine
flag is a `Config` field; `inferweave serve --help` lists them. Design and
milestones are in [docs/online-serving.md](docs/online-serving.md).

## Benchmarks

vLLM is the baseline. Every project reports before/after numbers against it on
the same GPU, same model, same request trace — throughput, TTFT, TPOT, and
p50/p95/p99 latency, with the metrics that matter to that project called out.

```bash
uv run python bench.py
```

No benchmark numbers are published yet. The attention backends have been
verified for numerical correctness against a dense reference on an A100, but
throughput has not been measured on a GPU since the fork.

## Tests

```bash
uv run pytest tests/
```

The tests check every available backend against a dense reference that uses
neither SDPA nor paging, so the same suite runs on a laptop and on a GPU box.
Attention backend design and selection are covered in
[docs/attention-backends.md](docs/attention-backends.md).

## Credit

InferWeave is a fork of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
by Xingkai Yu, branched at
[`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8).
The scheduler, block manager, paged KV cache, and Qwen3 implementation are its
work. MIT licensed, as is this fork.
