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
| 1 | Online serving + advanced scheduler | next |
| 2 | DeepSeek-style model support: MLA + MoE + YaRN | |
| 3 | Speculative decoding | |
| 4 | Disaggregated prefill / decode | |

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # deps, dev tools and the package, into .venv
uv sync --extra cuda     # add FlashAttention and Triton (NVIDIA only)
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

## Benchmarks

vLLM is the baseline. Every project reports before/after numbers against it on
the same GPU, same model, same request trace — throughput, TTFT, TPOT, and
p50/p95/p99 latency, with the metrics that matter to that project called out.

```bash
uv run python bench.py
```

No numbers are published yet: nothing has been measured on a GPU since the fork.

## Attention backends

Model code declares attention semantics; a backend owns execution. Selection is
automatic, or forced with `INFERWEAVE_ATTENTION_BACKEND=torch|flash_attn`.

```bash
uv run pytest tests/
```

The tests check every available backend against a dense reference that uses
neither SDPA nor paging, so the same suite runs on a laptop and on a GPU box.
Details in [docs/attention-backends.md](docs/attention-backends.md).

## Credit

InferWeave is a fork of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
by Xingkai Yu, branched at
[`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8).
The scheduler, block manager, paged KV cache, and Qwen3 implementation are its
work. MIT licensed, as is this fork.
