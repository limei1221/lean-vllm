r"""Fixed batches through either engine's offline `LLM`, so both run the same shapes rather than their own schedules.

    uv run python benchmarks/bench_fixed_batch.py --engine lean-vllm --model ~/workspace/huggingface/Qwen3-0.6B
    VLLM_ENABLE_V1_MULTIPROCESSING=0 ~/workspace/vllm-env/bin/python benchmarks/bench_fixed_batch.py --engine vllm ...

Each case submits one batch at once, with token ids drawn below --vocab-size from the same seed on both engines:

- prefill: batch_size fresh prompts of --prompt-len tokens, one output token each. Fresh ids every repeat, so
  prefix caching finds nothing to reuse.
- cached: the same, behind a --prefix-len prefix that a warmup request has already cached.
- decode: per-step time of a pure-decode batch at context about --prompt-len, as
  (T(1 + decode_steps) - median T(1)) / decode_steps, which cancels the prefill.

The token budget must hold batch_size * prompt_len, or the prefill splits into chunks; the run reports the batch as
skipped. --profile-dir adds a CUDA kernel breakdown per case from torch.profiler; vLLM runs its engine in this process
only with VLLM_ENABLE_V1_MULTIPROCESSING=0, without which it records no kernels.
"""

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

# Kernel name patterns, first match wins.
KERNEL_CATEGORIES = [
    ("attention", ("flash", "fmha", "mla", "attn")),
    ("moe", ("fused_moe", "moe", "topk", "align")),
    ("gemm", ("gemm", "nvjet", "cutlass", "sm90_xmma", "matmul", "splitk", "cublas")),
    ("kv_write", ("store_kvcache", "store_latents", "reshape_and_cache", "concat_and_cache")),
    ("norm", ("rms", "norm")),
    ("rope", ("rotary", "rope")),
    ("sampling", ("sample", "argmax", "softmax", "sort", "gumbel", "exponential")),
    ("copy", ("copy", "memcpy", "memset", "fill", "cat", "index")),
]


def category(name: str) -> str:
    lowered = name.lower()
    for label, patterns in KERNEL_CATEGORIES:
        if any(pattern in lowered for pattern in patterns):
            return label
    return "other"


class Engine:
    """The few calls the cases need, on either engine."""

    def __init__(self, args):
        self.name = args.engine
        common = dict(
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            async_scheduling=True,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
        )
        if self.name == "lean-vllm":
            from lean_vllm import LLM, SamplingParams

            self.llm = LLM(os.path.expanduser(args.model), cudagraph_mode="full_and_piecewise", **common)
        else:
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                os.path.expanduser(args.model),
                compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
                seed=0,
                **common,
            )
        self.sampling_params = SamplingParams

    def generate(self, prompts: list[list[int]], max_tokens: int) -> float:
        """Seconds to generate max_tokens for every prompt, submitted together."""
        params = self.sampling_params(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        if self.name == "vllm":
            prompts = [{"prompt_token_ids": prompt} for prompt in prompts]  # type: ignore[misc]
        start = time.perf_counter()
        self.llm.generate(prompts, params, use_tqdm=False)
        return time.perf_counter() - start


class Prompts:
    def __init__(self, seed: int, vocab_size: int):
        self.rng, self.vocab_size = random.Random(seed), vocab_size

    def __call__(self, count: int, length: int, prefix: list[int] | None = None) -> list[list[int]]:
        head = prefix or []
        return [head + [self.rng.randrange(self.vocab_size) for _ in range(length)] for _ in range(count)]


def timed(engine: Engine, args, make_prompts, max_tokens: int) -> list[float]:
    engine.generate(make_prompts(), max_tokens)  # one untimed pass, for any first-call cost
    return [engine.generate(make_prompts(), max_tokens) for _ in range(args.repeats)]


def run_case(engine: Engine, args, prompts: Prompts, case: str, batch_size: int) -> dict:
    if case == "prefill":
        times = timed(engine, args, lambda: prompts(batch_size, args.prompt_len), 1)
        tokens = batch_size * args.prompt_len
    elif case == "cached":
        prefix = prompts(1, args.prefix_len)[0]
        engine.generate([prefix + [0]], 1)  # caches the prefix's whole blocks
        times = timed(engine, args, lambda: prompts(batch_size, args.prompt_len, prefix), 1)
        tokens = batch_size * args.prompt_len  # the uncached suffix
    else:
        n = args.decode_steps
        prefill = timed(engine, args, lambda: prompts(batch_size, args.prompt_len), 1)
        full = timed(engine, args, lambda: prompts(batch_size, args.prompt_len), 1 + n)
        # The median prefill, not each repeat's own, so one slow prefill does not skew a repeat.
        times = [(f - statistics.median(prefill)) / n for f in full]
        tokens = batch_size
    return {
        "case": case,
        "batch_size": batch_size,
        "seconds": times,
        "median_ms": statistics.median(times) * 1000,
        "tokens": tokens,
        "tokens_per_second": tokens / statistics.median(times),
    }


def kernel_breakdown(engine: Engine, args, prompts: Prompts, case: str, batch_size: int) -> dict:
    """CUDA kernel time by category over one pass of the case, from torch.profiler."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    max_tokens = 1 + args.decode_steps if case == "decode" else 1
    prefix = prompts(1, args.prefix_len)[0] if case == "cached" else None
    if prefix is not None:
        engine.generate([prefix + [0]], 1)
    batch = prompts(batch_size, args.prompt_len, prefix)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        engine.generate(batch, max_tokens)
        torch.cuda.synchronize()
    kernels: dict[str, float] = {}
    by_category: dict[str, float] = {}
    count = 0
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        us = event.device_time
        kernels[event.name] = kernels.get(event.name, 0.0) + us
        label = category(event.name)
        by_category[label] = by_category.get(label, 0.0) + us
        count += 1
    top = sorted(kernels.items(), key=lambda item: -item[1])[:25]
    os.makedirs(args.profile_dir, exist_ok=True)
    prof.export_chrome_trace(str(Path(args.profile_dir) / f"{engine.name}-{case}-bs{batch_size}.json"))
    return {
        "kernel_ms": sum(kernels.values()) / 1000,
        "kernels": count,
        "by_category_ms": {key: value / 1000 for key, value in sorted(by_category.items(), key=lambda kv: -kv[1])},
        "top_kernels_ms": [(name, us / 1000) for name, us in top],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=("lean-vllm", "vllm"), default="lean-vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", default="decode,prefill,cached")
    parser.add_argument("--batch-sizes", default="1,8,32,64,128,256")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--prefix-len", type=int, default=1024, help="--cases cached: tokens every prompt shares")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-dir", default="", help="also record a kernel breakdown for each case")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)
    args.cases = args.cases.split(",")
    args.batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    engine = Engine(args)
    results = []
    for case in args.cases:
        for batch_size in args.batch_sizes:
            tokens = batch_size * (args.prompt_len + (args.prefix_len if case == "cached" else 0))
            if case != "decode" and batch_size * args.prompt_len > args.max_num_batched_tokens:
                print(f"{case} bs={batch_size}: skipped, {tokens} tokens split over steps", flush=True)
                continue
            prompts = Prompts(args.seed, args.vocab_size)  # the same ids for each case on both engines
            result = run_case(engine, args, prompts, case, batch_size)
            if args.profile_dir:
                result["profile"] = kernel_breakdown(engine, args, prompts, case, batch_size)
            results.append(result)
            spread = max(result["seconds"]) / min(result["seconds"]) - 1
            print(
                f"{case:>7} bs={batch_size:<4} {result['median_ms']:9.3f} ms  "
                f"{result['tokens_per_second']:10.0f} tok/s  spread {spread:.1%}",
                flush=True,
            )
    if args.output_json:
        Path(args.output_json).write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
        print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
