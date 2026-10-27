"""DeepSeek-V3's MTP drafter on CUDA: FlashMLA decode, the Triton MoE, CUDA graphs and async scheduling, on a tiny
random checkpoint with V3's attention and routing shapes. test_mtp.py checks the same steps exactly on CPU.

bf16 can flip a near-tied pick, so tokens and drafts are checked against an fp32 CPU reference within a tolerance.
A drafter that predicts the target exactly has margins no rounding flips, so its drafts must all be kept.
"""

import atexit
import json
import shutil

import pytest
import torch
from conftest import total
from test_mtp import (
    ENGINE_ARGS,
    PROMPTS,
    VOCAB,
    generate,
    keep_drafts_at_random,
    no_cache_context,
    predict_the_target,
    record_drafts,
    reference_draft_logits,
    write_checkpoint,
)

from lean_vllm.attention.flashmla_backend import FlashMLABackend
from lean_vllm.layers.fused_moe import use_triton
from lean_vllm.utils.context import set_context

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# V3's attention and routing in miniature: FlashMLA's 512 + 64 latent, and 8 of 256 experts from 4 of 8 groups.
V3_SHAPE = dict(
    hidden_size=256,
    num_attention_heads=16,
    num_key_value_heads=16,
    intermediate_size=256,
    moe_intermediate_size=32,
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_group=8,
    topk_group=4,
    q_lora_rank=64,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
)
CUDA_ARGS = {**ENGINE_ARGS, "kvcache_block_size": 64}  # FlashMLA's page size
MODES = {"eager": dict(enforce_eager=True), "graphs": dict(enforce_eager=False, cudagraph_mode="full_and_piecewise")}
# How far below the fp32 pick a bf16 pick may score, in standard deviations of the row's logits, and the share of
# picks that must match it exactly. bf16 on CPU measured 0.48 at worst and 86-95% exact; a token from a wrong
# distribution is within 1.0 about 5% of the time. The k=0 case of test_greedy_tokens_are_the_fp32_reference_s checks
# these against plain decoding's own noise.
TOLERANCE = 1.0
MIN_EXACT = 0.75


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """The same bf16 weights twice: run in bf16 on CUDA, and upcast to fp32 on CPU for the reference."""
    cuda = tmp_path_factory.mktemp("deepseek_v3_mtp_bf16")
    write_checkpoint(cuda, dtype=torch.bfloat16, **V3_SHAPE)
    cpu = tmp_path_factory.mktemp("deepseek_v3_mtp_fp32")
    shutil.copytree(cuda, cpu, dirs_exist_ok=True)
    with open(cpu / "config.json") as f:
        config = json.load(f)
    config["dtype"] = "float32"
    with open(cpu / "config.json", "w") as f:
        json.dump(config, f)
    return {"cuda": str(cuda), "cpu": str(cpu)}


@pytest.fixture
def make_engine(checkpoints, monkeypatch):
    from lean_vllm.engine.llm_engine import LLMEngine

    engines = []

    def close():
        """One process group at a time, so the engine before goes first."""
        while engines:
            engine = engines.pop()
            engine.exit()
            atexit.unregister(engine.exit)

    def make(num_speculative_tokens: int = 0, device: str = "cuda", **kwargs):
        close()
        if device == "cpu":
            monkeypatch.setenv("LEAN_VLLM_DEVICE", "cpu")
            monkeypatch.setenv("LEAN_VLLM_ATTENTION_BACKEND", "torch")
        else:
            monkeypatch.delenv("LEAN_VLLM_DEVICE", raising=False)
            monkeypatch.delenv("LEAN_VLLM_ATTENTION_BACKEND", raising=False)
        if num_speculative_tokens:
            kwargs["speculative_config"] = json.dumps(
                {"method": "mtp", "num_speculative_tokens": num_speculative_tokens}
            )
        args = ENGINE_ARGS if device == "cpu" else CUDA_ARGS
        engines.append(LLMEngine(checkpoints[device], **{**args, **kwargs}))
        return engines[-1]

    yield make
    close()


def check_cuda_paths(engine, mode: str):
    """The paths under test run, not a fallback: FlashMLA for every MLA layer, the Triton MoE, the graphs asked for."""
    runner = engine.model_runner
    assert runner.attention_backends == [FlashMLABackend], "FlashMLA is not available: build it (Hopper only)"
    assert use_triton(torch.zeros(1, device="cuda")), "the routed experts would not run in Triton"
    assert runner.cudagraph_mode == ("none" if mode == "eager" else "full_and_piecewise")


@torch.inference_mode()
def reference_logits(engine, token_ids: list[int]) -> torch.Tensor:
    """The target's logits after each prefix of token_ids, in one pass with no cache."""
    model = engine.model_runner.model
    n = len(token_ids)
    with set_context(**no_cache_context(n)):
        return model.compute_logits(model(torch.tensor(token_ids), torch.arange(n)))


def check_picks(picks: list[tuple[torch.Tensor, int, str]]):
    """Each (fp32 logits, token picked, where) scores within TOLERANCE of the fp32 pick, and most are that pick."""
    exact = 0
    for logits, token, where in picks:
        best = logits.max()
        assert logits[token] >= best - TOLERANCE * logits.std(), (
            f"{where}: picked {token} scoring {float(logits[token]):.3f}, "
            f"the fp32 pick {int(logits.argmax())} scores {float(best):.3f}"
        )
        exact += int(logits.argmax()) == token
    assert exact >= MIN_EXACT * len(picks), f"{exact} of {len(picks)} picks are the fp32 pick"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", [1, 3])
def test_a_drafter_that_predicts_the_target_has_every_draft_kept(make_engine, k, mode):
    """Weights changed in place after capture, so graph replays and compiled pieces read them too."""
    engine = make_engine(k, **MODES[mode])
    check_cuda_paths(engine, mode)
    predict_the_target(engine)
    for prompt, tokens in zip(PROMPTS, generate(engine)):
        assert all(b == (a + 1) % VOCAB for a, b in zip([prompt[-1], *tokens], tokens))
    metrics = engine.metrics
    assert total(metrics.spec_accepted_tokens) == total(metrics.spec_draft_tokens) > 0


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", [0, 2])
def test_greedy_tokens_are_the_fp32_reference_s(make_engine, k, mode, async_scheduling):
    """Each token is the fp32 target's pick after the tokens before it, whether rows verify 1 + k tokens and reject
    their drafts or decode one; long prompts prefill in chunks beside other rows' steps. k=0 checks the tolerance."""
    engine = make_engine(k, async_scheduling=async_scheduling, **MODES[mode])
    check_cuda_paths(engine, mode)
    outputs = generate(engine)
    if k:
        assert total(engine.metrics.spec_draft_tokens) > 0
    reference = make_engine(device="cpu")
    picks = []
    for prompt, tokens in zip(PROMPTS, outputs):
        logits = reference_logits(reference, prompt + tokens[:-1])
        for i, token in enumerate(tokens):
            picks.append((logits[len(prompt) - 1 + i], token, f"token {i} after a {len(prompt)}-token prompt"))
    check_picks(picks)


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("mode", MODES)
def test_the_drafts_are_the_fp32_reference_s(make_engine, monkeypatch, mode, async_scheduling):
    """Each draft is the fp32 drafter's pick after the tokens and drafts before it. Rows keep a random number of
    drafts, so kept and rejected drafts' slots both feed the drafter's cache, eager or replaying graphs."""
    k = 2
    engine = make_engine(k, async_scheduling=async_scheduling, **MODES[mode])
    check_cuda_paths(engine, mode)
    monkeypatch.setattr(engine.model_runner, "rejection_sampler", keep_drafts_at_random)
    proposed = record_drafts(engine, monkeypatch)
    generate(engine, max_tokens=12)
    assert len(proposed) > len(PROMPTS)
    assert total(engine.metrics.spec_accepted_tokens) > 0
    reference = make_engine(1, device="cpu")
    check_picks(
        [
            (logits, drafts[step], f"draft {step} after {len(token_ids)} tokens")
            for token_ids, drafts in proposed
            for step, logits in enumerate(reference_draft_logits(reference, token_ids, k, drafts))
        ]
    )
