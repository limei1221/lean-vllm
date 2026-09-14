"""DeepSeek-V2 (MLA, MoE, YaRN) against transformers' implementation, on a tiny random checkpoint.

fp32 on the torch backend. Paged steps go through the runner's real batch preparation, so the
latent cache, its key gather and mixed prefill/decode batches are all compared, not just a prompt.
"""

import pytest
import torch
import torch.distributed as dist
from transformers import DeepseekV2Config
from transformers import DeepseekV2ForCausalLM as HFDeepseekV2ForCausalLM

from lean_vllm.engine.model_runner import ModelRunner
from lean_vllm.engine.sequence import Sequence
from lean_vllm.layers.attention import MLAAttention, register_layers
from lean_vllm.models import get_model_class
from lean_vllm.models.deepseek_v2 import DeepseekV2ForCausalLM
from lean_vllm.utils.context import set_context
from lean_vllm.utils.loader import load_model

BLOCK_SIZE = 4
NUM_BLOCKS = 16

CONFIGS = {
    "lite": {},
    "q_lora": dict(q_lora_rank=24),
    "grouped_routing": dict(topk_method="group_limited_greedy", n_group=4, topk_group=2),
}


def tiny_config(**overrides) -> DeepseekV2Config:
    """DeepSeek-V2-Lite's shape in miniature: one dense layer, then MoE with a shared expert."""
    return DeepseekV2Config(**{
        **dict(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=4, intermediate_size=96,
            moe_intermediate_size=24, n_routed_experts=8, num_experts_per_tok=3, n_shared_experts=1,
            first_k_dense_replace=1, num_hidden_layers=3, vocab_size=128, max_position_embeddings=256,
            q_lora_rank=None, kv_lora_rank=16, qk_nope_head_dim=12, qk_rope_head_dim=8,
            v_head_dim=10,    # not the qk head size, so the value padding is exercised
            rope_parameters={
                "rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                "mscale": 1.0, "mscale_all_dim": 0.707,    # unequal, so cos and sin are scaled too
                "original_max_position_embeddings": 32, "beta_fast": 32, "beta_slow": 1,
            },
            initializer_range=0.1,
        ),
        **overrides,
    })


@pytest.fixture(scope="module")
def process_group():
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture(scope="module", params=list(CONFIGS))
def models(request, process_group, tmp_path_factory):
    """The transformers model and ours, loaded from the same checkpoint on disk."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp(request.param)
    HFDeepseekV2ForCausalLM(tiny_config(**CONFIGS[request.param])).save_pretrained(path)
    reference = HFDeepseekV2ForCausalLM.from_pretrained(path, attn_implementation="eager").eval()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LEAN_VLLM_ATTENTION_BACKEND", "torch")
        model = get_model_class(reference.config)(reference.config)
    load_model(model, str(path))
    return reference, model


@pytest.fixture
def runner():
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner.device = torch.device("cpu")
    runner.block_size = BLOCK_SIZE
    runner._prev_tokens = runner._prev_rows = None
    return runner


def reference_logits(reference, tokens: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return reference(torch.tensor([tokens])).logits[0]


def row(tokens: list[int], num_cached: int, num_new: int, block_table: list[int], decode: bool = False) -> Sequence:
    """A sequence with num_cached tokens already in the cache and the next num_new scheduled."""
    seq = Sequence(tokens[:num_cached + num_new])
    seq.num_cached_tokens, seq.num_scheduled_tokens = num_cached, num_new
    seq.is_prefill = not decode
    seq.block_table = block_table
    return seq


def step(runner, model, seqs: list[Sequence]) -> list[torch.Tensor]:
    """Logits at every scheduled position, split by row."""
    register_layers(model)    # the op finds layers by name, and each config's model reuses the names
    input_ids, positions, _, context = runner.prepare_batch(seqs)
    context["logits_indices"] = None
    with torch.inference_mode(), set_context(**context):
        logits = model.compute_logits(model(input_ids, positions))
    return list(logits.split([seq.num_scheduled_tokens for seq in seqs]))


def test_a_prompt_matches_transformers(models, runner):
    reference, model = models
    tokens = torch.randint(0, 128, (13,)).tolist()

    (logits,) = step(runner, model, [row(tokens, 0, len(tokens), block_table=[])])

    torch.testing.assert_close(logits, reference_logits(reference, tokens), rtol=1e-4, atol=1e-4)


def test_paged_steps_match_transformers(models, runner):
    """Chunks, a cold prompt beside a resumed one, pure decode, then decode mixed with a prompt."""
    reference, model = models
    layers = [module for module in model.modules() if isinstance(module, MLAAttention)]
    cache = torch.zeros(len(layers), *layers[0].kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE))
    for layer, layer_cache in zip(layers, cache):
        layer.bind_kv_cache(layer_cache)
    a, b, c = (torch.randint(0, 128, (n,)).tolist() for n in (11, 8, 3))
    table_a, table_b, table_c = [5, 2, 9], [7, 0], [3]    # scattered, so a wrong page walk shows

    want_a, want_b, want_c = (reference_logits(reference, tokens) for tokens in (a, b, c))

    steps = [
        [(row(a, 0, 5, table_a), want_a)],
        [(row(a, 5, 4, table_a), want_a), (row(b, 0, 7, table_b), want_b)],
        [(row(a, 9, 1, table_a, decode=True), want_a), (row(b, 7, 1, table_b, decode=True), want_b)],
        [(row(a, 10, 1, table_a, decode=True), want_a), (row(c, 0, 3, table_c), want_c)],
    ]
    for rows in steps:
        seqs = [seq for seq, _ in rows]
        for (seq, want), got in zip(rows, step(runner, model, seqs)):
            start = seq.num_cached_tokens
            torch.testing.assert_close(got, want[start:start + seq.num_scheduled_tokens], rtol=1e-4, atol=1e-4)


def test_the_cache_holds_one_latent_per_token(models):
    """The point of MLA: kv_lora_rank + rope dim per token and layer, not keys and values per head."""
    _, model = models
    layer = next(module for module in model.modules() if isinstance(module, MLAAttention))
    assert layer.kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE) == (1, NUM_BLOCKS, BLOCK_SIZE, 16 + 8)


def test_an_unknown_architecture_is_refused():
    config = DeepseekV2Config(architectures=["GPT2LMHeadModel"])
    with pytest.raises(ValueError, match="unsupported architectures"):
        get_model_class(config)


def test_the_registry_resolves_deepseek_v2():
    assert get_model_class(DeepseekV2Config(architectures=["DeepseekV2ForCausalLM"])) is DeepseekV2ForCausalLM
