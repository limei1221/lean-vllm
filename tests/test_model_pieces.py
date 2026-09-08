"""The decoder layer either side of attention, which is what piecewise capture takes.

bf16 throughout, because that is what the runner sets the default dtype to and
because RMSNorm's in-place arithmetic only copies its input when a cast is
needed: in fp32 it would rewrite the caller's tensor.
"""

import pytest
import torch
import torch.distributed as dist
from transformers import Qwen3Config

from lean_vllm.models.qwen3 import Qwen3DecoderLayer
from lean_vllm.utils.context import reset_context

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, TOKENS = 32, 4, 2, 8, 3


@pytest.fixture(scope="module")
def process_group():
    """The layers read the world size when they are built. A store, so no port."""
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture
def layer(process_group):
    torch.manual_seed(0)
    config = Qwen3Config(
        hidden_size=HIDDEN, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM, intermediate_size=64, num_hidden_layers=1,
        vocab_size=128, max_position_embeddings=64,
    )
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.inference_mode():
        layer = Qwen3DecoderLayer(config)
        for param in layer.parameters():
            param.normal_(0, 0.05)    # the weights are torch.empty until a checkpoint lands
        yield layer
    torch.set_default_dtype(default)
    reset_context()


@pytest.fixture
def inputs():
    return torch.arange(TOKENS), torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16)


class FakeAttention(torch.nn.Module):
    """Deterministic and shaped like the real thing, so both paths see one value."""

    def forward(self, q, k, v):
        return q * 0.5


fake_attention = FakeAttention()


def reference(layer, positions, hidden_states, residual):
    """The layer written out as it read before the split, module for module."""
    if residual is None:
        hidden_states, residual = layer.input_layernorm(hidden_states), hidden_states
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)
    attn = layer.self_attn
    qkv = attn.qkv_proj(hidden_states)
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q = q.view(-1, attn.num_heads, attn.head_dim)
    k = k.view(-1, attn.num_kv_heads, attn.head_dim)
    v = v.view(-1, attn.num_kv_heads, attn.head_dim)
    if not attn.qkv_bias:
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    q, k = attn.rotary_emb(positions, q, k)
    hidden_states = attn.o_proj(fake_attention(q, k, v).flatten(1, -1))
    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
    return layer.mlp(hidden_states), residual


@pytest.mark.parametrize("first_layer", [True, False])
def test_the_split_layer_matches_the_unsplit_one(layer, inputs, first_layer):
    """Bitwise, on the same weights: the split must not have moved any arithmetic."""
    positions, hidden_states = inputs
    residual = None if first_layer else torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16)

    q, k, v, carried = layer.pre_attention(positions, hidden_states.clone(), residual)
    got, got_residual = layer.post_attention(fake_attention(q, k, v), carried)
    want, want_residual = reference(layer, positions, hidden_states.clone(), residual)

    assert torch.equal(got, want)
    assert torch.equal(got_residual, want_residual)


def test_the_pieces_need_no_attention_context(layer, inputs):
    """What a graph replays cannot depend on this step's sequence layout."""
    reset_context()    # any read of it would see an empty Context and misbehave
    positions, hidden_states = inputs

    q, k, v, residual = layer.pre_attention(positions, hidden_states, None)
    hidden_states, residual = layer.post_attention(fake_attention(q, k, v), residual)

    assert q.shape == (TOKENS, HEADS, HEAD_DIM)
    assert k.shape == v.shape == (TOKENS, KV_HEADS, HEAD_DIM)
    assert hidden_states.shape == residual.shape == (TOKENS, HIDDEN)


def test_forward_still_runs_the_pieces(layer, inputs, monkeypatch):
    """forward is the composition, so the captured path cannot drift from it."""
    monkeypatch.setattr(layer.self_attn, "attn", FakeAttention())
    positions, hidden_states = inputs

    got, got_residual = layer(positions, hidden_states.clone(), None)
    want, want_residual = reference(layer, positions, hidden_states.clone(), None)

    assert torch.equal(got, want)
    assert torch.equal(got_residual, want_residual)
