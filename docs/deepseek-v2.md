# DeepSeek-V2: MLA, MoE and YaRN

Status: `DeepseekV2ForCausalLM` loads DeepSeek-V2-Lite and runs it eager on the
torch and FlashAttention-3 backends. It has been checked against transformers
on tiny random checkpoints only. It has not yet run on the real 16B weights or
been benchmarked against vLLM.

## Running it

```bash
uv run hf download deepseek-ai/DeepSeek-V2-Lite-Chat --local-dir ~/workspace/huggingface/DeepSeek-V2-Lite-Chat
uv run lean-vllm serve ~/workspace/huggingface/DeepSeek-V2-Lite-Chat --served-model-name deepseek
```

The weights are 31 GB in bf16, so a 40 GB GPU is the floor. The config and
tokenizer load natively from transformers 4.56 on, with no `trust_remote_code`.
The runner picks the model class from `architectures` in `config.json`
(`lean_vllm/models/__init__.py`).

## MLA: the cache holds latents

Each token caches `kv_lora_rank + qk_rope_head_dim` values per layer. That is
the normalized compressed KV plus the shared rope key, with rope already
applied. Keys and values per head are not cached. For V2-Lite that is 576
values against the 2 × 16 × 192 = 6144 a plain KV cache would need, about 31 KB
per token in bf16 across 27 layers.

`MLAAttention` (`layers/attention.py`) is an `Attention` layer with its own
layout. The runner asks every layer for `kv_cache_shape` and hands its slice
back through `bind_kv_cache`, rather than reading head counts off the config.
Each step it:

1. scatters the step's latents into their slots;
2. reads the latent of every key the step attends. When no row resumes
   (`keys_are_new`) these are the step's own latents. Otherwise it gathers
   through `key_slots`, which is computed once per step from the block tables
   and `cu_seqlens_k`. The gather's size comes from `num_keys` on the host, so
   it does not sync;
3. expands the latents into keys and values with `kv_b_proj`;
4. runs the backend's varlen `prefill` with no page table, where a decode row
   is simply a row whose query is one token long.

Values are zero-padded from `v_head_dim` (128) to the query/key head size (192),
so a backend still sees one head size. The softmax scale is passed explicitly,
so the padding changes nothing, and the output is cut back to 128.

The trade-off is decode. Every step re-expands the whole context in every layer,
which costs compute in proportion to context length. FlashMLA avoids that by
attending over the latents directly. This path comes first because it is
obviously correct. An MLA decode kernel should land before any numbers are
published.

## MoE

`FusedMoE` (`layers/moe.py`) stacks the routed experts into one `gate_up_proj`
of shape `[E, 2I, H]` and one `down_proj` of shape `[E, H, I]`. The loader maps
each checkpoint weight `experts.{e}.{proj}.weight` into its expert's row. The
forward pass sorts tokens by expert and runs two `F.grouped_mm` calls. Group
offsets come from `searchsorted` rather than `bincount`, which syncs on CUDA.
Routing follows the original V2 code: softmax scores, `greedy` or
`group_limited_greedy` selection, then either `routed_scaling_factor` or
`norm_topk_prob`. Tensor parallelism shards each expert's intermediate size.
Shared experts reuse the dense gated MLP.

## YaRN

`rotary_embedding.py` computes YaRN's inverse frequencies and scales cos and sin
by `mscale / mscale_all_dim`, a ratio that is 1 for V2-Lite. The attention layer
multiplies the softmax scale by `yarn_get_mscale(factor, mscale_all_dim)²`,
about 1.59 at V2-Lite's factor of 40. DeepSeek's rope rotates adjacent pairs,
GPT-J style, so its rotary embedding is built with `is_neox_style=False`.

## CUDA graphs

This model reports `supports_cuda_graph = False`, so it runs eager everywhere.
The key gather is sized per step, and piecewise capture expects the q/k/v
pieces of the Qwen3 layer split.

## Testing

`tests/test_deepseek_v2.py` loads one checkpoint on disk into both
transformers' eager implementation and this one, in fp32 on the torch backend.
It runs three configs: V2-Lite's shape, one with `q_lora_rank`, and one with
group-limited routing. It compares a whole prompt, then a sequence of paged
steps built by the runner's own `prepare_batch`: a chunk, a resumed chunk beside
a cold prompt, pure decode, and decode mixed with a prompt. The block tables are
scattered.

Mutations the suite catches: resumed rows reading only their new latents, a key
gather that reads only the first page, uncut value padding, a dropped softmax
mscale, a dropped cos/sin attention factor, and unscaled routing weights.

Writing it turned up an fp32 bug in `RMSNorm`. `.float()` and `.to()` alias an
fp32 tensor, so the in-place normalization rewrote the residual. bf16 always
copies, which is why Qwen3 never hit it.

End to end, `LLM.generate` ran greedy on a tiny random checkpoint with
V2-Lite's tokenizer, an 8-token step budget and repeated prompts. It matched
transformers' `generate` token for token.

## Next

1. Run the real weights: compare outputs with vLLM, then benchmark.
2. An MLA decode kernel over latents (FlashMLA, or Triton off Hopper).
3. CUDA graphs for MLA and MoE, and a check of `grouped_mm` against a fused
   Triton MoE on an H100.
