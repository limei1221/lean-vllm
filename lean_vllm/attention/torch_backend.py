import torch
import torch.nn.functional as F

from lean_vllm.attention.abstract import AttentionBackend
from lean_vllm.utils.context import Context


def _probe_enable_gqa() -> bool:
    q = torch.zeros(1, 2, 1, 1)
    k = torch.zeros(1, 1, 1, 1)
    try:
        F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
    except TypeError:
        return False
    return True


_SDPA_ENABLE_GQA = _probe_enable_gqa()


class TorchAttention(AttentionBackend):
    """SDPA reference backend. Runs anywhere; optimized for clarity, not speed."""

    @staticmethod
    def get_name() -> str:
        return "torch"

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def supports_cuda_graph() -> bool:
        return False

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping) -> None:
        num_tokens = key.size(0)
        assert slot_mapping.numel() == num_tokens
        dim = self.num_kv_heads * self.head_dim

        slots = slot_mapping.long()
        keep = slots >= 0
        if not keep.all():    # costs a host sync; kernels mask in-kernel instead
            slots = slots[keep]
            key = key[keep]
            value = value[keep]

        k_cache.view(-1, dim).index_copy_(0, slots, key.reshape(-1, dim))
        v_cache.view(-1, dim).index_copy_(0, slots, value.reshape(-1, dim))

    def prefill(self, q, k, v, k_cache, v_cache, context: Context) -> torch.Tensor:
        cu_seqlens_q = context.cu_seqlens_q.tolist()
        cu_seqlens_k = context.cu_seqlens_k.tolist()
        block_tables = context.block_tables

        outputs = []
        for i in range(len(cu_seqlens_q) - 1):
            seqlen_q = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
            seqlen_k = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
            q_i = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
            if block_tables is not None:    # prefix cache
                k_i = self._gather_pages(k_cache, block_tables[i], seqlen_k)
                v_i = self._gather_pages(v_cache, block_tables[i], seqlen_k)
            else:
                k_i = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
                v_i = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
            mask = self._causal_mask(seqlen_q, seqlen_k, q.device)
            outputs.append(self._sdpa(q_i, k_i, v_i, mask))
        return torch.cat(outputs, dim=0)

    def decode(self, q, k_cache, v_cache, context: Context) -> torch.Tensor:
        block_tables = context.block_tables
        outputs = []
        for i, seqlen_k in enumerate(context.context_lens.tolist()):
            k_i = self._gather_pages(k_cache, block_tables[i], seqlen_k)
            v_i = self._gather_pages(v_cache, block_tables[i], seqlen_k)
            outputs.append(self._sdpa(q[i:i + 1], k_i, v_i, None))
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _gather_pages(cache: torch.Tensor, block_table: torch.Tensor, seqlen: int) -> torch.Tensor:
        block_size = cache.size(1)
        num_blocks = (seqlen + block_size - 1) // block_size
        blocks = block_table[:num_blocks].long()
        return cache[blocks].reshape(-1, cache.size(2), cache.size(3))[:seqlen]

    @staticmethod
    def _causal_mask(seqlen_q: int, seqlen_k: int, device: torch.device) -> torch.Tensor | None:
        # Bottom-right aligned: query j sits at absolute position seqlen_k - seqlen_q + j.
        # Not the same as SDPA is_causal=True, which aligns top-left and is wrong here.
        if seqlen_q == 1:
            return None
        q_pos = torch.arange(seqlen_k - seqlen_q, seqlen_k, device=device).unsqueeze(1)
        k_pos = torch.arange(seqlen_k, device=device).unsqueeze(0)
        return q_pos >= k_pos

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        q = q.transpose(0, 1).unsqueeze(0) # [1, H, Lq, D]
        k = k.transpose(0, 1).unsqueeze(0) # [1, H_kv, Lk, D]
        v = v.transpose(0, 1).unsqueeze(0) # [1, H_kv, Lk, D]
        if mask is not None:
            mask = mask.view(1, 1, *mask.shape)

        gqa = self.num_heads != self.num_kv_heads
        if gqa and not _SDPA_ENABLE_GQA:    # torch < 2.5
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
            kwargs = {}
        else:
            kwargs = {"enable_gqa": gqa} if _SDPA_ENABLE_GQA else {}

        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale, **kwargs) # [1, H, Lq, D]
        return o.squeeze(0).transpose(0, 1)
