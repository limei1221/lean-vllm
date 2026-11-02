_IMPORT_ERROR: ImportError | None = None
try:
    import triton
    import triton.language as tl
except ImportError as e:  # installed by the cuda extra
    _IMPORT_ERROR = e
else:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        block_size,
        block_stride,
        token_stride,
        head_stride,
        HEAD_DIM: tl.constexpr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx).to(tl.int64)
        if slot == -1:
            return
        offsets = tl.arange(0, D)
        key = tl.load(key_ptr + idx * key_stride + offsets)
        value = tl.load(value_ptr + idx * value_stride + offsets)
        # Use the cache strides to support both NHD storage and a transposed view of HND storage.
        block, token = slot // block_size, slot % block_size
        head, dim = offsets // HEAD_DIM, offsets % HEAD_DIM
        cache_offsets = block * block_stride + token * token_stride + head * head_stride + dim
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    @triton.jit
    def store_latents_kernel(
        latent_ptr,
        latent_stride,
        cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        offsets = tl.arange(0, BLOCK)
        mask = offsets < D  # skip padding when D is smaller than the power-of-two block
        latent = tl.load(latent_ptr + idx * latent_stride + offsets, mask=mask)
        tl.store(cache_ptr + slot * D + offsets, latent, mask=mask)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping) -> None:
    """Write keys and values [num_tokens, heads, dim] into caches at the slots given by slot_mapping.

    Caches have shape [num_blocks, block_size, heads, dim]. Use their strides to support both NHD storage
    and HND storage viewed with the token and head axes transposed. Skip tokens whose slot_mapping entry is -1.
    """
    num_tokens, num_heads, head_dim = key.shape
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride() == v_cache.stride() and k_cache.stride(-1) == 1
    assert slot_mapping.numel() == num_tokens
    block_stride, token_stride, head_stride, _ = k_cache.stride()
    store_kvcache_kernel[(num_tokens,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        k_cache.size(1),
        block_stride,
        token_stride,
        head_stride,
        head_dim,
        num_heads * head_dim,
    )


def store_latents(latent, latent_cache, slot_mapping) -> None:
    """Write latents [num_tokens, dim] into a contiguous cache of shape [num_blocks, block_size, dim].

    slot_mapping gives each token's slot across all blocks. Skip tokens whose slot_mapping entry is -1.
    """
    num_tokens, dim = latent.shape
    assert latent.stride(-1) == 1 and latent_cache.stride(-2) == dim
    assert slot_mapping.numel() == num_tokens
    store_latents_kernel[(num_tokens,)](
        latent, latent.stride(0), latent_cache, slot_mapping, dim, triton.next_power_of_2(dim)
    )
