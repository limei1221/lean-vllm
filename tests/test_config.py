import json

import pytest

from lean_vllm.config import Config


@pytest.fixture
def model(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "max_position_embeddings": 4096,
    }))
    return str(tmp_path)


@pytest.mark.parametrize("block_size", [16, 32, 256])
def test_supported_page_sizes(model, block_size):
    assert Config(model, kvcache_block_size=block_size).kvcache_block_size == block_size


@pytest.mark.parametrize("block_size", [0, -16, 8, 17])
def test_invalid_page_sizes_are_rejected(model, block_size):
    with pytest.raises(AssertionError, match="positive multiple of 16"):
        Config(model, kvcache_block_size=block_size)
