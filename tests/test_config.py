"""Config's defaults that depend on other fields, without a model on disk."""

import pytest

from lean_vllm import config as config_module
from lean_vllm.config import Config


class FakeHFConfig:
    max_position_embeddings = 4096


@pytest.fixture
def make_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module.AutoConfig, "from_pretrained", lambda path: FakeHFConfig())
    return lambda **kwargs: Config(str(tmp_path), **kwargs)


def test_async_scheduling_is_on_by_default(make_config):
    assert make_config().async_scheduling


def test_tensor_parallelism_turns_async_scheduling_off(make_config, caplog):
    """Ranks above zero never see the sampled tokens."""
    config = make_config(tensor_parallel_size=2)
    assert not config.async_scheduling
    assert "async_scheduling is off" in caplog.text
