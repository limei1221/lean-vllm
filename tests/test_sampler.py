"""temperature == 0 means greedy, and must not divide the logits by zero."""

import torch

from inferweave.layers.sampler import Sampler

torch.manual_seed(0)


def logits(batch: int = 4, vocab: int = 32) -> torch.Tensor:
    return torch.randn(batch, vocab)


def test_zero_temperature_is_argmax():
    x = logits()
    temperatures = torch.zeros(x.size(0))
    assert torch.equal(Sampler()(x, temperatures), x.argmax(dim=-1))


def test_greedy_and_random_rows_coexist_in_one_batch():
    x = logits()
    temperatures = torch.tensor([0.0, 1.0, 0.0, 1.0])
    tokens = Sampler()(x, temperatures)
    assert torch.equal(tokens[::2], x.argmax(dim=-1)[::2])
    assert ((tokens >= 0) & (tokens < x.size(1))).all()


def test_a_peaked_distribution_still_samples_its_peak():
    x = torch.full((2, 16), -20.0)
    x[:, 3] = 20.0
    tokens = Sampler()(x, torch.ones(2))
    assert torch.equal(tokens, torch.tensor([3, 3]))
