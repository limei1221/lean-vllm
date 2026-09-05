"""A scheduler harness with no torch, no model and no GPU, so scheduling is testable on a laptop."""

import pytest
from dataclasses import dataclass
from itertools import count

from inferweave.engine.output import RequestOutput
from inferweave.engine.scheduler import Scheduler
from inferweave.engine.sequence import Sequence
from inferweave.sampling_params import SamplingParams

EOS = 7


@dataclass
class FakeConfig:
    """The fields Scheduler reads. Config itself needs a model directory on disk."""
    num_kvcache_blocks: int = 64
    kvcache_block_size: int = 8
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 1024
    eos: int = EOS
    enable_chunked_prefill: bool = True
    scheduling_policy: str = "fcfs"
    max_waiting_requests: int = 0
    max_num_partial_prefills: int = 0
    long_prefill_token_threshold: int = 0


class FakeModelRunner:
    """Stands in for ModelRunner: same call() surface, deterministic tokens, no torch.

    Records every batch it was handed, which is how tests assert what the
    scheduler decided rather than only what came out the far end.
    """

    def __init__(self, eos_after: dict[str, int] | None = None):
        self.eos_after = eos_after or {}
        self.batches: list[tuple[bool, list[tuple[str, int]]]] = []

    def call(self, method_name, *args):
        return getattr(self, method_name)(*args)

    def run(self, seqs: list[Sequence]) -> list[int]:
        is_prefill = any(seq.is_prefill for seq in seqs)
        self.batches.append((is_prefill, [(seq.request_id, seq.num_scheduled_tokens) for seq in seqs]))
        return [self._token(seq) for seq in seqs if self._samples(seq)]

    @staticmethod
    def _samples(seq: Sequence) -> bool:
        return seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens

    def _token(self, seq: Sequence) -> int:
        limit = self.eos_after.get(seq.request_id)
        if limit is not None and seq.num_completion_tokens >= limit:
            return EOS
        return 1000 + seq.seq_id * 100 + seq.num_completion_tokens

    def exit(self):
        pass


class FakeEngine:
    """LLMEngine.step without the model. Mirrors it deliberately; M1 changes both together."""

    def __init__(self, config: FakeConfig, runner: FakeModelRunner):
        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = runner

    def add(self, prompt: list[int], sampling_params: SamplingParams | None = None) -> Sequence:
        seq = Sequence(prompt, sampling_params or SamplingParams())
        self.scheduler.add(seq)
        return seq

    def step(self) -> list[RequestOutput]:
        output = self.scheduler.schedule()
        if not output:
            return []
        token_ids = self.model_runner.call("run", output.scheduled)
        stepped = self.scheduler.postprocess(output.scheduled, token_ids)
        return [
            RequestOutput(
                request_id=seq.request_id,
                token_ids=[seq.last_token],
                finished=seq.is_finished,
                finish_reason=seq.finish_reason,
                metrics=seq.metrics() if seq.is_finished else None,
            )
            for seq in stepped
        ]

    def is_finished(self):
        return self.scheduler.is_finished()

    def run_to_completion(self, max_steps: int = 500) -> dict[str, list[int]]:
        """Completion token ids per request, as generate() would accumulate them."""
        outputs: dict[str, list[int]] = {}
        for _ in range(max_steps):
            if self.is_finished():
                return outputs
            for output in self.step():
                outputs.setdefault(output.request_id, []).extend(output.token_ids)
        raise AssertionError("engine did not finish; a sequence is stuck")


@pytest.fixture(autouse=True)
def _reset_sequence_globals():
    """Sequence.block_size and the id counter are class state shared across tests."""
    block_size, counter = Sequence.block_size, Sequence.counter
    Sequence.counter = count()    # so seq ids are deterministic per test
    yield
    Sequence.block_size, Sequence.counter = block_size, counter


@pytest.fixture
def make_engine():
    def _make(eos_after: dict[int, int] | None = None, **overrides) -> FakeEngine:
        config = FakeConfig(**overrides)
        Sequence.block_size = config.kvcache_block_size
        return FakeEngine(config, FakeModelRunner(eos_after))
    return _make
