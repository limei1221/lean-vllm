"""The real engine loop with a fake runner: no model loading or GPU needed."""

import asyncio
from dataclasses import dataclass
from functools import wraps
from itertools import count

import pytest

from lean_vllm.engine.llm_engine import LLMEngine
from lean_vllm.engine.metrics import Metrics
from lean_vllm.engine.output import RequestOutput
from lean_vllm.engine.scheduler import Scheduler
from lean_vllm.engine.sequence import Sequence
from lean_vllm.kv_transfer import KVConnectorOutput
from lean_vllm.sampling_params import SamplingParams

EOS = 7


@dataclass
class FakeConfig:
    """The fields Scheduler reads. Config itself needs a model directory on disk."""

    num_kvcache_blocks: int = 64
    kvcache_block_size: int = 8
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 1024
    max_model_len: int = 4096
    eos: int = EOS
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    prefix_caching_hash_algo: str = "sha256"
    scheduling_policy: str = "fcfs"
    long_prefill_token_threshold: int = 0
    async_scheduling: bool = False  # off, unlike Config: most tests count steps in sync order
    tensor_parallel_size: int = 1
    kv_transfer: object = None  # a KVTransferConfig turns the connector on
    speculative: object = None  # a SpeculativeConfig turns drafting on


class FakeSampledTokens:
    """What ModelRunner.run returns, without torch."""

    def __init__(self, token_ids: list[int]):
        self._token_ids = token_ids

    def tolist(self) -> list[int]:
        return self._token_ids


class FakeModelRunner:
    """Stands in for ModelRunner: same call() surface, deterministic tokens, every batch recorded."""

    step_kind = "enforced"

    def __init__(self, eos_after: dict[str, int] | None = None):
        self.eos_after = eos_after or {}
        self.batches: list[tuple[bool, list[tuple[str, int]]]] = []
        self.kv_metadata: list = []  # what each kv_connector_step was handed
        self.kv_outputs: list = []  # what the next ones report, oldest first

    def kv_connector_step(self, metadata):
        self.kv_metadata.append(metadata)
        return self.kv_outputs.pop(0) if self.kv_outputs else KVConnectorOutput()

    def call(self, method_name, *args):
        return getattr(self, method_name)(*args)

    def run(self, seqs: list[Sequence]) -> "FakeSampledTokens":
        is_prefill = any(seq.is_prefill for seq in seqs)
        self.batches.append((is_prefill, [(seq.request_id, seq.num_scheduled_tokens) for seq in seqs]))
        return FakeSampledTokens([self._token(seq) for seq in seqs if self._samples(seq)])

    @staticmethod
    def _samples(seq: Sequence) -> bool:
        return seq.num_cached_tokens + seq.num_scheduled_tokens >= seq.num_planned_tokens

    @staticmethod
    def _completion_index(seq: Sequence) -> int:
        """Which completion token this row is about to produce, read off the batch so a recomputed suffix repeats."""
        return seq.num_cached_tokens + seq.num_scheduled_tokens - seq.num_prompt_tokens

    def _token(self, seq: Sequence) -> int:
        limit = self.eos_after.get(seq.request_id)
        index = self._completion_index(seq)
        if limit is not None and index >= limit:
            return EOS
        return 1000 + seq.seq_id * 100 + index

    def exit(self):
        pass


class FakeLLMEngine(LLMEngine):
    """The production step loop with a fake runner and direct scheduler admission."""

    tokenizer = None

    def __init__(self, config: FakeConfig, runner: FakeModelRunner):
        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = runner
        self.metrics = Metrics()
        self.last_output = None
        self.in_flight = None
        self.profiler = None
        self.output_processor = None
        self.ps = []

    def add(
        self, prompt: list[int], sampling_params: SamplingParams | None = None, request_id: str | None = None
    ) -> Sequence:
        seq = Sequence(prompt, sampling_params or SamplingParams(), request_id)
        self.scheduler.add(seq)
        self.metrics.record_received()
        return seq

    def add_request(self, prompt: list[int], sampling_params=None, request_id: str | None = None) -> str:
        return self.add(prompt, sampling_params, request_id).request_id

    def _launch(self):
        self.last_output = super()._launch()
        return self.last_output


class FakeEngine(FakeLLMEngine):
    """Scheduler tests consume only outputs; core-client tests use FakeLLMEngine's full step result."""

    def step(self) -> list[RequestOutput]:
        outputs, _, _ = super().step()
        return outputs

    def run_to_completion(self, max_steps: int = 500) -> dict[str, list[int]]:
        """Completion token ids per request, as generate() would accumulate them."""
        outputs: dict[str, list[int]] = {}
        for _ in range(max_steps):
            if self.is_finished():
                return outputs
            for output in self.step():
                outputs.setdefault(output.request_id, []).extend(output.token_ids)
        raise AssertionError("engine did not finish; a sequence is stuck")


def by_label(metric) -> dict:
    """A counter's or gauge's value per label value; None keys the one value of an unlabelled metric."""
    (family,) = metric.collect()
    names = (family.name, family.name + "_total")
    return {next(iter(s.labels.values()), None): s.value for s in family.samples if s.name in names}


def total(metric) -> float:
    """A counter's total over its labels, or a gauge's value."""
    return sum(by_label(metric).values())


def observed(histogram, suffix: str = "_count") -> float:
    """A histogram's observation count, or with suffix "_sum" their sum."""
    (family,) = histogram.collect()
    return next(s.value for s in family.samples if s.name == family.name + suffix)


def asyncio_test(test):
    """No async plugin in the dev group, so each test drives its own loop."""

    @wraps(test)
    def wrapper(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return wrapper


@pytest.fixture(autouse=True)
def _reset_sequence_globals():
    """What Sequence takes from Config, plus the id counter, is class state."""
    saved = (Sequence.block_size, Sequence.enable_prefix_caching, Sequence.hash_algo, Sequence.counter)
    Sequence.counter = count()  # so seq ids are deterministic per test
    yield
    (Sequence.block_size, Sequence.enable_prefix_caching, Sequence.hash_algo, Sequence.counter) = saved


@pytest.fixture
def make_engine():
    def _make(eos_after: dict[int, int] | None = None, **overrides) -> FakeEngine:
        config = FakeConfig(**overrides)
        Sequence.counter = count()  # each engine gets its own seq ids, so two engines compare like-for-like
        Sequence.block_size = config.kvcache_block_size
        Sequence.enable_prefix_caching = config.enable_prefix_caching
        Sequence.hash_algo = config.prefix_caching_hash_algo
        return FakeEngine(config, FakeModelRunner(eos_after))

    return _make
