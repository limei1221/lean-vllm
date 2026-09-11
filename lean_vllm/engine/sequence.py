import hashlib
import pickle
import xxhash
from copy import copy
from enum import Enum, auto
from itertools import count
from time import perf_counter

from lean_vllm.engine.output import RequestMetrics
from lean_vllm.sampling_params import SamplingParams


def _serialize(token_ids: tuple[int, ...], prefix: int) -> bytes:
    """Pickle, as vLLM does. It is stable within a Python version but not
    promised across them, which is why vLLM also offers CBOR variants."""
    return pickle.dumps((prefix, token_ids), protocol=pickle.HIGHEST_PROTOCOL)


def sha256_hash(token_ids: tuple[int, ...], prefix: int) -> int:
    """The default, and vLLM's. A collision here serves one tenant another's
    tokens, so it is ruled out rather than made unlikely."""
    return int.from_bytes(hashlib.sha256(_serialize(token_ids, prefix)).digest(), "big")


def xxhash_hash(token_ids: tuple[int, ...], prefix: int) -> int:
    """Faster, not cryptographic. Worth it only where every request is trusted."""
    return int.from_bytes(xxhash.xxh128(_serialize(token_ids, prefix)).digest(), "big")


HASH_ALGOS = {"sha256": sha256_hash, "xxhash": xxhash_hash}


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # Set once from Config, since a Sequence is built in places that carry no
    # config: the engine, the profiling warmup, and the spawned TP workers.
    block_size = 256
    enable_prefix_caching = True
    hash_algo = "sha256"
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams(), request_id: str | None = None):
        self.seq_id = next(Sequence.counter)
        self.request_id = request_id if request_id is not None else f"req-{self.seq_id}"
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.block_hashes: list[int] = []    # chained, one per full block, filled on demand
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.stop_token_ids = sampling_params.stop_token_ids
        self.skip_special_tokens = sampling_params.skip_special_tokens
        self.priority = sampling_params.priority
        self.finish_reason: str | None = None
        self.num_preemptions = 0
        self.arrival_time = perf_counter()
        self.first_scheduled_time: float | None = None
        self.first_token_time: float | None = None
        self.finish_time: float | None = None
        self._extend_block_hashes()

    def metrics(self) -> RequestMetrics:
        return RequestMetrics(
            arrival_time=self.arrival_time,
            num_prompt_tokens=self.num_prompt_tokens,
            num_completion_tokens=self.num_completion_tokens,
            num_preemptions=self.num_preemptions,
            first_scheduled_time=self.first_scheduled_time,
            first_token_time=self.first_token_time,
            finish_time=self.finish_time,
        )

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def _extend_block_hashes(self):
        """Hash every block that has just become full, chaining on the one before.

        A full block never takes another token, so its hash is final and is
        computed exactly once, here, as the tokens arrive. The scheduler queries
        the cache again on every step a request spends at the head of the
        waiting queue, and rehashing the prompt each time is the whole cost.
        """
        if not self.enable_prefix_caching:
            return
        algo = HASH_ALGOS[self.hash_algo]
        for i in range(len(self.block_hashes), self.num_tokens // self.block_size):
            prefix = self.block_hashes[-1] if self.block_hashes else -1
            self.block_hashes.append(algo(tuple(self.block(i)), prefix))

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self._extend_block_hashes()

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.is_prefill, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.is_prefill, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
