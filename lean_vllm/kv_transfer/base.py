"""The KV connector interface follows vLLM's KVConnectorBase_V1, with a scheduler in the engine and a worker per rank.

The scheduler decides which requests load or share KV and keeps their blocks allocated during transfers. Each step,
it sends KVConnectorMetadata to the workers, which report transfers completed since their previous report.
"""

import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(config=ConfigDict(extra="forbid"), slots=True)
class KVTransferConfig:
    """Settings for --kv-transfer-config. Producers serve prefilled KV, consumers load it, and kv_both allows both."""

    kv_connector: str = "NixlConnector"
    kv_role: Literal["kv_producer", "kv_consumer", "kv_both"] = "kv_both"
    kv_ip: str = "127.0.0.1"  # address the producer listens on and gives consumers to connect to
    kv_port: int = 14579  # rank r listens on kv_port + r
    engine_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def parse(cls, value: "str | dict | KVTransferConfig") -> "KVTransferConfig":
        if isinstance(value, KVTransferConfig):
            return value
        adapter = TypeAdapter(cls)
        return adapter.validate_json(value) if isinstance(value, str) else adapter.validate_python(value)

    @property
    def is_producer(self) -> bool:
        return self.kv_role in ("kv_producer", "kv_both")

    @property
    def is_consumer(self) -> bool:
        return self.kv_role in ("kv_consumer", "kv_both")


class _RemotePrefill(BaseModel):
    """Fields a decode instance needs to load remote KV blocks, returned by the producer's request_finished."""

    model_config = ConfigDict(strict=True)  # reject incorrect field types, such as a string for remote_port

    remote_block_ids: list[int]
    remote_request_id: str
    remote_engine_id: str
    remote_host: str
    remote_port: int
    tp_size: int = 1


def check_kv_transfer_params(params, kv_transfer: KVTransferConfig | None) -> str | None:
    """Return a validation error for the request's kv_transfer_params, or None if these checks pass."""
    if not isinstance(params, dict):
        return "kv_transfer_params must be an object"
    if kv_transfer is None:
        return "kv_transfer_params needs an engine started with kv_transfer_config"
    if params.get("do_remote_decode") and not kv_transfer.is_producer:
        return f"do_remote_decode needs kv_role kv_producer or kv_both, not {kv_transfer.kv_role}"
    if params.get("do_remote_prefill"):
        if not kv_transfer.is_consumer:
            return f"do_remote_prefill needs kv_role kv_consumer or kv_both, not {kv_transfer.kv_role}"
        try:
            _RemotePrefill.model_validate(params)
        except ValidationError as error:
            first = error.errors()[0]
            return f"do_remote_prefill needs {'.'.join(map(str, first['loc']))}: {first['msg']}"
    return None


@dataclass(slots=True)
class ReqToRecv:
    """Remote KV blocks to load into local blocks. Block IDs at the same index refer to the same prompt tokens."""

    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_request_id: str
    remote_engine_id: str
    remote_host: str
    remote_port: int
    remote_tp_size: int = 1


@dataclass(slots=True)
class KVConnectorMetadata:
    """Transfer instructions sent to every worker for one step."""

    reqs_to_recv: dict[str, ReqToRecv] = field(default_factory=dict)
    reqs_to_send: dict[str, list[int]] = field(default_factory=dict)  # blocks held for remote decode requests to read

    def __bool__(self):
        return bool(self.reqs_to_recv or self.reqs_to_send)


@dataclass(slots=True)
class KVTransferStats:
    """Elapsed time and bytes read for a completed KV load, including setup time."""

    seconds: float
    num_bytes: int

    def merge(self, other: "KVTransferStats") -> "KVTransferStats":
        """Combine parallel loads across ranks using the longest elapsed time and the total bytes read."""
        return KVTransferStats(max(self.seconds, other.seconds), self.num_bytes + other.num_bytes)


@dataclass(slots=True)
class KVConnectorOutput:
    """Transfer results since the last report, from one worker or aggregated across ranks.

    Failed loads require local recomputation. Finished sends include blocks released after a timeout.
    """

    finished_sending: set[str] = field(default_factory=set)
    finished_recving: set[str] = field(default_factory=set)
    failed_recving: set[str] = field(default_factory=set)
    expired_sending: set[str] = field(default_factory=set)  # finished_sending requests that timed out on any rank
    recv_stats: dict[str, KVTransferStats] = field(default_factory=dict)  # stats for requests in finished_recving

    def __bool__(self):
        return bool(self.finished_sending or self.finished_recving or self.failed_recving)


class KVOutputAggregator:
    """Report a transfer's result once every rank has reported it. A load fails if any rank reports failure."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._sent: Counter[str] = Counter()
        self._recved: Counter[str] = Counter()
        self._failed: set[str] = set()
        self._expired: set[str] = set()
        self._stats: dict[str, KVTransferStats] = {}

    def aggregate(self, outputs: list[KVConnectorOutput]) -> KVConnectorOutput:
        result = KVConnectorOutput()
        for output in outputs:
            self._failed |= output.failed_recving
            self._expired |= output.expired_sending
            for request_id, stats in output.recv_stats.items():
                seen = self._stats.get(request_id)
                self._stats[request_id] = stats if seen is None else seen.merge(stats)
            for request_id in output.finished_sending:
                self._sent[request_id] += 1
                if self._sent[request_id] == self.world_size:
                    del self._sent[request_id]
                    result.finished_sending.add(request_id)
                    if request_id in self._expired:
                        self._expired.discard(request_id)
                        result.expired_sending.add(request_id)
            for request_id in output.finished_recving | output.failed_recving:
                self._recved[request_id] += 1
                if self._recved[request_id] == self.world_size:
                    del self._recved[request_id]
                    failed = request_id in self._failed
                    self._failed.discard(request_id)
                    stats = self._stats.pop(request_id, None)
                    if failed:
                        result.failed_recving.add(request_id)
                    else:
                        result.finished_recving.add(request_id)
                        if stats is not None:
                            result.recv_stats[request_id] = stats
        return result


class KVConnectorScheduler(ABC):
    """The scheduler side of the connector. Every method runs on the engine thread, between steps."""

    @abstractmethod
    def get_num_new_matched_tokens(self, seq, num_computed_tokens: int) -> tuple[int, bool]:
        """Return how many prompt tokens beyond num_computed_tokens have remote KV, and whether loading is async."""

    @abstractmethod
    def update_state_after_alloc(self, seq, num_external_tokens: int):
        """Queue a load into seq's allocated prompt blocks, skipping blocks already cached locally."""

    @abstractmethod
    def build_connector_meta(self) -> KVConnectorMetadata:
        """Return this step's transfer instructions for the workers and clear the scheduler's pending instructions."""

    @abstractmethod
    def request_finished(self, seq) -> tuple[bool, dict | None]:
        """Return whether to hold seq's blocks until all workers report finished_sending, and any kv_transfer_params."""


class KVConnectorWorker(ABC):
    """The worker side of the connector: transfers blocks between this rank's KV cache and another instance's."""

    @abstractmethod
    def register_kv_caches(self, kv_caches: list):
        """Register each layer's cache in model order, with blocks along dimension 1. Called once, before any step."""

    @abstractmethod
    def start_load_kv(self, metadata: KVConnectorMetadata):
        """Start this step's loads and track blocks held for remote reads. Must not block on the network."""

    @abstractmethod
    def get_finished(self) -> KVConnectorOutput:
        """Return transfer results on this rank since the last call."""

    def shutdown(self):
        pass
