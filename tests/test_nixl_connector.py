"""NixlConnector's worker halves, prefill and decode ranks in one process on CPU caches, over a fake NIXL agent.

The fake reads by address with memmove, so every descriptor the workers build is checked against real memory: a
wrong offset or length lands the wrong bytes, and a read outside a registered tensor raises.
"""

import ctypes
import socket
import threading
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from time import monotonic, sleep

import numpy as np
import pytest
import torch

from lean_vllm.kv_transfer import (
    KVConnectorMetadata,
    KVConnectorOutput,
    KVOutputAggregator,
    KVTransferConfig,
    ReqToRecv,
    nixl_connector,
)
from lean_vllm.kv_transfer.nixl_connector import NixlConnectorWorker, Share, head_shares

NUM_BLOCKS, BLOCK, HEAD_DIM, LATENT_DIM = 12, 4, 8, 6
NUM_HEADS = 12  # splits over 1, 2, 3, 4 and 6 ranks


class FakeAgent:
    """The NIXL agent calls NixlConnectorWorker makes, moving host memory within this process."""

    agents: dict[str, "FakeAgent"] = {}
    fail_reads: str | None = None  # "posted" or "in flight"

    def __init__(self, name: str):
        self.name = name
        self.registered: list[tuple[int, int]] = []
        self.remotes: set[str] = set()
        self.xfers: dict[int, list] = {}
        self.notifs: dict[str, list[bytes]] = defaultdict(list)
        self.lock = threading.Lock()
        self.handles = count()
        FakeAgent.agents[name] = self

    def get_reg_descs(self, memory, mem_type):
        assert mem_type == "DRAM"
        return [(addr, length) for addr, length, _, _ in memory]

    def register_memory(self, descs):
        self.registered += descs

    def deregister_memory(self, descs):
        self.registered = [desc for desc in self.registered if desc not in descs]

    def get_agent_metadata(self) -> bytes:
        return self.name.encode()

    def add_remote_agent(self, metadata: bytes) -> str:
        self.remotes.add(metadata.decode())
        return metadata.decode()

    def remove_remote_agent(self, name: str):
        self.remotes.discard(name)

    def get_xfer_descs(self, rows, mem_type):
        return np.asarray(rows)

    def initialize_xfer(self, op, local, remote, agent_name, notif):
        assert op == "READ" and agent_name in self.remotes
        assert len(local) == len(remote) and (local[:, 1] == remote[:, 1]).all()
        peer = FakeAgent.agents[agent_name]
        for agent, rows in ((self, local), (peer, remote)):
            for addr, length, _ in rows.tolist():
                assert any(lo <= addr and addr + length <= lo + n for lo, n in agent.registered), "unregistered memory"
        handle = next(self.handles)
        self.xfers[handle] = [local, remote, peer, notif, 1]  # done on the second poll
        return handle

    def transfer(self, handle) -> str:
        local, remote, _, _, _ = self.xfers[handle]
        if FakeAgent.fail_reads == "posted":
            return "ERR"
        for (dst, length, _), (src, _, _) in zip(local.tolist(), remote.tolist()):
            ctypes.memmove(dst, src, length)
        return "PROC"

    def check_xfer_state(self, handle) -> str:
        xfer = self.xfers[handle]
        if FakeAgent.fail_reads:
            return "ERR"
        if xfer[4]:
            xfer[4] -= 1
            return "PROC"
        xfer[2].deliver(self.name, xfer[3])  # NIXL notifies once the data has landed
        return "DONE"

    def release_xfer_handle(self, handle):
        del self.xfers[handle]

    def send_notif(self, agent_name: str, message: bytes):
        FakeAgent.agents[agent_name].deliver(self.name, message)

    def deliver(self, sender: str, message: bytes):
        with self.lock:
            self.notifs[sender].append(message)

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        with self.lock:
            notifs, self.notifs = dict(self.notifs), defaultdict(list)
        return notifs


@pytest.fixture(autouse=True)
def fake_nixl(monkeypatch):
    monkeypatch.setattr(nixl_connector, "make_agent", FakeAgent)
    monkeypatch.setattr(FakeAgent, "agents", {})
    monkeypatch.setattr(FakeAgent, "fail_reads", None)


@dataclass
class WorkerConfig:
    kv_transfer: KVTransferConfig
    tensor_parallel_size: int = 1
    kvcache_block_size: int = BLOCK
    kv_cache_layout: str = "HND"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def full_cache(seed: int, mla: bool) -> torch.Tensor:
    """Every rank's heads together, as a TP-1 rank would hold them, HND."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, NUM_BLOCKS, BLOCK, LATENT_DIM) if mla else (2, NUM_BLOCKS, NUM_HEADS, BLOCK, HEAD_DIM)
    return torch.randn(*shape, generator=generator).to(torch.bfloat16)


def rank_cache(full: torch.Tensor, rank: int, tp_size: int) -> torch.Tensor:
    """One rank's slice of the heads; latents are whole on every rank."""
    if full.dim() == 4:
        return full.clone()
    heads = NUM_HEADS // tp_size
    return full[:, :, rank * heads : (rank + 1) * heads].clone()


@pytest.fixture
def make_side():
    """Every rank of one instance, each registering its slice of a full cache."""
    workers: list[NixlConnectorWorker] = []

    def _make(role: str, tp_size: int, full: torch.Tensor, port: int = 0, engine_id: str = "prefill"):
        port = port or free_port()
        side = []
        for rank in range(tp_size):
            kv_transfer = KVTransferConfig(
                kv_connector="NixlConnector", kv_role=role, kv_port=port, engine_id=engine_id
            )
            worker = NixlConnectorWorker(WorkerConfig(kv_transfer, tensor_parallel_size=tp_size), rank)
            worker.register_kv_caches([rank_cache(full, rank, tp_size)])
            side.append(worker)
        workers.extend(side)
        return side

    yield _make
    for worker in workers:
        worker.shutdown()


def recv(producers: list[NixlConnectorWorker], local: list[int], remote: list[int], **overrides) -> ReqToRecv:
    kv = producers[0].kv_transfer
    fields = dict(
        local_block_ids=local,
        remote_block_ids=remote,
        remote_request_id="p-1",
        remote_engine_id=kv.engine_id,
        remote_host=kv.kv_ip,
        remote_port=kv.kv_port,
        remote_tp_size=len(producers),
    )
    return ReqToRecv(**(fields | overrides))


def wait_for(worker: NixlConnectorWorker, timeout: float = 10.0) -> KVConnectorOutput:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if output := worker.get_finished():
            return output
        sleep(0.005)
    raise AssertionError("no transfer finished")


def test_each_rank_s_shares_are_the_overlap_of_its_heads_with_each_producer_rank_s():
    assert head_shares(0, 1, 1, 8) == [Share(0, 0, 0, 8)]
    assert head_shares(3, 4, 2, 8) == [Share(1, 2, 0, 2)]  # more decode ranks: a slice of one producer rank's
    assert head_shares(1, 2, 4, 8) == [Share(2, 0, 0, 2), Share(3, 0, 2, 2)]  # fewer: whole ones, side by side
    # Neither size divides the other: rank 1 of 6 holds heads 2 and 3, which ranks 0 and 1 of 4 split.
    assert head_shares(1, 6, 4, 12) == [Share(0, 2, 0, 1), Share(1, 0, 1, 1)]


TP_PAIRS = [(1, 1), (2, 2), (1, 2), (2, 1), (1, 4), (4, 1), (2, 4), (4, 2), (4, 6), (6, 4), (2, 3)]


@pytest.mark.parametrize("mla", [False, True], ids=["kv", "latents"])
@pytest.mark.parametrize("prefill_tp, decode_tp", TP_PAIRS)
def test_each_decode_rank_gets_its_heads_whatever_the_two_tensor_parallel_sizes(make_side, prefill_tp, decode_tp, mla):
    prefill, decode = full_cache(0, mla), full_cache(1, mla)
    producers = make_side("kv_producer", prefill_tp, prefill)
    consumers = make_side("kv_consumer", decode_tp, decode)
    for producer in producers:
        producer.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [7, 2, 9]}))
    for consumer in consumers:
        consumer.start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producers, [0, 5, 3], [7, 2, 9])}))

    aggregator = KVOutputAggregator(decode_tp)
    assert aggregator.aggregate([wait_for(consumer) for consumer in consumers]).finished_recving == {"d-1"}
    expected = decode.clone()
    expected[:, [0, 5, 3]] = prefill[:, [7, 2, 9]]
    for rank, consumer in enumerate(consumers):
        assert torch.equal(consumer.kv_caches[0], rank_cache(expected, rank, decode_tp)), f"decode rank {rank}"
    # Every producer rank frees the blocks, including those of latents no decode rank read.
    aggregator = KVOutputAggregator(prefill_tp)
    assert aggregator.aggregate([wait_for(producer) for producer in producers]).finished_sending == {"p-1"}


def test_a_producer_rank_waits_for_every_decode_rank_that_reads_it(make_side):
    producers = make_side("kv_producer", 1, full_cache(0, False))
    consumers = make_side("kv_consumer", 2, full_cache(1, False))
    producers[0].start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [4]}))
    consumers[0].start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producers, [0], [4])}))
    assert wait_for(consumers[0]).finished_recving == {"d-1"}
    sleep(0.05)
    assert not producers[0].get_finished(), "freed while decode rank 1 has yet to read"
    consumers[1].start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producers, [0], [4])}))
    assert wait_for(consumers[1]).finished_recving == {"d-1"}
    assert wait_for(producers[0]) == KVConnectorOutput(finished_sending={"p-1"})


def test_a_read_before_the_hand_over_frees_the_blocks_once_it_comes(make_side):
    """A read is one-sided, so it may finish before the producer's engine has handed the request over."""
    producers = make_side("kv_producer", 1, full_cache(0, False))
    consumers = make_side("kv_consumer", 1, full_cache(1, False))
    consumers[0].start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producers, [0], [4])}))
    assert wait_for(consumers[0]).finished_recving == {"d-1"}
    sleep(0.05)
    producers[0].start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [4]}))
    assert producers[0].get_finished() == KVConnectorOutput(finished_sending={"p-1"})


def test_only_the_blocks_not_cached_here_are_read(make_side):
    prefill, decode = full_cache(0, False), full_cache(1, False)
    producers = make_side("kv_producer", 2, prefill)
    consumers = make_side("kv_consumer", 1, decode)
    producers[0].start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [1, 2, 3]}))
    producers[1].start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [1, 2, 3]}))
    consumers[0].start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producers, [8], [3])}))
    stats = wait_for(consumers[0]).recv_stats["d-1"]
    assert stats.num_bytes == 2 * NUM_HEADS * BLOCK * HEAD_DIM * 2  # one block of keys and values, bf16
    expected = decode.clone()
    expected[:, 8] = prefill[:, 3]
    assert torch.equal(consumers[0].kv_caches[0], expected)


def test_a_full_prefix_hit_reads_nothing_but_still_frees_the_producer_s_blocks(make_side):
    producers = make_side("kv_producer", 1, full_cache(0, False))
    consumers = make_side("kv_consumer", 1, full_cache(1, False))
    before = consumers[0].kv_caches[0].clone()
    producers[0].start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [4]}))
    consumers[0].start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producers, [], [])}))
    assert wait_for(consumers[0]).finished_recving == {"d-1"}
    assert wait_for(producers[0]).finished_sending == {"p-1"}
    assert torch.equal(consumers[0].kv_caches[0], before)


def test_an_absent_producer_fails_the_load(make_side, monkeypatch):
    monkeypatch.setattr(nixl_connector, "HANDSHAKE_TIMEOUT", 0.2)
    consumers = make_side("kv_consumer", 1, full_cache(1, False))
    req = ReqToRecv([0], [4], "p-1", "prefill", "127.0.0.1", free_port())
    consumers[0].start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": req}))
    assert wait_for(consumers[0]) == KVConnectorOutput(failed_recving={"d-1"})


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(remote_engine_id="before-a-restart"), "is engine prefill, not before-a-restart"),
        (dict(remote_tp_size=2), "runs tp_size 1, not 2"),
        (dict(remote_block_ids=[NUM_BLOCKS]), "past the producer's"),
    ],
)
def test_a_producer_that_does_not_match_the_request_fails_the_load(make_side, caplog, overrides, message):
    producers = make_side("kv_producer", 1, full_cache(0, False))
    consumers = make_side("kv_consumer", 1, full_cache(1, False))
    consumers[0].start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producers, [0], [4], **overrides)}))
    assert wait_for(consumers[0]) == KVConnectorOutput(failed_recving={"d-1"})
    assert message in caplog.text


def test_a_producer_of_another_model_fails_the_load(make_side, caplog):
    producers = make_side("kv_producer", 1, full_cache(0, True))
    consumers = make_side("kv_consumer", 1, full_cache(1, False))
    consumers[0].start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producers, [0], [4])}))
    assert wait_for(consumers[0]) == KVConnectorOutput(failed_recving={"d-1"})
    assert "serve the same model, dtype and block size" in caplog.text


def test_a_read_refused_when_posted_fails_the_load_and_tells_the_producer(make_side):
    """It never reached the producer, so the consumer notifies it, and the blocks go now, not at the timeout."""
    producers = make_side("kv_producer", 1, full_cache(0, False))
    consumers = make_side("kv_consumer", 1, full_cache(1, False))
    FakeAgent.fail_reads = "posted"
    producers[0].start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [4]}))
    consumers[0].start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producers, [0], [4])}))
    assert wait_for(consumers[0]) == KVConnectorOutput(failed_recving={"d-1"})
    assert wait_for(producers[0]) == KVConnectorOutput(finished_sending={"p-1"})


def test_a_read_failing_in_flight_fails_the_load_and_leaves_the_blocks_to_expire(make_side, monkeypatch):
    """Its notification may yet arrive, so the consumer sends none; one more would free blocks still being read."""
    producers = make_side("kv_producer", 1, full_cache(0, False))
    consumers = make_side("kv_consumer", 1, full_cache(1, False))
    FakeAgent.fail_reads = "in flight"
    producers[0].start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [4]}))
    consumers[0].start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producers, [0], [4])}))
    assert wait_for(consumers[0]) == KVConnectorOutput(failed_recving={"d-1"})
    sleep(0.05)
    assert not producers[0].get_finished()
    monkeypatch.setenv("LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT", "0")
    producers[0].start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [4]}))  # handed over again, to expire now
    sleep(0.01)
    assert producers[0].get_finished() == KVConnectorOutput(finished_sending={"p-1"}, expired_sending={"p-1"})


def test_a_second_request_reuses_the_handshake(make_side):
    producers = make_side("kv_producer", 1, full_cache(0, False))
    consumers = make_side("kv_consumer", 1, full_cache(1, False))
    for request_id, block in (("p-1", 4), ("p-2", 5)):
        producers[0].start_load_kv(KVConnectorMetadata(reqs_to_send={request_id: [block]}))
        req = recv(producers, [block], [block], remote_request_id=request_id)
        consumers[0].start_load_kv(KVConnectorMetadata(reqs_to_recv={request_id: req}))
        assert wait_for(consumers[0]).finished_recving == {request_id}
    assert len(consumers[0]._remotes) == 1 and len(consumers[0].agent.remotes) == 1


def test_a_taken_port_is_named(make_side):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        with pytest.raises(RuntimeError, match="pick another kv_port"):
            make_side("kv_producer", 1, full_cache(0, False), port=taken.getsockname()[1])


def test_a_consumer_serves_nothing(make_side):
    assert make_side("kv_consumer", 1, full_cache(1, False))[0]._server is None
