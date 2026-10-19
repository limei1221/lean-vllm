"""The KV connector's shared pieces: the rank aggregator, the config flag and request params; and, given NIXL and
two GPUs, a real read between two processes."""

import multiprocessing as mp
import socket
from dataclasses import dataclass
from importlib.util import find_spec
from time import monotonic, sleep

import pytest
import torch

from lean_vllm.kv_transfer import (
    KVConnectorMetadata,
    KVConnectorOutput,
    KVOutputAggregator,
    KVTransferConfig,
    KVTransferStats,
    ReqToRecv,
    check_kv_transfer_params,
    parse_kv_transfer_config,
)
from lean_vllm.kv_transfer.nixl_connector import NixlConnectorWorker

NUM_BLOCKS, BLOCK = 12, 4


@dataclass
class WorkerConfig:
    kv_transfer: KVTransferConfig
    tensor_parallel_size: int = 1
    kvcache_block_size: int = BLOCK
    kv_cache_layout: str = "HND"


def make_caches(seed: int) -> list[torch.Tensor]:
    """A key/value layer, HND, and an MLA latent layer, on the last GPU."""
    generator = torch.Generator().manual_seed(seed)
    caches = [
        torch.randn(2, NUM_BLOCKS, 2, BLOCK, 8, generator=generator).to(torch.bfloat16),
        torch.randn(1, NUM_BLOCKS, BLOCK, 6, generator=generator).to(torch.bfloat16),
    ]
    return [cache.cuda() for cache in caches]


def _producer(conn, kv_port: int):
    torch.cuda.set_device(0)
    kv_transfer = KVTransferConfig(kv_role="kv_producer", kv_port=kv_port, engine_id="engine")
    worker = NixlConnectorWorker(WorkerConfig(kv_transfer), 0)
    worker.register_kv_caches(make_caches(0))
    worker.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [7, 2, 9]}))
    conn.send("ready")
    while not worker.get_finished().finished_sending:
        sleep(0.01)
    worker.shutdown()


@pytest.mark.skipif(find_spec("nixl") is None or torch.cuda.device_count() < 2, reason="needs NIXL and two GPUs")
def test_a_real_nixl_read_between_two_processes():
    """A producer process on GPU 0, read by this one on GPU 1 through a real agent."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        kv_port = s.getsockname()[1]
    conn, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=_producer, args=(child, kv_port))
    process.start()
    try:
        assert conn.poll(60) and conn.recv() == "ready"
        torch.cuda.set_device(1)
        consumer = NixlConnectorWorker(WorkerConfig(KVTransferConfig(kv_role="kv_consumer")), 0)
        consumer.register_kv_caches(make_caches(1))
        req = ReqToRecv([0, 5, 3], [7, 2, 9], "p-1", "engine", "127.0.0.1", kv_port)
        consumer.start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": req}))
        deadline = monotonic() + 60
        while not (output := consumer.get_finished()) and monotonic() < deadline:
            sleep(0.01)
        assert output.finished_recving == {"d-1"}
        for theirs, ours in zip(make_caches(0), consumer.kv_caches):
            assert torch.equal(ours[:, [0, 5, 3]], theirs[:, [7, 2, 9]].to(ours.device))
        consumer.shutdown()
    finally:
        process.join(timeout=60)
        if process.is_alive():
            process.kill()


class TestAggregator:
    def test_a_transfer_finishes_once_every_rank_reports_it(self):
        aggregator = KVOutputAggregator(world_size=2)
        assert not aggregator.aggregate([KVConnectorOutput(finished_recving={"a"}), KVConnectorOutput()])
        result = aggregator.aggregate([KVConnectorOutput(), KVConnectorOutput(finished_recving={"a"})])
        assert result == KVConnectorOutput(finished_recving={"a"})

    def test_one_rank_failing_fails_the_load(self):
        aggregator = KVOutputAggregator(world_size=2)
        result = aggregator.aggregate(
            [KVConnectorOutput(failed_recving={"a"}), KVConnectorOutput(finished_recving={"a"})]
        )
        assert result == KVConnectorOutput(failed_recving={"a"})

    def test_sends_count_the_same_way(self):
        aggregator = KVOutputAggregator(world_size=2)
        assert not aggregator.aggregate([KVConnectorOutput(finished_sending={"a"}), KVConnectorOutput()])
        assert aggregator.aggregate(
            [KVConnectorOutput(), KVConnectorOutput(finished_sending={"a"})]
        ).finished_sending == {"a"}

    def test_a_load_s_stats_take_the_slowest_rank_and_every_rank_s_bytes(self):
        aggregator = KVOutputAggregator(world_size=2)
        result = aggregator.aggregate(
            [
                KVConnectorOutput(finished_recving={"a"}, recv_stats={"a": KVTransferStats(0.1, 100)}),
                KVConnectorOutput(finished_recving={"a"}, recv_stats={"a": KVTransferStats(0.3, 100)}),
            ]
        )
        assert result.recv_stats == {"a": KVTransferStats(0.3, 200)}

    def test_a_failed_load_has_no_stats(self):
        aggregator = KVOutputAggregator(world_size=2)
        result = aggregator.aggregate(
            [
                KVConnectorOutput(failed_recving={"a"}),
                KVConnectorOutput(finished_recving={"a"}, recv_stats={"a": KVTransferStats(0.3, 100)}),
            ]
        )
        assert result.failed_recving == {"a"} and result.recv_stats == {}

    def test_an_expiry_on_any_rank_counts_once(self):
        aggregator = KVOutputAggregator(world_size=2)
        aggregator.aggregate([KVConnectorOutput(finished_sending={"a"}, expired_sending={"a"}), KVConnectorOutput()])
        result = aggregator.aggregate([KVConnectorOutput(), KVConnectorOutput(finished_sending={"a"})])
        assert result.expired_sending == {"a"}


class TestConfig:
    def test_the_json_flag_parses(self):
        kv_transfer = parse_kv_transfer_config('{"kv_role": "kv_consumer", "kv_port": 15000}')
        assert (kv_transfer.kv_connector, kv_transfer.kv_role, kv_transfer.kv_port) == (
            "NixlConnector",
            "kv_consumer",
            15000,
        )
        assert kv_transfer.is_consumer and not kv_transfer.is_producer

    @pytest.mark.parametrize(
        "value, message",
        [
            ('{"kv_role": "decoder"}', "kv_role\n  Input should be 'kv_producer'"),
            ('{"kv_connector": "MooncakeConnector"}', "unknown kv_connector"),
            ('{"kv_buffer_size": 1}', "kv_buffer_size\n  Unexpected keyword argument"),
        ],
    )
    def test_a_bad_flag_is_named(self, value, message):
        with pytest.raises(ValueError, match=message):
            parse_kv_transfer_config(value)

    def test_each_engine_gets_its_own_id(self):
        assert parse_kv_transfer_config("{}").engine_id != parse_kv_transfer_config("{}").engine_id


class TestRequestParams:
    BOTH = KVTransferConfig(kv_role="kv_both")
    PULL = dict(
        do_remote_prefill=True,
        remote_block_ids=[1],
        remote_request_id="p",
        remote_engine_id="e",
        remote_host="h",
        remote_port=1,
    )

    def test_well_formed_params_pass(self):
        assert check_kv_transfer_params({"do_remote_decode": True}, self.BOTH) is None
        assert check_kv_transfer_params(self.PULL, self.BOTH) is None

    @pytest.mark.parametrize(
        "params, kv_role, message",
        [
            ({"do_remote_decode": True}, None, "needs an engine started with kv_transfer_config"),
            ({"do_remote_decode": True}, "kv_consumer", "do_remote_decode needs kv_role"),
            (PULL, "kv_producer", "do_remote_prefill needs kv_role"),
            (PULL | {"remote_port": "1"}, "kv_both", "remote_port: Input should be a valid integer"),
            (PULL | {"remote_block_ids": ["x"]}, "kv_both", "remote_block_ids.0: Input should be a valid integer"),
            ({k: v for k, v in PULL.items() if k != "remote_host"}, "kv_both", "remote_host: Field required"),
        ],
    )
    def test_what_the_engine_cannot_honour_is_named(self, params, kv_role, message):
        kv_transfer = KVTransferConfig(kv_role=kv_role) if kv_role else None
        assert message in check_kv_transfer_params(params, kv_transfer)
