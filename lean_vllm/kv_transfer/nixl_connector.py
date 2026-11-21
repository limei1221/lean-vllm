"""NixlConnector lets a decode instance load a producer's prompt KV blocks through NIXL's UCX backend.

A producer holds the prompt blocks of a request configured for remote decode when it finishes with length or stop,
and returns their location in kv_transfer_params. Each rank registers its KV cache with a NIXL agent. A producer rank
serves its agent metadata and cache addresses from a ROUTER socket on kv_port + rank. A consumer rank caches a handshake
with each producer rank it uses, then issues one-sided READs directly into its own blocks. Reads include completion
notifications. A producer rank reports a request's blocks ready for release after all expected notifications arrive
or LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT expires. The engine frees the blocks once all producer ranks have reported them.

The two sides may use different tensor-parallel sizes. HND caches store each head's tokens contiguously within a block.
A consumer rank reads only the heads it needs from each producer rank. MLA latents are replicated across ranks, so a
consumer reads whole latent blocks from the first producer rank in its shares and omits latents from the other reads.
If a share has no data to read, the consumer sends its notification directly.
"""

import json
import logging
import math
import os
import queue
import threading
import uuid
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, NamedTuple

import numpy as np
import torch
import zmq

from lean_vllm import envs
from lean_vllm.kv_transfer.base import (
    KVConnectorMetadata,
    KVConnectorOutput,
    KVConnectorScheduler,
    KVConnectorWorker,
    KVTransferStats,
    ReqToRecv,
)

logger = logging.getLogger(__name__)

HANDSHAKE_TIMEOUT = 5.0  # bound the handshake wait when a producer does not respond
IDLE_WAIT = 0.01  # queue wait in seconds when no reads are pending
BUSY_WAIT = 0.001  # shorter queue wait while polling reads in flight


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def make_agent(name: str):
    """Create a NIXL agent using the UCX backend."""
    # Apply vLLM's UCX memory-leak workaround before importing NIXL, unless the user set a limit.
    os.environ.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as error:
        raise RuntimeError("NixlConnector needs NIXL: uv sync --extra nixl") from error
    return nixl_agent(name, nixl_agent_config(backends=["UCX"]))


class Share(NamedTuple):
    """The heads one consumer rank reads from one producer rank."""

    remote_rank: int
    remote_head: int  # first head index within the producer rank's cache
    local_head: int  # first destination head index within the consumer rank's cache
    num_heads: int


def head_shares(rank: int, tp_size: int, remote_tp_size: int, num_heads: int) -> list[Share]:
    """Return the head ranges this consumer rank reads from each producer rank.

    num_heads is the total head count, split evenly across the ranks on each side.
    """
    local, remote = num_heads // tp_size, num_heads // remote_tp_size
    start, end = rank * local, (rank + 1) * local
    shares = []
    for remote_rank in range(start // remote, -(-end // remote)):
        first, last = max(start, remote_rank * remote), min(end, (remote_rank + 1) * remote)
        shares.append(Share(remote_rank, first - remote_rank * remote, first - start, last - first))
    return shares


@dataclass(slots=True)
class Region:
    """Address and byte strides for one registered key, value or latent tensor."""

    addr: int  # address of block 0
    block_bytes: int  # byte stride between blocks
    head_bytes: int  # one head's tokens within a block; 0 for latents, which are read whole


def block_descs(
    regions: list[Region], block_ids: list[int], first_head: int, num_heads: int, latents: bool, device_id: int
) -> np.ndarray:
    """Build NIXL (address, length, device) rows in region order, then block_ids order.

    Key and value regions include num_heads starting at first_head in each block. Latent regions include whole blocks
    when latents is True and are skipped otherwise. Addresses and lengths are in bytes.
    """
    blocks = np.asarray(block_ids, dtype=np.uint64)
    parts = []
    for region in regions:
        if region.head_bytes:
            start, length = region.addr + first_head * region.head_bytes, num_heads * region.head_bytes
        elif latents:
            start, length = region.addr, region.block_bytes
        else:
            continue
        rows = np.empty((len(blocks), 3), dtype=np.uint64)
        rows[:, 0] = start + blocks * np.uint64(region.block_bytes)
        rows[:, 1] = length
        rows[:, 2] = device_id
        parts.append(rows)
    return np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.uint64)


class NixlConnectorScheduler(KVConnectorScheduler):
    def __init__(self, config):
        self.kv_transfer = config.kv_transfer
        self.block_size = config.kvcache_block_size
        self.tp_size = config.tensor_parallel_size
        self._reqs_to_recv: dict[str, ReqToRecv] = {}
        self._reqs_to_send: dict[str, list[int]] = {}

    def get_num_new_matched_tokens(self, seq, num_computed_tokens: int) -> tuple[int, bool]:
        params = seq.kv_transfer_params
        if not params or not params.get("do_remote_prefill"):
            return 0, False
        num_blocks = _cdiv(seq.num_prompt_tokens, self.block_size)
        if len(params["remote_block_ids"]) != num_blocks:
            problem = f"{len(params['remote_block_ids'])} remote blocks for a {num_blocks}-block prompt"
            logger.warning("prefilling %s here instead: %s", seq.request_id, problem)
            params["do_remote_prefill"] = False
            return 0, False
        # Different tensor-parallel sizes are supported: each consumer rank reads the producer ranks it needs.
        return seq.num_prompt_tokens - num_computed_tokens, True

    def update_state_after_alloc(self, seq, num_external_tokens: int):
        params = seq.kv_transfer_params
        params["do_remote_prefill"] = False  # load only once; recompute locally if the request is preempted
        num_local_blocks = seq.num_cached_tokens // self.block_size  # skip blocks already in the local prefix cache
        self._reqs_to_recv[seq.request_id] = ReqToRecv(
            local_block_ids=seq.block_table[num_local_blocks:],
            remote_block_ids=params["remote_block_ids"][num_local_blocks:],
            remote_request_id=params["remote_request_id"],
            remote_engine_id=params["remote_engine_id"],
            remote_host=params["remote_host"],
            remote_port=params["remote_port"],
            remote_tp_size=params.get("tp_size", 1),
        )

    def build_connector_meta(self) -> KVConnectorMetadata:
        metadata = KVConnectorMetadata(self._reqs_to_recv, self._reqs_to_send)
        self._reqs_to_recv, self._reqs_to_send = {}, {}
        return metadata

    def request_finished(self, seq) -> tuple[bool, dict | None]:
        params = seq.kv_transfer_params
        # Share prompt blocks only for remote decode requests that finished with length or stop, not an abort.
        if not params or not params.get("do_remote_decode") or seq.finish_reason not in ("length", "stop"):
            return False, None
        block_ids = seq.block_table[: _cdiv(seq.num_prompt_tokens, self.block_size)]
        self._reqs_to_send[seq.request_id] = block_ids
        kv = self.kv_transfer
        return True, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_request_id=seq.request_id,
            remote_engine_id=kv.engine_id,
            remote_block_ids=block_ids,
            remote_host=kv.kv_ip,
            remote_port=kv.kv_port,
            tp_size=self.tp_size,
        )


@dataclass(slots=True)
class _Remote:
    """Cached agent and cache information from a producer rank's handshake."""

    engine_id: str
    agent_name: str
    regions: list[Region]
    num_blocks: int
    mem_type: str
    device_id: int


@dataclass(slots=True)
class _Load:
    """Track one request's reads and the producer ranks still needing notifications."""

    req: ReqToRecv
    handles: list = field(default_factory=list)  # read handles awaiting completion checks
    unnotified: set[int] = field(default_factory=set)  # ranks needing a fallback notification if setup fails
    started: float = field(default_factory=monotonic)
    num_bytes: int = 0
    failed: bool = False


class NixlConnectorWorker(KVConnectorWorker):
    def __init__(self, config, rank: int):
        assert config.kv_cache_layout == "HND", "NixlConnector reads heads out of blocks, so needs HND caches"
        self.kv_transfer = config.kv_transfer
        self.rank = rank
        self.tp_size = config.tensor_parallel_size
        self.block_size = config.kvcache_block_size
        self.kv_caches: list[torch.Tensor] = []
        self.regions: list[Region] = []
        self.num_heads = 0  # total key heads across ranks; 0 when all layers use latents
        self.layout: dict = {}
        self.agent: Any = None  # a nixl_agent, once the caches are registered
        self._reg_descs = None
        self._lock = threading.Lock()  # protects _reqs_to_send, _notified and _finished across threads
        self._reqs_to_send: dict[str, float] = {}  # request ID -> deadline for consumer notifications
        # Notifications may arrive before the request is added to _reqs_to_send.
        self._notified: dict[str, list] = {}  # request ID -> [notification count, expected readers, first arrival time]
        self._finished = KVConnectorOutput()
        self._queue: queue.Queue = queue.Queue()
        self._context = zmq.Context()
        self._context.setsockopt(zmq.LINGER, 0)
        self._server: zmq.Socket | None = None
        self._remotes: dict[tuple[str, int], _Remote] = {}  # keyed by producer (host, port); transfer thread only
        self._loads: dict[str, _Load] = {}  # transfer thread only
        self._thread: threading.Thread | None = None

    def register_kv_caches(self, kv_caches: list[torch.Tensor]):
        self.kv_caches = kv_caches
        self.mem_type = "VRAM" if kv_caches[0].is_cuda else "DRAM"
        self.device_id = max(kv_caches[0].get_device(), 0)  # use device ID 0 for host memory instead of -1
        layers = []
        for cache in kv_caches:
            itemsize = cache.element_size()
            if cache.dim() == 5:  # keys and values, (2, block, head, token, dim)
                _, _, heads, tokens, dim = cache.shape
                self.num_heads = heads * self.tp_size
                layers.append(["kv", str(cache.dtype), self.num_heads, tokens, dim])
            else:  # latents, (1, block, token, dim)
                layers.append(["latent", str(cache.dtype), *cache.shape[2:]])
            for part in cache:
                head_bytes = part.stride(1) * itemsize if cache.dim() == 5 else 0
                self.regions.append(Region(part.data_ptr(), part.stride(0) * itemsize, head_bytes))
        # Compare total head counts so peers can match layouts even with different tensor-parallel sizes.
        self.layout = dict(block_size=self.block_size, layers=layers)

        kv = self.kv_transfer
        self.agent = make_agent(f"{kv.engine_id}-{self.rank}-{uuid.uuid4().hex[:8]}")
        memory = [(cache.data_ptr(), cache.numel() * cache.element_size(), self.device_id, "") for cache in kv_caches]
        self._reg_descs = self.agent.get_reg_descs(memory, self.mem_type)
        self.agent.register_memory(self._reg_descs)
        self.agent_metadata = self.agent.get_agent_metadata()
        if kv.is_producer:
            port = kv.kv_port + self.rank
            server = self._context.socket(zmq.ROUTER)
            try:
                server.bind(f"tcp://{kv.kv_ip}:{port}")
            except zmq.ZMQError as error:
                server.close()
                raise RuntimeError(f"cannot serve KV on {kv.kv_ip}:{port}: {error}; pick another kv_port") from error
            self._server = server
        self._thread = threading.Thread(target=self._run, name=f"lean-vllm-nixl-{self.rank}", daemon=True)
        self._thread.start()

    def start_load_kv(self, metadata: KVConnectorMetadata):
        if metadata.reqs_to_send:
            deadline = monotonic() + envs.LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT
            with self._lock:
                for request_id in metadata.reqs_to_send:
                    self._reqs_to_send[request_id] = deadline
                    self._release_if_read(request_id)
        if metadata.reqs_to_recv:
            # A step in flight may still write these blocks. Record an event for the transfer thread to wait on.
            ready = torch.cuda.current_stream().record_event() if self.mem_type == "VRAM" else None
            for request_id, req in metadata.reqs_to_recv.items():
                self._queue.put((request_id, req, ready))

    def get_finished(self) -> KVConnectorOutput:
        now = monotonic()
        timeout = envs.LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT
        with self._lock:
            for request_id, deadline in list(self._reqs_to_send.items()):
                # The producer tracks notifications, not reads in flight; the timeout must allow reads to finish.
                if deadline < now:
                    logger.warning(
                        "freeing the KV blocks of %s, which no decode instance read in %.0f s", request_id, timeout
                    )
                    self._release(request_id)
                    self._finished.expired_sending.add(request_id)
            for request_id, (_, _, seen) in list(self._notified.items()):
                if request_id not in self._reqs_to_send and seen + timeout < now:
                    del self._notified[request_id]  # discard stale notifications with no pending send
            finished, self._finished = self._finished, KVConnectorOutput()
        return finished

    def shutdown(self):
        if self._context.closed:
            return
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=10)
        self._context.term()

    def _release(self, request_id: str):
        """Mark a pending send finished on this rank. Caller must hold _lock.

        The engine frees the blocks after all ranks report completion.
        """
        if self._reqs_to_send.pop(request_id, None) is not None:
            self._notified.pop(request_id, None)
            self._finished.finished_sending.add(request_id)

    def _release_if_read(self, request_id: str):
        """Report the send as finished once all expected notifications arrive. Caller must hold _lock."""
        entry = self._notified.get(request_id)
        if entry is not None and entry[0] >= entry[1]:
            self._release(request_id)

    # After cache registration, the transfer thread handles NIXL operations and cleanup.

    def _run(self):
        try:
            while True:
                try:
                    items = [self._queue.get(timeout=BUSY_WAIT if self._loads else IDLE_WAIT)]
                except queue.Empty:
                    items = []
                while not self._queue.empty():
                    items.append(self._queue.get_nowait())
                if None in items:
                    return
                for request_id, req, ready in items:
                    self._start(request_id, req, ready)
                if self._server is not None:
                    self._serve()
                self._take_notifications()
                self._poll()
        except zmq.ContextTerminated:
            pass
        finally:
            self._close()

    def _close(self):
        for load in self._loads.values():
            for handle in load.handles:
                self.agent.release_xfer_handle(handle)
        for remote in self._remotes.values():
            self.agent.remove_remote_agent(remote.agent_name)
        self.agent.deregister_memory(self._reg_descs)
        if self._server is not None:
            self._server.close()

    # Producer: answer handshakes and count consumer notifications before releasing requests.

    def _serve(self):
        while True:
            try:
                identity, raw = self._server.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            try:
                op = json.loads(raw).get("op")
            except (ValueError, AttributeError):
                op = None
            if op == "handshake":
                reply = dict(
                    ok=True,
                    engine_id=self.kv_transfer.engine_id,
                    tp_size=self.tp_size,
                    layout=self.layout,
                    regions=[[r.addr, r.block_bytes, r.head_bytes] for r in self.regions],
                    num_blocks=self.kv_caches[0].size(1),
                    mem_type=self.mem_type,
                    device_id=self.device_id,
                )
            else:
                reply = dict(ok=False, error=f"unknown op {op!r}")
            try:
                self._server.send_multipart([identity, json.dumps(reply).encode(), self.agent_metadata], zmq.NOBLOCK)
            except zmq.Again:
                logger.warning("a KV handshake went unanswered: its consumer is not reading")

    def _take_notifications(self):
        for messages in self.agent.get_new_notifs().values():
            for message in messages:
                try:
                    request_id, readers = message.decode().rsplit(":", 1)
                    int(readers)
                except ValueError:
                    logger.warning("ignoring a malformed KV notification %r", message)
                    continue
                with self._lock:
                    entry = self._notified.setdefault(request_id, [0, int(readers), monotonic()])
                    entry[0] += 1
                    self._release_if_read(request_id)

    # Consumer: post each request's reads, then poll them.

    def _start(self, request_id: str, req: ReqToRecv, ready):
        load = self._loads[request_id] = _Load(req)
        try:
            heads = self._heads(req)
            shares = head_shares(self.rank, self.tp_size, req.remote_tp_size, heads)
            load.unnotified = {share.remote_rank for share in shares}
            if ready is not None:
                ready.synchronize()
            for i, share in enumerate(shares):
                remote = self._handshake(req, share.remote_rank)
                if max(req.remote_block_ids, default=-1) >= remote.num_blocks:
                    raise RuntimeError(f"a block id past the producer's {remote.num_blocks} blocks")
                # Swap producer and consumer roles to count the consumers assigned to this producer rank.
                readers = len(head_shares(share.remote_rank, req.remote_tp_size, self.tp_size, heads))
                notification = f"{req.remote_request_id}:{readers}".encode()
                args = (share.num_heads, i == 0)  # read whole latent blocks only from the first share
                local = block_descs(self.regions, req.local_block_ids, share.local_head, *args, self.device_id)
                if not len(local):
                    self.agent.send_notif(remote.agent_name, notification)
                else:
                    remote_rows = block_descs(
                        remote.regions, req.remote_block_ids, share.remote_head, *args, remote.device_id
                    )
                    handle = self.agent.initialize_xfer(
                        "READ",
                        self.agent.get_xfer_descs(local, self.mem_type),
                        self.agent.get_xfer_descs(remote_rows, remote.mem_type),
                        remote.agent_name,
                        notification,
                    )
                    load.handles.append(handle)
                    if self.agent.transfer(handle) == "ERR":
                        raise RuntimeError(f"NIXL refused the read from producer rank {share.remote_rank}")
                    load.num_bytes += int(local[:, 1].sum())
                load.unnotified.discard(share.remote_rank)
        except Exception as error:
            self._fail(load, error)

    def _poll(self):
        for request_id, load in list(self._loads.items()):
            pending = []
            for handle in load.handles:
                try:
                    state = self.agent.check_xfer_state(handle)
                except Exception as error:
                    state = f"ERR ({error})"
                if state == "PROC":
                    pending.append(handle)
                    continue
                self.agent.release_xfer_handle(handle)
                if state != "DONE":
                    self._fail(load, RuntimeError(f"a NIXL read ended {state}"))
            load.handles = pending
            if pending:
                continue  # wait even after failure: pending reads may still write blocks the engine could reuse
            del self._loads[request_id]
            with self._lock:
                if load.failed:
                    self._finished.failed_recving.add(request_id)
                else:
                    self._finished.finished_recving.add(request_id)
                    self._finished.recv_stats[request_id] = KVTransferStats(monotonic() - load.started, load.num_bytes)

    def _fail(self, load: _Load, error: Exception):
        req = load.req
        if not load.failed:
            logger.warning(
                "loading the KV of %s from %s:%d failed, so it prefills here: %s",
                req.remote_request_id,
                req.remote_host,
                req.remote_port,
                error,
            )
        load.failed = True
        # Attempt missing notifications for known producer ranks so block release need not wait for a timeout.
        for remote_rank in list(load.unnotified):
            remote = self._remotes.get((req.remote_host, req.remote_port + remote_rank))
            if remote is not None and remote.engine_id == req.remote_engine_id:
                readers = len(head_shares(remote_rank, req.remote_tp_size, self.tp_size, self._heads(req)))
                try:
                    self.agent.send_notif(remote.agent_name, f"{req.remote_request_id}:{readers}".encode())
                except Exception as notify_error:  # the producer's timeout allows the blocks to be released later
                    logger.debug("notifying producer rank %d failed: %s", remote_rank, notify_error)
            load.unnotified.discard(remote_rank)

    def _heads(self, req: ReqToRecv) -> int:
        """Return the head count used to assign producer ranks to consumers.

        For models with only replicated latents, use a virtual head count divisible by both tensor-parallel sizes.
        """
        return self.num_heads or math.lcm(self.tp_size, req.remote_tp_size)

    def _handshake(self, req: ReqToRecv, remote_rank: int) -> _Remote:
        """Return cached producer information, or fetch and validate its engine ID, tensor-parallel size and layout."""
        key = (req.remote_host, req.remote_port + remote_rank)
        remote = self._remotes.get(key)
        if remote is not None:
            if remote.engine_id == req.remote_engine_id:
                return remote
            del self._remotes[key]  # the requested engine ID changed; refresh the cached producer
            self.agent.remove_remote_agent(remote.agent_name)
        sock = self._context.socket(zmq.DEALER)
        try:
            sock.rcvtimeo = sock.sndtimeo = int(HANDSHAKE_TIMEOUT * 1000)
            sock.connect(f"tcp://{key[0]}:{key[1]}")
            sock.send(json.dumps(dict(op="handshake")).encode())
            try:
                raw, agent_metadata = sock.recv_multipart()
            except zmq.Again:
                raise ConnectionError(f"{key[0]}:{key[1]} did not answer the handshake") from None
        finally:
            sock.close()
        reply = json.loads(raw)
        if not reply["ok"]:
            raise RuntimeError(reply["error"])
        if reply["engine_id"] != req.remote_engine_id:
            raise RuntimeError(f"{key[0]}:{key[1]} is engine {reply['engine_id']}, not {req.remote_engine_id}")
        if reply["tp_size"] != req.remote_tp_size:
            raise RuntimeError(f"{key[0]}:{key[1]} runs tp_size {reply['tp_size']}, not {req.remote_tp_size}")
        if reply["layout"] != self.layout:
            raise RuntimeError(
                f"{key[0]}:{key[1]} lays its KV cache out as {reply['layout']}, "
                f"this engine as {self.layout}; serve the same model, dtype and block size"
            )
        agent_name = self.agent.add_remote_agent(agent_metadata)
        remote = _Remote(
            reply["engine_id"],
            agent_name,
            [Region(*region) for region in reply["regions"]],
            reply["num_blocks"],
            reply["mem_type"],
            reply["device_id"],
        )
        self._remotes[key] = remote
        return remote
