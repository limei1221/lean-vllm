"""NixlConnector's protocol over ZMQ: the decode instance pulls a finished prefill's blocks, then releases them.

A producer holds a request's prompt blocks once it has sampled its one token, and serves them from a ROUTER socket on
kv_port + rank. A consumer reads them into blocks it allocated, over a DEALER socket on a thread of its own, then sends
"done" so the producer frees them; blocks no consumer reads are freed after LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT. NIXL
moves the same blocks GPU to GPU over RDMA; this stages them through host memory, one message per layer.
"""

import json
import logging
import queue
import threading
from collections import Counter
from contextlib import nullcontext
from time import monotonic

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

REGISTRATION_WAIT = 30.0  # how long a read waits for the engine to hand its request's blocks over
HANDSHAKE_TIMEOUT = 5.0  # ZMQ never refuses a connect, so an absent producer is one that does not answer
SOCKET_TIMEOUT = 60.0
POLL_MS = 50  # how often the server retries reads that arrived before their blocks
LAYERS_IN_FLIGHT = 2  # high-water mark, in layer messages, so a read holds a few layers in host memory, not all


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def send_message(sock: zmq.Socket, message: dict, *identity: bytes):
    """A JSON message; a ROUTER prefixes the peer it answers."""
    sock.send_multipart([*identity, json.dumps(message).encode()])


def recv_message(sock: zmq.Socket) -> dict:
    try:
        return json.loads(sock.recv())
    except zmq.Again:
        raise ConnectionError("the peer did not reply in time") from None


class TcpConnectorScheduler(KVConnectorScheduler):
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
        problem = None
        if len(params["remote_block_ids"]) != num_blocks:
            problem = f"{len(params['remote_block_ids'])} remote blocks for a {num_blocks}-block prompt"
        elif params.get("tp_size", 1) != self.tp_size:
            problem = f"the prefill instance runs tp_size {params.get('tp_size', 1)}, this one {self.tp_size}"
        if problem:
            logger.warning("prefilling %s here instead: %s", seq.request_id, problem)
            params["do_remote_prefill"] = False
            return 0, False
        return seq.num_prompt_tokens - num_computed_tokens, True

    def update_state_after_alloc(self, seq, num_external_tokens: int):
        params = seq.kv_transfer_params
        params["do_remote_prefill"] = False  # one load per request, so a preempted one recomputes
        num_local_blocks = seq.num_cached_tokens // self.block_size  # prefix-cache hits here are not read
        self._reqs_to_recv[seq.request_id] = ReqToRecv(
            local_block_ids=seq.block_table[num_local_blocks:],
            remote_block_ids=params["remote_block_ids"][num_local_blocks:],
            remote_request_id=params["remote_request_id"],
            remote_engine_id=params["remote_engine_id"],
            remote_host=params["remote_host"],
            remote_port=params["remote_port"],
        )

    def build_connector_meta(self) -> KVConnectorMetadata:
        metadata = KVConnectorMetadata(self._reqs_to_recv, self._reqs_to_send)
        self._reqs_to_recv, self._reqs_to_send = {}, {}
        return metadata

    def request_finished(self, seq) -> tuple[bool, dict | None]:
        params = seq.kv_transfer_params
        # A prefill that hit its max_tokens or a stop token hands over; an abort ends there.
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


class TcpConnectorWorker(KVConnectorWorker):
    def __init__(self, config, rank: int):
        self.kv_transfer = config.kv_transfer
        self.rank = rank
        self.block_size = config.kvcache_block_size
        self.kv_caches: list[torch.Tensor] = []
        self.layout: dict = {}
        self._lock = threading.Lock()  # guards the three below, shared with the server and loader threads
        self._reqs_to_send: dict[str, tuple[set[int], float]] = {}  # request -> (its blocks, when to give up)
        self._reading: Counter[str] = Counter()  # reads in progress, which expiry must wait out
        self._finished = KVConnectorOutput()
        self._recv_queue: queue.Queue = queue.Queue()
        self._context = zmq.Context()
        self._context.setsockopt(zmq.LINGER, 0)  # a socket closes at once, with nobody left to read it
        self._peers: dict[tuple[str, int], tuple[zmq.Socket, str]] = {}  # loader thread only
        self._server: zmq.Socket | None = None  # server thread only, once started
        self._threads: list[threading.Thread] = []

    def register_kv_caches(self, kv_caches: list[torch.Tensor]):
        self.kv_caches = kv_caches
        # What a peer must match: the same blocks, in the same layout, layer for layer.
        self.layout = dict(
            block_size=self.block_size,
            layers=[[str(cache.dtype), [cache.size(0), *cache.shape[2:]]] for cache in kv_caches],
        )
        kv = self.kv_transfer
        if kv.is_producer:
            port = kv.kv_port + self.rank
            server = self._context.socket(zmq.ROUTER)
            server.sndhwm = LAYERS_IN_FLIGHT
            server.sndtimeo = int(SOCKET_TIMEOUT * 1000)
            server.router_mandatory = True  # a full or vanished peer raises rather than drops a layer
            try:
                server.bind(f"tcp://{kv.kv_ip}:{port}")
            except zmq.ZMQError as error:
                server.close()
                raise RuntimeError(f"cannot serve KV on {kv.kv_ip}:{port}: {error}; pick another kv_port") from error
            self._server = server
            self._start(self._serve_loop, "kv-server")
        if kv.is_consumer:
            self._start(self._load_loop, "kv-loader")

    def _start(self, target, name: str):
        thread = threading.Thread(target=target, name=f"lean-vllm-{name}-{self.rank}", daemon=True)
        thread.start()
        self._threads.append(thread)

    def start_load_kv(self, metadata: KVConnectorMetadata):
        if metadata.reqs_to_send:
            deadline = monotonic() + envs.LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT
            with self._lock:
                for request_id, block_ids in metadata.reqs_to_send.items():
                    self._reqs_to_send[request_id] = (set(block_ids), deadline)
        if metadata.reqs_to_recv:
            # Freshly allocated blocks may still be written by a step in flight; the loads wait for it.
            ready = torch.cuda.current_stream().record_event() if self._on_cuda else None
            for request_id, req in metadata.reqs_to_recv.items():
                self._recv_queue.put((request_id, req, ready))

    def get_finished(self) -> KVConnectorOutput:
        now = monotonic()
        with self._lock:
            for request_id, (_, deadline) in list(self._reqs_to_send.items()):
                if deadline < now and not self._reading[request_id]:
                    logger.warning(
                        "freeing the KV blocks of %s, which no decode instance read in %.0f s",
                        request_id,
                        envs.LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT,
                    )
                    self._release(request_id)
                    self._finished.expired_sending.add(request_id)
            finished, self._finished = self._finished, KVConnectorOutput()
        return finished

    def shutdown(self):
        if self._context.closed:
            return
        self._recv_queue.put(None)
        # Wakes a thread blocked on a socket with ETERM; returns once each has closed its own.
        self._context.term()
        for thread in self._threads:
            thread.join(timeout=5)

    @property
    def _on_cuda(self) -> bool:
        return bool(self.kv_caches) and self.kv_caches[0].is_cuda

    def _thread_stream(self):
        """A side stream for this thread's copies, so they overlap the engine's steps."""
        if not self._on_cuda:
            return None
        torch.cuda.set_device(self.kv_caches[0].device)
        return torch.cuda.Stream()

    def _layer_nbytes(self, cache: torch.Tensor, num_blocks: int) -> int:
        return cache[:, 0].numel() * cache.element_size() * num_blocks

    def _release(self, request_id: str):
        """Lock held. The engine frees the blocks once every rank has reported them."""
        if self._reqs_to_send.pop(request_id, None) is not None:
            self._finished.finished_sending.add(request_id)

    # Producer: one server thread, answering every consumer in turn.

    def _serve_loop(self):
        server = self._server
        assert server is not None
        stream = self._thread_stream()
        pending = []  # (peer, message, when a read stops waiting for its blocks)
        try:
            while True:
                if server.poll(POLL_MS):
                    while True:
                        try:
                            identity, raw = server.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        pending.append((identity, json.loads(raw), monotonic() + REGISTRATION_WAIT))
                waiting = []
                for identity, message, deadline in pending:
                    # The engine hands the blocks over on the step after the request finishes, which can trail the reply.
                    if message.get("op") == "read" and deadline > monotonic():
                        with self._lock:
                            if message["request_id"] not in self._reqs_to_send:
                                waiting.append((identity, message, deadline))
                                continue
                    try:
                        self._serve(server, identity, message, stream)
                    except zmq.ContextTerminated:
                        raise
                    except zmq.ZMQError as error:  # the consumer left or stalled; it fails the load on its side
                        logger.warning("answering a KV %s failed: %s", message.get("op"), error)
                pending = waiting
        except zmq.ContextTerminated:
            pass
        finally:
            server.close()

    def _serve(self, server: zmq.Socket, identity: bytes, message: dict, stream):
        op = message.get("op")
        if op == "handshake":
            send_message(server, dict(ok=True, engine_id=self.kv_transfer.engine_id, layout=self.layout), identity)
        elif op == "read":
            self._serve_read(server, identity, message["request_id"], message["block_ids"], stream)
        elif op == "done":
            with self._lock:
                self._release(message["request_id"])
            send_message(server, dict(ok=True), identity)
        else:
            send_message(server, dict(ok=False, error=f"unknown op {op!r}"), identity)

    def _serve_read(self, server: zmq.Socket, identity: bytes, request_id: str, block_ids: list[int], stream):
        with self._lock:
            held = self._reqs_to_send.get(request_id, (None,))[0]
            readable = held is not None and set(block_ids) <= held
            if readable:
                self._reading[request_id] += 1
        if not readable:
            error = f"{request_id} holds no such blocks here; it expired, or never prefilled here"
            send_message(server, dict(ok=False, error=error), identity)
            return
        try:
            num_bytes = sum(self._layer_nbytes(cache, len(block_ids)) for cache in self.kv_caches)
            send_message(server, dict(ok=True, num_bytes=num_bytes), identity)
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                index = torch.tensor(block_ids, device=self.kv_caches[0].device)
                for cache in self.kv_caches:
                    chunk = cache.index_select(1, index).cpu()  # blocking, so the bytes are there to send
                    server.send_multipart([identity, chunk.flatten().view(torch.uint8).numpy()], copy=False)
        finally:
            with self._lock:
                self._reading[request_id] -= 1
                if not self._reading[request_id]:
                    del self._reading[request_id]

    # Consumer: one loader thread, reading requests in turn.

    def _load_loop(self):
        stream = self._thread_stream()
        try:
            while (item := self._recv_queue.get()) is not None:
                request_id, req, ready = item
                stats = None
                try:
                    stats = self._load(req, ready, stream)
                except zmq.ContextTerminated:
                    return
                except Exception as error:
                    logger.warning(
                        "loading the KV of %s from %s:%d failed, so it prefills here: %s",
                        request_id,
                        req.remote_host,
                        req.remote_port + self.rank,
                        error,
                    )
                with self._lock:
                    if stats is None:
                        self._finished.failed_recving.add(request_id)
                    else:
                        self._finished.finished_recving.add(request_id)
                        self._finished.recv_stats[request_id] = stats
        finally:
            for sock, _ in self._peers.values():
                sock.close()

    def _load(self, req: ReqToRecv, ready, stream) -> KVTransferStats:
        key = (req.remote_host, req.remote_port + self.rank)
        sock = self._connect(key, req.remote_engine_id)
        try:
            started = monotonic()
            sock.rcvtimeo = int((REGISTRATION_WAIT + SOCKET_TIMEOUT) * 1000)  # the producer may wait for the blocks
            send_message(sock, dict(op="read", request_id=req.remote_request_id, block_ids=req.remote_block_ids))
            reply = recv_message(sock)
            if reply["ok"]:
                self._receive_blocks(sock, req.local_block_ids, reply["num_bytes"], ready, stream)
                seconds = monotonic() - started
            # Done either way: blocks this engine will not read are no use held.
            send_message(sock, dict(op="done", request_id=req.remote_request_id))
            recv_message(sock)
        except BaseException:
            self._peers.pop(key, None)
            sock.close()  # replies may still be on their way
            raise
        if not reply["ok"]:
            raise RuntimeError(reply["error"])
        num_bytes = sum(self._layer_nbytes(cache, len(req.local_block_ids)) for cache in self.kv_caches)
        return KVTransferStats(seconds, num_bytes)

    def _receive_blocks(self, sock: zmq.Socket, block_ids: list[int], num_bytes: int, ready, stream):
        expected = sum(self._layer_nbytes(cache, len(block_ids)) for cache in self.kv_caches)
        if num_bytes != expected:
            raise RuntimeError(f"the producer sent {num_bytes} bytes for {len(block_ids)} blocks, not {expected}")
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            if ready is not None:
                stream.wait_event(ready)
            index = torch.tensor(block_ids, device=self.kv_caches[0].device)
            for cache in self.kv_caches:
                chunk = torch.empty(cache.size(0), len(block_ids), *cache.shape[2:], dtype=cache.dtype)
                data = chunk.view(-1).view(torch.uint8).numpy()
                try:
                    received = sock.recv_into(data)
                except zmq.Again:
                    raise ConnectionError("the producer stopped sending") from None
                if received != data.nbytes:
                    raise RuntimeError(f"a layer of {received} bytes, not {data.nbytes}")
                cache.index_copy_(1, index, chunk.to(cache.device))
        # The engine schedules the request as soon as this reports it.
        if stream is not None:
            stream.synchronize()

    def _connect(self, key: tuple[str, int], engine_id: str) -> zmq.Socket:
        """A connection to the producer rank at key, checked to hold engine_id's blocks in this layout."""
        if key in self._peers:
            sock, peer_engine_id = self._peers[key]
            if peer_engine_id == engine_id:
                return sock
            del self._peers[key]  # restarted since, so ask afresh
            sock.close()
        sock = self._context.socket(zmq.DEALER)
        try:
            sock.rcvhwm = LAYERS_IN_FLIGHT
            sock.rcvtimeo = int(HANDSHAKE_TIMEOUT * 1000)
            sock.sndtimeo = int(SOCKET_TIMEOUT * 1000)
            sock.connect(f"tcp://{key[0]}:{key[1]}")
            send_message(sock, dict(op="handshake"))
            reply = recv_message(sock)
            if reply["engine_id"] != engine_id:
                raise RuntimeError(f"{key[0]}:{key[1]} is engine {reply['engine_id']}, not {engine_id}; it restarted")
            if reply["layout"] != self.layout:
                raise RuntimeError(
                    f"{key[0]}:{key[1]} lays its KV cache out as {reply['layout']}, "
                    f"this engine as {self.layout}; serve the same model, dtype and block size"
                )
        except BaseException:
            sock.close()
            raise
        self._peers[key] = (sock, engine_id)
        return sock
