"""Run the engine in its own process, like vLLM's EngineCore, so it does not share the HTTP server's GIL.

Requests come in and outputs go out over two ZMQ sockets, as pickled tuples:
in:  ("add", request_id, prompt_token_ids, sampling_params), ("abort", request_id, reason),
     ("metrics", call_id), ("shutdown",)
out: ("added", request_id, error or None), ("outputs", [RequestOutput]), ("metrics", call_id, text), ("dead", error)
"""

import logging
import pickle
import signal
from collections.abc import Callable
from multiprocessing import parent_process
from multiprocessing.connection import Connection

import zmq

from lean_vllm.engine.scheduler import InvalidRequest

logger = logging.getLogger(__name__)

IDLE_POLL_MS = 5  # wait for input briefly when a step processes no tokens although work is outstanding


def make_engine(model: str, **kwargs):
    """Create the core's engine. AsyncLLM handles detokenization in the client process."""
    from lean_vllm.engine.llm_engine import LLMEngine

    return LLMEngine(model, detokenize=False, **kwargs)


def picklable(error: BaseException) -> BaseException:
    """Return error if it can be pickled and unpickled; otherwise return a RuntimeError containing its repr."""
    try:
        pickle.loads(pickle.dumps(error))
        return error
    except Exception:
        return RuntimeError(repr(error))


class EngineCore:
    """Handle queued messages, run an engine step, and send outputs. Wait for input when the engine is idle."""

    def __init__(self, engine, input_socket: zmq.Socket, output_socket: zmq.Socket):
        self.engine = engine
        self.input = input_socket
        self.output = output_socket
        self.running = True
        # Also watch the parent process's sentinel so an idle core wakes when the client exits.
        self.poller = zmq.Poller()
        self.poller.register(self.input, zmq.POLLIN)
        parent = parent_process()
        if parent is not None:
            self.poller.register(parent.sentinel, zmq.POLLIN)

    def run(self):
        while self.running:
            self._handle_inputs(wait_ms=None if self.engine.is_finished() else 0)
            if not self.running or self.engine.is_finished():
                continue
            outputs, num_prefill_tokens, num_decode_tokens = self.engine.step()
            if outputs:
                self.output.send_pyobj(("outputs", outputs))
            if not (outputs or num_prefill_tokens or num_decode_tokens):
                self._handle_inputs(wait_ms=IDLE_POLL_MS)

    def _handle_inputs(self, wait_ms: int | None):
        """Handle queued messages, waiting up to wait_ms for the first (0: none; None: until input or parent exit)."""
        if wait_ms is None:
            if self.input not in dict(self.poller.poll()):  # only the sentinel is ready
                logger.warning("the client process is gone, so the engine core exits")
                self.running = False
                return
        elif wait_ms and not self.input.poll(wait_ms):
            return
        while self.running:
            try:
                message = self.input.recv_pyobj(zmq.NOBLOCK)
            except zmq.Again:
                return
            self._handle(message)

    def _handle(self, message: tuple):
        kind = message[0]
        if kind == "add":
            _, request_id, prompt, sampling_params = message
            error: BaseException | None = None
            try:
                self.engine.add_request(prompt, sampling_params, request_id)
            except InvalidRequest as refused:  # the API server returns HTTP 400 for this error
                error = refused
            except Exception as unexpected:  # report this request's error and keep serving
                logger.exception("add_request failed for %s", request_id)
                error = picklable(unexpected)
            self.output.send_pyobj(("added", request_id, error))
        elif kind == "abort":
            _, request_id, reason = message
            self.engine.abort_request(request_id, reason)
        elif kind == "metrics":
            _, call_id = message
            self.output.send_pyobj(("metrics", call_id, self.engine.metrics.render()))
        elif kind == "shutdown":
            self.running = False
        else:
            raise ValueError(f"unknown message {kind!r}")


def _exit(signum, frame):
    raise SystemExit(0)


def run_engine_core(
    engine_factory: Callable,
    args: tuple,
    kwargs: dict,
    input_address: str,
    output_address: str,
    ready: Connection,
):
    """Build the engine in the spawned process, report startup status through ready, then handle requests."""
    # Ctrl-C reaches the whole process group; ignore it here so the client controls shutdown. SIGTERM raises SystemExit,
    # allowing atexit handlers to clean up the tensor-parallel workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _exit)
    try:
        engine = engine_factory(*args, **kwargs)
    except BaseException as error:
        ready.send(("failed", picklable(error)))
        raise
    context = zmq.Context()
    input_socket = context.socket(zmq.PULL)
    input_socket.connect(input_address)
    output_socket = context.socket(zmq.PUSH)
    output_socket.setsockopt(zmq.SNDHWM, 0)  # no send queue limit: a slow reader never blocks a step; outputs queue
    output_socket.connect(output_address)
    ready.send(("ready", None))
    ready.close()
    try:
        EngineCore(engine, input_socket, output_socket).run()
    except SystemExit:
        pass
    except BaseException as error:
        logger.exception("the engine core died")
        output_socket.send_pyobj(("dead", picklable(error)))
        raise
    finally:
        context.destroy(linger=1000)  # allow up to one second for queued messages to be sent
