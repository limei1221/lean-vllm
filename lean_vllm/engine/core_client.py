"""Move requests and outputs between AsyncLLM and the engine core, like vLLM's AsyncMPClient.

AsyncLLM owns the streams and detokenizers. The core runs in a separate process, so its step loop does not compete
with the client's event loop for the GIL. If the core dies, pending admission and metrics calls fail with
EngineDeadError, and get_output_async() raises that error after any queued outputs have been read.
"""

import asyncio
import atexit
import logging
import multiprocessing as mp
import pickle
import shutil
import tempfile
from collections.abc import Callable
from itertools import count
from multiprocessing.connection import wait

import zmq
import zmq.asyncio

from lean_vllm.engine.core import run_engine_core
from lean_vllm.engine.exceptions import EngineDeadError
from lean_vllm.engine.output import RequestOutput
from lean_vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_MS = 10_000


def _dead_error(prefix: str, error: BaseException) -> EngineDeadError:
    dead = EngineDeadError(f"{prefix}: {error!r}")
    dead.__cause__ = error
    return dead


class AsyncMPClient:
    """Communicate with the engine core in a separate process over two ZMQ sockets."""

    def __init__(
        self,
        engine_factory: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        shutdown_timeout: float = 30.0,
    ):
        """Start the core and wait for its engine to be built and its input socket to connect."""
        self.error: BaseException | None = None
        self.shutdown_timeout = shutdown_timeout
        self._outputs: asyncio.Queue = asyncio.Queue()
        self._adding: dict[str, asyncio.Future] = {}  # requests awaiting admission replies
        self._calls: dict[int, asyncio.Future] = {}
        self._call_ids = count()
        self._stopped = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader: asyncio.Task | None = None

        # Bind the IPC sockets in a private directory before starting the core so it can connect immediately.
        self._ipc_dir = tempfile.mkdtemp(prefix="lean-vllm-")
        input_address, output_address = f"ipc://{self._ipc_dir}/in", f"ipc://{self._ipc_dir}/out"
        self._context = zmq.Context()
        self._input = self._context.socket(zmq.PUSH)
        self._input.setsockopt(zmq.SNDHWM, 0)  # no send queue limit, so zmq.Again from _send means the core is gone
        self._input.bind(input_address)
        self._async_context = zmq.asyncio.Context()
        self._output = self._async_context.socket(zmq.PULL)
        self._output.bind(output_address)

        ctx = mp.get_context("spawn")
        ready, child_ready = ctx.Pipe(duplex=False)
        # In multiprocessing, a daemon is terminated when its parent exits and cannot spawn child processes.
        # The core must not be a daemon because it spawns tensor-parallel workers.
        self._process = ctx.Process(
            target=run_engine_core,
            name="EngineCore",
            args=(engine_factory, args, kwargs or {}, input_address, output_address, child_ready),
        )
        self._process.start()
        child_ready.close()
        atexit.register(self._shutdown_process)  # clean up even if the server's lifespan handler never starts
        wait([ready, self._process.sentinel])
        try:
            status, error = ready.recv()
        except EOFError:  # the core exited without reporting its startup status
            status, error = "exited", None
        ready.close()
        if status != "ready":
            self._shutdown_process()
            raise RuntimeError(f"the engine core failed to start (exit code {self._process.exitcode})") from error
        # connect() returns before the connection is ready; wait so the first non-blocking send has a peer.
        if not self._input.poll(CONNECT_TIMEOUT_MS, zmq.POLLOUT):
            self._shutdown_process()
            raise RuntimeError("the engine core started but never connected")

    def start(self):
        """Start reading outputs and watching for core exit on the current event loop."""
        self._loop = asyncio.get_running_loop()
        self._reader = self._loop.create_task(self._read_outputs())
        self._loop.add_reader(self._process.sentinel, self._on_process_exit)

    def shutdown(self, timeout: float | None = None):
        """From the event loop's thread: stop the core (terminating it after timeout), the reader and pending calls."""
        if self._stopped:
            return
        self._stopped = True
        if self._loop is not None and not self._loop.is_closed():
            self._loop.remove_reader(self._process.sentinel)
        self._shutdown_process(timeout)
        if self._reader is not None:
            self._reader.cancel()
        self._fail_pending(EngineDeadError("the server is shutting down"))

    async def add_request_async(self, request_id: str, prompt: list[int], sampling_params: SamplingParams):
        """Wait for the core to admit the request, or raise the error that prevented admission."""
        if self._stopped:
            raise EngineDeadError("the server is shutting down")
        if self.error is not None:
            raise _dead_error("the engine core died", self.error)
        assert self._loop is not None
        added = self._adding[request_id] = self._loop.create_future()
        self._send(("add", request_id, prompt, sampling_params))
        try:
            error = await added
        except BaseException:  # cancelled while waiting, or the core died
            self._adding.pop(request_id, None)
            raise
        if error is not None:
            raise error

    def abort_request(self, request_id: str, reason: str = "abort"):
        """Send an abort without waiting; a no-op once the core is dead or shutting down."""
        if self.error is not None or self._stopped:
            return  # the blocks went with the engine
        self._send(("abort", request_id, reason))

    async def get_output_async(self) -> list[RequestOutput]:
        item = await self._outputs.get()
        if isinstance(item, EngineDeadError):
            raise item
        return item

    async def render_metrics_async(self) -> str:
        if self.error is not None:
            raise _dead_error("the engine core died", self.error)
        call_id = next(self._call_ids)
        assert self._loop is not None
        result = self._calls[call_id] = self._loop.create_future()
        self._send(("metrics", call_id))
        try:
            return await result
        finally:
            self._calls.pop(call_id, None)

    async def _read_outputs(self):
        try:
            while True:
                # Wait for input, then receive and unpickle each available message without yielding to the loop.
                # recv(NOBLOCK) returns a completed future. recv_pyobj schedules an unpickling callback,
                # which allows cancellation after receiving a message but before delivering it.
                await self._output.poll()
                while True:
                    try:
                        message = pickle.loads(await self._output.recv(zmq.NOBLOCK))
                    except zmq.Again:
                        break
                    if not self._handle(message):
                        return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._stopped:  # ignore read errors during shutdown, which closes the socket
                self._die(error)

    def _handle(self, message: tuple) -> bool:
        """Handle one core message. Return False on a death notification to stop the output reader."""
        kind = message[0]
        if kind == "outputs":
            self._outputs.put_nowait(message[1])
        elif kind == "added":
            _, request_id, error = message
            added = self._adding.pop(request_id, None)
            if added is not None and not added.done():  # ignore replies whose caller is no longer waiting
                added.set_result(error)
        elif kind == "metrics":
            _, call_id, value = message
            result = self._calls.get(call_id)
            if result is not None and not result.done():
                result.set_result(value)
        elif kind == "dead":
            self._die(message[1])
            return False
        return True

    def _send(self, message: tuple):
        """Send without waiting for socket readiness. Treat zmq.Again as a lost core connection."""
        try:
            self._input.send_pyobj(message, zmq.NOBLOCK)
        except zmq.Again:
            self._die(RuntimeError("the engine core is not connected"))

    def _on_process_exit(self):
        self._loop.remove_reader(self._process.sentinel)
        if not self._stopped:
            self._die(RuntimeError(f"the engine core process exited with code {self._process.exitcode}"))

    def _fail_pending(self, error: Exception):
        for futures in (self._adding, self._calls):
            for future in futures.values():
                if not future.done():
                    future.set_exception(error)
            futures.clear()

    def _die(self, error: BaseException):
        if self.error is not None:
            return
        logger.error("the engine core died", exc_info=error)
        self.error = error
        dead = _dead_error("the engine core died", error)
        self._fail_pending(dead)
        self._outputs.put_nowait(dead)

    def _shutdown_process(self, timeout: float | None = None):
        """Stop the core and release sockets and IPC files. Safe to call again after cleanup."""
        atexit.unregister(self._shutdown_process)
        process = self._process
        if process.is_alive():
            try:
                self._input.send_pyobj(("shutdown",), zmq.NOBLOCK)
            except zmq.ZMQError:
                pass  # still wait for the process and stop it if the shutdown message could not be sent
            process.join(self.shutdown_timeout if timeout is None else timeout)
        if process.is_alive():
            logger.warning("the engine core did not stop in time, so it is terminated")
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        self._context.destroy(linger=0)
        self._async_context.destroy(linger=0)
        shutil.rmtree(self._ipc_dir, ignore_errors=True)
