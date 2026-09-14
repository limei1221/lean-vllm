"""The synchronous engine on a dedicated thread, with an async front door.

`step()` blocks for a whole forward pass, which would starve the HTTP handlers on
the event loop. Requests go in on a thread-safe queue; each step's outputs come back
to the loop in one callback, which feeds the per-request streams.
"""

import asyncio
import threading
from dataclasses import dataclass
from queue import Empty, Queue
from typing import AsyncIterator, Callable
from uuid import uuid4

from lean_vllm.engine.llm_engine import LLMEngine
from lean_vllm.engine.output import RequestOutput
from lean_vllm.sampling_params import SamplingParams

# Wait on intake after an empty step, but poll so a wedged request cannot deadlock the loop.
IDLE_POLL = 0.005


class EngineDeadError(RuntimeError):
    """The engine thread raised. Nothing can be served until the process restarts."""


class AsyncStream:
    """One request's outputs: fed on the event loop, read by its handler."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()

    def put(self, item: RequestOutput | Exception):
        self._queue.put_nowait(item)

    async def __aiter__(self) -> AsyncIterator[RequestOutput]:
        while True:
            item = await self._queue.get()
            if isinstance(item, Exception):
                raise item
            yield item
            if item.finished:
                return


@dataclass(slots=True)
class _Add:
    prompt: list[int]
    sampling_params: SamplingParams
    request_id: str
    accepted: asyncio.Future


@dataclass(slots=True)
class _Abort:
    request_id: str
    reason: str = "abort"


class _Stop:
    pass


class AsyncLLMEngine:

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.tokenizer = engine.tokenizer
        self.metrics = engine.metrics
        self.error: BaseException | None = None
        self.on_death: Callable[[], None] | None = None
        self._intake: Queue = Queue()
        self._streams: dict[str, AsyncStream] = {}    # event loop only
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @classmethod
    def from_engine_args(cls, model: str, **kwargs) -> "AsyncLLMEngine":
        return cls(LLMEngine(model, **kwargs))

    @property
    def is_dead(self) -> bool:
        return self.error is not None

    def start(self):
        """Called from the event loop that will read the streams."""
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, name="lean-vllm-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0):
        self._intake.put(_Stop())    # FIFO, so adds already submitted are settled first
        if self._thread is not None:
            self._thread.join(timeout)
        self._call_soon(self._fail_streams, EngineDeadError("the server is shutting down"))

    async def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncIterator[RequestOutput]:
        """Returns a generator of per-step outputs; closing it aborts the request."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        request_id = request_id or f"req-{uuid4().hex}"
        stream = AsyncStream()
        accepted = self._loop.create_future()
        self._submit(_Add(prompt, sampling_params, request_id, accepted))
        self._streams[request_id] = stream    # no callback for it can run before this line
        try:
            await accepted    # admission is settled before the caller sends a status code
        except asyncio.CancelledError:
            self.abort(request_id)    # intake is FIFO, so the abort lands behind the add
            raise
        return self._generate(request_id, stream)

    async def _generate(self, request_id: str, stream: AsyncStream) -> AsyncIterator[RequestOutput]:
        try:
            async for output in stream:
                yield output
        finally:
            self.abort(request_id)    # this is what makes a disconnect free KV blocks

    def abort(self, request_id: str, reason: str = "abort"):
        """Non-blocking and never raising, so it is safe in a generator's finally."""
        self._streams.pop(request_id, None)
        try:
            self._submit(_Abort(request_id, reason))
        except EngineDeadError:
            pass    # the blocks went with the thread

    def _submit(self, action: _Add | _Abort):
        if self.is_dead:
            raise EngineDeadError("the engine thread died") from self.error
        self._intake.put(action)
        if self.is_dead and isinstance(action, _Add) and not action.accepted.done():
            action.accepted.set_exception(self._dead_error())    # it died as we submitted

    # --- event loop, called from the engine thread ---

    def _call_soon(self, callback: Callable, *args):
        try:
            self._loop.call_soon_threadsafe(callback, *args)
        except (AttributeError, RuntimeError):
            pass    # never started, or the loop is closed and its handlers with it

    def _settle(self, action: _Add, error: Exception | None):
        if error is not None:
            self._streams.pop(action.request_id, None)
        if action.accepted.done():
            return    # cancelled, or already failed by _submit
        action.accepted.set_exception(error) if error else action.accepted.set_result(None)

    def _deliver(self, outputs: list[RequestOutput]):
        for output in outputs:
            stream = self._streams.get(output.request_id)
            if stream is None:
                continue    # aborted since the step
            stream.put(output)
            if output.finished:
                del self._streams[output.request_id]

    def _fail_streams(self, error: Exception):
        streams, self._streams = self._streams, {}
        for stream in streams.values():
            stream.put(error)

    # --- engine thread ---

    def _run(self):
        try:
            wait: float | None = 0
            while self._drain_intake(None if self.engine.is_finished() else wait):
                if self.engine.is_finished():
                    continue
                outputs, num_prefill_tokens, num_decode_tokens = self.engine.step()
                wait = 0 if outputs or num_prefill_tokens or num_decode_tokens else IDLE_POLL
                if outputs:
                    self._call_soon(self._deliver, outputs)
        except BaseException as error:
            self._die(error)
        finally:
            # Flush the profiler on the thread that started it; atexit runs on the main thread.
            if self.engine.profiler is not None:
                self.engine.profiler.close()

    def _drain_intake(self, timeout: float | None) -> bool:
        """Waits up to timeout (None: forever) for the first action. False once told to stop."""
        while True:
            try:
                action = self._intake.get(timeout=timeout)
            except Empty:
                return True
            timeout = 0
            if isinstance(action, _Stop):
                return False
            if isinstance(action, _Abort):
                self.engine.abort_request(action.request_id, action.reason)
                continue
            try:
                self.engine.add_request(action.prompt, action.sampling_params, action.request_id)
            except Exception as error:    # QueueFull, and anything else intake can reject
                self._call_soon(self._settle, action, error)
                continue
            self._call_soon(self._settle, action, None)

    def _die(self, error: BaseException):
        """A status code cannot be retracted, so live streams get the error instead."""
        self.error = error
        self._call_soon(self._fail_streams, self._dead_error())
        while True:    # nobody is left to accept the intake queue
            try:
                action = self._intake.get_nowait()
            except Empty:
                break
            if isinstance(action, _Add):
                self._call_soon(self._settle, action, self._dead_error())
        if self.on_death is not None:
            self._call_soon(self.on_death)

    def _dead_error(self) -> EngineDeadError:
        dead = EngineDeadError(f"the engine thread died: {self.error!r}")
        dead.__cause__ = self.error
        return dead
