"""The synchronous engine on a dedicated thread, with an async front door.

`step()` blocks in C for a whole forward pass. Running it on the event loop
would starve the HTTP handlers between steps and show up as TTFT jitter, so the
loop owns a thread and talks to it through two queues: a thread-safe intake
queue in, a per-request `asyncio.Queue` out.
"""

import asyncio
import threading
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import AsyncIterator, Callable
from uuid import uuid4

from lean_vllm.engine.llm_engine import LLMEngine
from lean_vllm.engine.output import RequestOutput
from lean_vllm.sampling_params import SamplingParams

# The scheduler can hand back an empty step (a prompt that no longer fits the
# cache, say). Nothing but intake can change that, but poll rather than block
# forever so a wedged request cannot deadlock the loop.
IDLE_POLL = 0.005


class EngineDeadError(RuntimeError):
    """The engine thread raised. Nothing can be served until the process restarts."""


class AsyncStream:
    """One request's outputs: written by the engine thread, read by its handler."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue()
        self._done = False

    def put(self, item: RequestOutput | Exception):
        if self._done:
            return
        self._done = isinstance(item, Exception) or item.finished
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            pass    # loop already closed; the handler is gone with it

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
    stream: AsyncStream
    accepted: asyncio.Future


@dataclass(slots=True)
class _Abort:
    request_id: str


class AsyncLLMEngine:

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.tokenizer = engine.tokenizer
        self.max_model_len = engine.config.max_model_len
        self.metrics = engine.metrics
        self.error: BaseException | None = None
        self.on_death: Callable[[], None] | None = None
        self._intake: SimpleQueue = SimpleQueue()
        self._streams: dict[str, AsyncStream] = {}
        self._work = threading.Event()
        self._stopping = threading.Event()
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
        self._stopping.set()
        self._work.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._fail_streams(EngineDeadError("the server is shutting down"))

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
        stream = AsyncStream(self._loop)
        accepted = self._loop.create_future()
        self._submit(_Add(prompt, sampling_params, request_id, stream, accepted))
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

    def abort(self, request_id: str):
        """Non-blocking and never raising, so it is safe in a generator's finally."""
        try:
            self._submit(_Abort(request_id))
        except EngineDeadError:
            pass    # the blocks went with the thread

    def _submit(self, action: _Add | _Abort):
        if self.is_dead:
            raise EngineDeadError("the engine thread died") from self.error
        self._intake.put(action)
        self._work.set()
        if self.is_dead and isinstance(action, _Add) and not action.accepted.done():
            action.accepted.set_exception(self._dead_error())    # it died as we submitted

    # --- engine thread ---

    def _run(self):
        try:
            while not self._stopping.is_set():
                self._work.clear()
                self._drain_intake()
                if self.engine.is_finished():
                    self._work.wait()    # idle rather than spin
                    continue
                outputs, num_prefill_tokens, num_decode_tokens = self.engine.step()
                if not (outputs or num_prefill_tokens or num_decode_tokens):
                    self._work.wait(IDLE_POLL)
                for output in outputs:
                    self._dispatch(output)
        except BaseException as error:
            self._die(error)

    def _drain_intake(self):
        while True:
            try:
                action = self._intake.get_nowait()
            except Empty:
                return
            if isinstance(action, _Abort):
                self.engine.abort_request(action.request_id)
                self._streams.pop(action.request_id, None)
                continue
            try:
                self.engine.add_request(action.prompt, action.sampling_params, action.request_id)
            except Exception as error:    # QueueFull, and anything else intake can reject
                self._settle(action.accepted, error)
                continue
            self._streams[action.request_id] = action.stream
            self._settle(action.accepted, None)

    def _dispatch(self, output: RequestOutput):
        stream = self._streams.get(output.request_id)
        if stream is None:
            return    # aborted between the step and here
        stream.put(output)
        if output.finished:
            del self._streams[output.request_id]

    def _settle(self, accepted: asyncio.Future, error: Exception | None):
        def resolve():
            if accepted.cancelled():
                return
            accepted.set_exception(error) if error else accepted.set_result(None)

        try:
            self._loop.call_soon_threadsafe(resolve)
        except RuntimeError:
            pass

    def _die(self, error: BaseException):
        """A status code cannot be retracted, so live streams get the error instead."""
        self.error = error
        self._fail_streams(self._dead_error())
        while True:    # nobody is left to accept the intake queue
            try:
                action = self._intake.get_nowait()
            except Empty:
                break
            if isinstance(action, _Add):
                self._settle(action.accepted, self._dead_error())
        if self.on_death is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self.on_death)
            except RuntimeError:
                pass

    def _dead_error(self) -> EngineDeadError:
        dead = EngineDeadError(f"the engine thread died: {self.error!r}")
        dead.__cause__ = self.error
        return dead

    def _fail_streams(self, error: Exception):
        streams, self._streams = self._streams, {}
        for stream in streams.values():
            stream.put(error)
