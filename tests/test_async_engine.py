"""The thread boundary: streaming out, abort in, and what a dead engine owes its callers."""

import asyncio
import threading

import pytest

from conftest import FakeConfig, FakeLLMEngine, FakeModelRunner, asyncio_test
from lean_vllm.engine.async_engine import AsyncLLMEngine, EngineDeadError
from lean_vllm.engine.scheduler import QueueFull
from lean_vllm.engine.sequence import Sequence
from lean_vllm.sampling_params import SamplingParams

FOREVER = SamplingParams(max_tokens=64, ignore_eos=True)


def prompt(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


class GatedModelRunner(FakeModelRunner):
    """One forward pass per release(), so a test is never racing the engine thread.

    Only safe to gate while no add is outstanding: a blocked runner is a thread
    that is not draining intake.
    """

    def __init__(self):
        super().__init__()
        self.gate = threading.Semaphore(0)

    def run(self, seqs):
        self.gate.acquire()
        return super().run(seqs)

    def release(self, steps: int = 1):
        for _ in range(steps):
            self.gate.release()


@pytest.fixture
def make_async_engine():
    engines = []

    def _make(gated: bool = False, **overrides) -> AsyncLLMEngine:
        config = FakeConfig(**overrides)
        Sequence.block_size = config.kvcache_block_size
        runner = GatedModelRunner() if gated else FakeModelRunner()
        engine = AsyncLLMEngine(FakeLLMEngine(config, runner))
        engine.start()
        engines.append((engine, runner))
        return engine

    yield _make
    for engine, runner in engines:
        if isinstance(runner, GatedModelRunner):
            runner.release(100)    # so a blocked thread can see the stop flag
        engine.stop(timeout=5)


def runner_of(engine: AsyncLLMEngine) -> GatedModelRunner:
    return engine.engine.model_runner


async def collect(engine: AsyncLLMEngine, tokens: int, prompt_ids: list[int]) -> list[int]:
    outputs = await engine.add_request(prompt_ids, SamplingParams(max_tokens=tokens, ignore_eos=True))
    return [token async for output in outputs for token in output.token_ids]


class TestStreaming:

    @asyncio_test
    async def test_every_token_reaches_the_caller(self, make_async_engine):
        engine = make_async_engine()
        assert len(await collect(engine, 5, prompt(8))) == 5

    @asyncio_test
    async def test_the_last_output_carries_the_reason_and_metrics(self, make_async_engine):
        engine = make_async_engine()
        outputs = await engine.add_request(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        last = [output async for output in outputs][-1]
        assert last.finished and last.finish_reason == "length"
        assert last.metrics.ttft is not None

    @asyncio_test
    async def test_a_step_feeds_every_stream_it_produced(self, make_async_engine):
        engine = make_async_engine()
        results = await asyncio.gather(*[collect(engine, 4, prompt(8, i * 100)) for i in range(3)])
        assert [len(tokens) for tokens in results] == [4, 4, 4]

    @asyncio_test
    async def test_the_thread_idles_between_requests(self, make_async_engine):
        """Not a spin loop: the runner is untouched while nothing is scheduled."""
        engine = make_async_engine()
        await asyncio.sleep(0.05)
        assert engine.engine.model_runner.batches == []


class TestAbort:

    @pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
    @pytest.mark.parametrize("disconnect_at", ["http.response.start", "http.response.body"])
    @asyncio_test
    async def test_chat_disconnect_before_tokens_frees_the_blocks(
        self, make_async_engine, spec_version, disconnect_at,
    ):
        pytest.importorskip("fastapi", reason="the serve extra is not installed")
        from starlette.requests import ClientDisconnect
        from lean_vllm.entrypoints.api_server import _serve
        from lean_vllm.entrypoints.protocol import ChatCompletionRequest

        engine = make_async_engine(gated=True)
        blocks = engine.engine.scheduler.block_manager
        free_before = len(blocks.free_block_ids)
        body = ChatCompletionRequest(
            model="fake", messages=[{"role": "user", "content": "hi"}], stream=True,
        )
        response = await _serve(engine, "fake", body, prompt(8), chat=True)
        disconnected = asyncio.Event()

        async def send(message):
            if message["type"] == disconnect_at:
                if spec_version == "2.4":
                    raise OSError("client disconnected")
                disconnected.set()
                await asyncio.Future()    # cancelled by the disconnect listener

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        scope = {"type": "http", "asgi": {"spec_version": spec_version}}
        if spec_version == "2.4":
            with pytest.raises(ClientDisconnect):
                await asyncio.wait_for(response(scope, receive, send), timeout=1)
        else:
            await asyncio.wait_for(response(scope, receive, send), timeout=1)

        runner_of(engine).release()    # drain the abort after the in-flight step
        await wait_until(lambda: not engine._streams)
        assert not engine.engine.scheduler.seqs
        assert len(blocks.free_block_ids) == free_before
        assert engine.metrics.requests_aborted.total == 1

    @asyncio_test
    async def test_closing_the_generator_frees_the_blocks(self, make_async_engine):
        """This is what makes a client disconnect release KV, so it is the load-bearing test."""
        engine = make_async_engine(gated=True)
        blocks = engine.engine.scheduler.block_manager
        free_before = len(blocks.free_block_ids)

        outputs = await engine.add_request(prompt(8), FOREVER)
        runner_of(engine).release()
        await outputs.__anext__()
        assert len(blocks.free_block_ids) < free_before

        await outputs.aclose()
        runner_of(engine).release()    # the abort is drained at the top of the next step
        await settle(engine)
        assert len(blocks.free_block_ids) == free_before
        assert not engine.engine.scheduler.seqs

    @asyncio_test
    async def test_cancelling_admission_leaves_nothing_running(self, make_async_engine):
        """No generator exists yet to abort in its finally, so add_request has to do it."""
        engine = make_async_engine(gated=True)
        blocks = engine.engine.scheduler.block_manager
        free_before = len(blocks.free_block_ids)

        adding = asyncio.create_task(engine.add_request(prompt(8), FOREVER, "cancelled"))
        await asyncio.sleep(0)    # submitted, now waiting on admission
        adding.cancel()
        with pytest.raises(asyncio.CancelledError):
            await adding

        runner_of(engine).release(2)    # the add is drained, then the abort behind it
        await settle(engine)
        assert not engine.engine.scheduler.seqs
        assert not engine._streams
        assert len(blocks.free_block_ids) == free_before

    @asyncio_test
    async def test_abort_of_an_unknown_request_is_harmless(self, make_async_engine):
        engine = make_async_engine()
        engine.abort("req-nobody")
        await settle(engine)
        assert not engine.is_dead


class TestAdmission:

    @asyncio_test
    async def test_a_full_queue_is_refused_before_any_output(self, make_async_engine):
        """The 429 must surface from add_request, while a status code can still be chosen."""
        engine = make_async_engine(gated=True, max_waiting_requests=1, num_kvcache_blocks=1)
        await engine.add_request(prompt(1), FOREVER)    # occupies the cache
        waiting = asyncio.create_task(engine.add_request(prompt(8, 100), FOREVER))
        await asyncio.sleep(0)    # enqueue before releasing the current step
        runner_of(engine).release()
        await waiting

        refused = asyncio.create_task(engine.add_request(prompt(8, 200), FOREVER))
        await asyncio.sleep(0)
        runner_of(engine).release()
        with pytest.raises(QueueFull):
            await refused

    @pytest.mark.parametrize("chunked", [False, True])
    @asyncio_test
    async def test_oversized_prompt_reports_capacity_and_engine_keeps_serving(self, make_async_engine, chunked):
        engine = make_async_engine(num_kvcache_blocks=1, enable_chunked_prefill=chunked)
        outputs = await engine.add_request(prompt(9), FOREVER)

        output = await asyncio.wait_for(anext(outputs), timeout=1)

        assert output.finished and output.finish_reason == "capacity"
        assert output.token_ids == []
        await outputs.aclose()
        assert len(await asyncio.wait_for(collect(engine, 2, prompt(1)), timeout=1)) == 2
        assert not engine.is_dead

    @asyncio_test
    async def test_a_prompt_that_cannot_fit_finishes_rather_than_hanging(self, make_async_engine):
        engine = make_async_engine(max_num_batched_tokens=16, enable_chunked_prefill=False)
        outputs = await engine.add_request(prompt(40), FOREVER)
        collected = [output async for output in outputs]
        assert [(output.finished, output.finish_reason) for output in collected] == [(True, "capacity")]


class TestEngineDeath:

    @asyncio_test
    async def test_a_live_stream_gets_the_error(self, make_async_engine):
        engine = make_async_engine(gated=True)
        outputs = await engine.add_request(prompt(8), FOREVER)
        runner_of(engine).release()
        await outputs.__anext__()

        kill(engine)
        runner_of(engine).release()
        with pytest.raises(EngineDeadError):
            async for _ in outputs:
                pass
        assert engine.is_dead

    @asyncio_test
    async def test_new_requests_are_refused(self, make_async_engine):
        engine = make_async_engine(gated=True)
        await engine.add_request(prompt(8), FOREVER)
        kill(engine)
        runner_of(engine).release()
        await wait_until(lambda: engine.is_dead)
        with pytest.raises(EngineDeadError):
            await engine.add_request(prompt(8, 100), FOREVER)


def kill(engine: AsyncLLMEngine):
    """Blow up the next step, the way an exception in the runner would."""
    def explode():
        raise RuntimeError("boom")

    engine.engine.step = explode
    engine._work.set()


async def settle(engine: AsyncLLMEngine):
    await wait_until(engine.engine.is_finished)


async def wait_until(predicate, steps: int = 200):
    for _ in range(steps):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never held")
