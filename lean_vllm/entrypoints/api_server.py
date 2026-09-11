"""OpenAI-compatible HTTP server over `AsyncLLMEngine`."""

import json
from contextlib import aclosing, asynccontextmanager
from time import time
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from lean_vllm.engine.async_engine import AsyncLLMEngine, EngineDeadError
from lean_vllm.engine.output import RequestOutput
from lean_vllm.engine.scheduler import InvalidRequest, QueueFull
from lean_vllm.entrypoints import protocol
from lean_vllm.entrypoints.protocol import (
    BaseRequest,
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
)
from lean_vllm.entrypoints.stop_checker import StopChecker
from lean_vllm.sampling_params import SamplingParams

DONE = "data: [DONE]\n\n"

# The engine's own reasons; the drops below are not completions and never appear here.
FINISH_REASONS = {"stop": "stop", "length": "length", "abort": "stop"}
DROP_STATUS = {"capacity": 503, "timeout": 504}


class _RequestStreamingResponse(StreamingResponse):
    """Own admission cleanup even if sending headers or the first chunk fails."""

    def __init__(self, stream, engine: AsyncLLMEngine, request_id: str):
        super().__init__(stream, media_type="text/event-stream")
        self.engine = engine
        self.request_id = request_id

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Closing an unstarted generator does not run its finally block.
            self.engine.abort(self.request_id)
            await self.body_iterator.aclose()


def build_app(engine: AsyncLLMEngine, model: str) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start()
        yield
        engine.stop()

    app = FastAPI(title="lean-vLLM", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content=_error("invalid_request_error", _first_message(exc)))

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        kind = "server_error" if exc.status_code >= 500 else "invalid_request_error"
        return JSONResponse(status_code=exc.status_code, content=_error(kind, exc.detail))

    @app.get("/health")
    async def health():
        if engine.is_dead:
            raise HTTPException(503, f"the engine thread died: {engine.error!r}")
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(engine.metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/metrics.json")
    async def metrics_json():
        return engine.metrics.summary()

    @app.get("/v1/models")
    async def models():
        return ModelList(data=[ModelCard(id=model)])

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        prompt_token_ids = body.prompt if isinstance(body.prompt, list) else engine.tokenizer.encode(body.prompt)
        return await _serve(engine, model, body, prompt_token_ids, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        messages = [message.model_dump() for message in body.messages]
        # return_dict=False, or newer tokenizers hand back a BatchEncoding rather than ids.
        prompt_token_ids = engine.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        return await _serve(engine, model, body, prompt_token_ids, chat=True)

    return app


async def _serve(engine: AsyncLLMEngine, model: str, body: BaseRequest, prompt_token_ids: list[int], chat: bool):
    if body.model != model:
        # OpenAI's semantics, and vLLM's: a name the server does not serve is a
        # 404, not a field to ignore. A benchmark client pointed at the wrong
        # server should find out at the first request, not in the numbers.
        raise HTTPException(404, f"the model {body.model!r} does not exist")
    if engine.is_dead:
        raise HTTPException(503, f"the engine thread died: {engine.error!r}")
    _check_length(engine, body, len(prompt_token_ids))
    request_id = f"{'chatcmpl' if chat else 'cmpl'}-{uuid4().hex}"
    sampling_params = SamplingParams(
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        ignore_eos=body.ignore_eos,
        priority=body.priority,
    )
    try:
        outputs = await engine.add_request(prompt_token_ids, sampling_params, request_id)
    except InvalidRequest as invalid:
        raise HTTPException(400, str(invalid))
    except QueueFull as full:
        raise HTTPException(429, f"the engine is at capacity: {full}")
    except EngineDeadError as dead:
        raise HTTPException(503, str(dead))

    deltas = _deltas(outputs, StopChecker(body.stop_strings))
    if body.stream:
        stream = _stream(deltas, request_id, model, body, len(prompt_token_ids), chat)
        return _RequestStreamingResponse(stream, engine, request_id)
    try:
        return await _collect(deltas, request_id, model, len(prompt_token_ids), chat)
    except EngineDeadError as dead:
        raise HTTPException(503, str(dead))


def _check_length(engine: AsyncLLMEngine, body: BaseRequest, num_prompt_tokens: int):
    """Refused here rather than asserted deep in the runner."""
    limit = engine.max_model_len
    if num_prompt_tokens >= limit:
        raise HTTPException(400, f"prompt is {num_prompt_tokens} tokens, over the {limit}-token context")
    if num_prompt_tokens + body.max_tokens > limit:
        raise HTTPException(
            400,
            f"prompt ({num_prompt_tokens}) plus max_tokens ({body.max_tokens}) is over the {limit}-token context",
        )


async def _deltas(outputs: AsyncIterator[RequestOutput], checker: StopChecker):
    """Yields (text, finish_reason, num_completion_tokens); the last has a reason."""
    num_tokens = 0
    try:
        async for output in outputs:
            num_tokens += len(output.token_ids)
            text = checker.push(output.text)
            if checker.matched:
                yield text, "stop", num_tokens
                return    # the finally aborts, which is what frees the blocks
            if output.finished:
                if output.finish_reason not in FINISH_REASONS:
                    status = DROP_STATUS.get(output.finish_reason, 503)
                    raise HTTPException(status, f"the engine dropped the request: {output.finish_reason}")
                yield text + checker.flush(), FINISH_REASONS[output.finish_reason], num_tokens
                return
            if text:
                yield text, None, num_tokens
    finally:
        await outputs.aclose()


async def _collect(deltas, request_id: str, model: str, num_prompt_tokens: int, chat: bool):
    text, finish_reason, num_tokens = "", "stop", 0
    async for delta, reason, num_tokens in deltas:
        text += delta
        finish_reason = reason or finish_reason
    build = protocol.chat_body if chat else protocol.completion_body
    return build(request_id, model, text, finish_reason, protocol.usage(num_prompt_tokens, num_tokens))


async def _stream(deltas, request_id: str, model: str, body: BaseRequest, num_prompt_tokens: int, chat: bool):
    async with aclosing(deltas):
        created = int(time())
        num_tokens = 0
        if chat:
            yield _event(protocol.chat_chunk(request_id, model, created, {"role": "assistant", "content": ""}, None))
        try:
            async for delta, reason, num_tokens in deltas:
                if chat:
                    yield _event(protocol.chat_chunk(request_id, model, created, {"content": delta}, reason))
                else:
                    yield _event(protocol.completion_chunk(request_id, model, created, delta, reason))
        except (EngineDeadError, HTTPException) as error:
            # A status code cannot be retracted once the 200 went out, so the error
            # rides in the stream instead.
            yield _event(_error("server_error", str(getattr(error, "detail", error))))
            yield DONE
            return
        if body.include_usage:
            chunk = protocol.chat_chunk if chat else protocol.completion_chunk
            body_dict = chunk(request_id, model, created, {} if chat else "", None)
            body_dict["choices"] = []
            body_dict["usage"] = protocol.usage(num_prompt_tokens, num_tokens).model_dump()
            yield _event(body_dict)
        yield DONE


def _event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _error(kind: str, message: str) -> dict:
    return {"error": {"message": message, "type": kind}}


def _first_message(exc: RequestValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
    message = error.get("msg", "invalid request")
    message = message.removeprefix("Value error, ")
    return f"{location}: {message}" if location else message
