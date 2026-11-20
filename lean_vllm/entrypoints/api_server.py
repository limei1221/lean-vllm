"""OpenAI-compatible HTTP server using AsyncLLM."""

import asyncio
import json
from collections.abc import AsyncGenerator, Coroutine
from contextlib import aclosing, asynccontextmanager
from time import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel

from lean_vllm.engine.async_llm import AsyncLLM
from lean_vllm.engine.exceptions import EngineDeadError
from lean_vllm.engine.output import RequestOutput
from lean_vllm.engine.scheduler import InvalidRequest
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

# Map engine finish reasons to API finish reasons. Other reasons, such as "capacity", become errors.
FINISH_REASONS = {"stop": "stop", "length": "length", "abort": "stop"}


class _RequestStreamingResponse(StreamingResponse):
    """Abort the request and close the stream even if sending headers or the first chunk fails."""

    def __init__(self, stream, engine: AsyncLLM, request_id: str):
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


def build_app(engine: AsyncLLM, model: str) -> FastAPI:

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

    # Handle rejected requests and engine failures in one place for all routes.
    @app.exception_handler(InvalidRequest)
    async def _invalid_request(request: Request, exc: InvalidRequest):
        return JSONResponse(status_code=400, content=_error("invalid_request_error", str(exc)))

    @app.exception_handler(EngineDeadError)
    async def _engine_dead(request: Request, exc: EngineDeadError):
        return JSONResponse(status_code=503, content=_error("server_error", str(exc)))

    @app.get("/health")
    async def health():
        if engine.is_dead:
            raise HTTPException(503, f"the engine died: {engine.error!r}")
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(await engine.render_metrics(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models():
        return ModelList(data=[ModelCard(id=model)])

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        prompt_token_ids = body.prompt if isinstance(body.prompt, list) else engine.tokenizer.encode(body.prompt)
        return await _serve(engine, model, body, prompt_token_ids, chat=False, request=request)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        messages = [message.model_dump() for message in body.messages]
        # Request token ids instead of a BatchEncoding.
        prompt_token_ids = engine.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        return await _serve(engine, model, body, prompt_token_ids, chat=True, request=request)

    return app


async def _serve(
    engine: AsyncLLM,
    model: str,
    body: BaseRequest,
    prompt_token_ids: list[int],
    chat: bool,
    request: Request | None = None,
):
    if body.model != model:
        raise HTTPException(404, f"the model {body.model!r} does not exist")
    if engine.is_dead:
        raise HTTPException(503, f"the engine died: {engine.error!r}")
    kv_transfer_params = body.kv_transfer_params
    if body.stream and kv_transfer_params and kv_transfer_params.get("do_remote_decode"):
        # A stream can't carry KV transfer params, so no decode instance reads the blocks; they're held until timeout.
        raise HTTPException(400, "do_remote_decode needs a non-streaming request")
    request_id = f"{'chatcmpl' if chat else 'cmpl'}-{uuid4().hex}"
    sampling_params = SamplingParams(
        temperature=body.temperature,
        top_p=1.0 if body.top_p is None else body.top_p,
        top_k=0 if body.top_k is None else body.top_k,
        max_tokens=body.max_tokens,
        ignore_eos=body.ignore_eos,
        priority=body.priority,
        kv_transfer_params=kv_transfer_params,
    )
    outputs = await engine.add_request(prompt_token_ids, sampling_params, request_id)

    deltas = _deltas(outputs, StopChecker(body.stop_strings), engine, request_id)
    if body.stream:
        stream = _stream(deltas, request_id, model, body, len(prompt_token_ids), chat)
        return _RequestStreamingResponse(stream, engine, request_id)
    reply = await _unless_disconnected(request, _collect(deltas, request_id, model, len(prompt_token_ids), chat))
    if reply is None:
        engine.abort(request_id)  # unstarted generators cannot run their finally blocks
        return Response(status_code=499)  # the client has disconnected
    return reply


async def _unless_disconnected(request: Request | None, work: Coroutine):
    """Watch for a disconnect, which Starlette only does while streaming. None if the client left first."""
    if request is None:
        return await work
    task = asyncio.create_task(work)
    listener = asyncio.create_task(_disconnected(request))
    try:
        await asyncio.wait((task, listener), return_when=asyncio.FIRST_COMPLETED)
    finally:
        listener.cancel()
        task.cancel()  # a no-op once done
        await asyncio.wait((task,))
    return None if task.cancelled() else task.result()


async def _disconnected(request: Request):
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def _deltas(
    outputs: AsyncGenerator[RequestOutput, None],
    checker: StopChecker,
    engine: AsyncLLM,
    request_id: str,
):
    """Yield (text, finish_reason, num_completion_tokens, kv_transfer_params).

    Token counts are cumulative; only the last item has a finish reason, from a stop-string match or the engine.
    """
    num_tokens = 0
    try:
        async for output in outputs:
            num_tokens += len(output.token_ids)
            text = checker.push(output.text)
            if checker.matched:
                engine.abort(request_id, "stop")  # ask the core to record a normal finish and release the blocks
                yield text, "stop", num_tokens, None
                return
            if output.finished:
                if output.finish_reason not in FINISH_REASONS:
                    raise HTTPException(503, f"the engine dropped the request: {output.finish_reason}")
                reason = FINISH_REASONS[output.finish_reason]
                yield text + checker.flush(), reason, num_tokens, output.kv_transfer_params
                return
            if text:
                yield text, None, num_tokens, None
    finally:
        await outputs.aclose()


async def _collect(deltas, request_id: str, model: str, num_prompt_tokens: int, chat: bool):
    text, finish_reason, num_tokens, kv_transfer_params = "", "stop", 0, None
    async for delta, reason, num_tokens, kv_transfer_params in deltas:  # noqa: B007 - the last values are kept
        text += delta
        finish_reason = reason or finish_reason
    usage = protocol.usage(num_prompt_tokens, num_tokens)
    if chat:
        message = protocol.ChatMessage(role="assistant", content=text)
        choice = protocol.ChatCompletionResponseChoice(index=0, message=message, finish_reason=finish_reason)
        return protocol.ChatCompletionResponse(
            id=request_id,
            model=model,
            choices=[choice],
            usage=usage,
            kv_transfer_params=kv_transfer_params,
        )
    text_choice = protocol.CompletionResponseChoice(index=0, text=text, finish_reason=finish_reason, logprobs=None)
    return protocol.CompletionResponse(
        id=request_id,
        model=model,
        choices=[text_choice],
        usage=usage,
        kv_transfer_params=kv_transfer_params,
    )


async def _stream(deltas, request_id: str, model: str, body: BaseRequest, num_prompt_tokens: int, chat: bool):
    created = int(time())
    response: type[BaseModel]
    if chat:
        response, kind = protocol.ChatCompletionStreamResponse, "chat.completion.chunk"
    else:
        response, kind = protocol.CompletionStreamResponse, "text_completion"

    def chunk(choices: list, **extra) -> str:
        # Each response class requires its matching kind literal, but mypy cannot infer that pairing.
        return _event(response(id=request_id, object=kind, created=created, model=model, choices=choices, **extra))  # type: ignore[arg-type]

    def choice(text: str, reason: str | None, role: Literal["assistant"] | None = None):
        if not chat:
            return protocol.CompletionResponseChoice(index=0, text=text, finish_reason=reason, logprobs=None)
        delta = protocol.DeltaMessage(role=role, content=text) if role else protocol.DeltaMessage(content=text)
        return protocol.ChatCompletionResponseStreamChoice(index=0, delta=delta, finish_reason=reason)

    async with aclosing(deltas):
        num_tokens = 0
        if chat:
            yield chunk([choice("", None, role="assistant")])
        try:
            async for delta, reason, num_tokens, _ in deltas:  # noqa: B007 - the last count is kept
                yield chunk([choice(delta, reason)])
        except (EngineDeadError, HTTPException) as error:
            # The HTTP 200 status has already been sent, so send the error as a stream event.
            yield _event(_error("server_error", str(getattr(error, "detail", error))))
            yield DONE
            return
        if body.include_usage:
            yield chunk([], usage=protocol.usage(num_prompt_tokens, num_tokens))
        yield DONE


def _event(payload: BaseModel | dict) -> str:
    data = payload.model_dump_json(exclude_unset=True) if isinstance(payload, BaseModel) else json.dumps(payload)
    return f"data: {data}\n\n"


def _error(kind: str, message: str) -> dict:
    return {"error": {"message": message, "type": kind}}


def _first_message(exc: RequestValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
    message = error.get("msg", "invalid request")
    message = message.removeprefix("Value error, ")
    return f"{location}: {message}" if location else message
