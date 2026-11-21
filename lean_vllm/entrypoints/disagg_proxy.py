"""Proxy requests through separate prefill and decode servers.

For each completions or chat completions request, first call a prefill server with max_tokens=1, streaming disabled and
do_remote_decode set. Forward the original request to a decode server, adding any kv_transfer_params returned by
prefill so decode can load the prompt's KV. Return the decode response to the client. Select prefill and decode servers
independently in round-robin order.
"""

import itertools
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


def build_proxy_app(
    prefill_urls: list[str], decode_urls: list[str], client: httpx.AsyncClient | None = None
) -> FastAPI:
    prefill_urls = [url.rstrip("/") for url in prefill_urls]
    decode_urls = [url.rstrip("/") for url in decode_urls]
    prefills, decodes = itertools.cycle(prefill_urls), itertools.cycle(decode_urls)
    # Disable timeouts for the default client so long generations can finish.
    client = client or httpx.AsyncClient(timeout=None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="lean-vLLM disaggregated proxy", lifespan=lifespan)

    @app.get("/health")
    async def health():
        for url in prefill_urls + decode_urls:
            try:
                ok = (await client.get(f"{url}/health")).status_code == 200
            except httpx.HTTPError:
                ok = False
            if not ok:
                return JSONResponse({"error": {"message": f"{url} is unhealthy", "type": "server_error"}}, 503)
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return _relay(await client.get(f"{next(decodes)}/v1/models"))

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _disaggregate(client, next(prefills), next(decodes), "/v1/completions", await request.json())

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _disaggregate(client, next(prefills), next(decodes), "/v1/chat/completions", await request.json())

    return app


async def _disaggregate(client: httpx.AsyncClient, prefill_url: str, decode_url: str, path: str, body: dict):
    prefill_body = dict(body, max_tokens=1, stream=False, kv_transfer_params={"do_remote_decode": True})
    prefill_body.pop("stream_options", None)
    prefill = await client.post(prefill_url + path, json=prefill_body)
    if prefill.status_code != 200:
        return _relay(prefill)  # return the prefill error without calling decode
    kv_transfer_params = prefill.json().get("kv_transfer_params")
    # If prefill returns no transfer params (for example, after an abort), forward the original request body.
    decode_body = dict(body, kv_transfer_params=kv_transfer_params) if kv_transfer_params else body
    upstream = await client.send(client.build_request("POST", decode_url + path, json=decode_body), stream=True)
    if upstream.status_code != 200 or not body.get("stream"):
        await upstream.aread()
        await upstream.aclose()
        return _relay(upstream)
    return _UpstreamStreamingResponse(upstream)


class _UpstreamStreamingResponse(StreamingResponse):
    """Close the decode response when streaming ends, including on a client disconnect.

    Closing an unfinished response lets the decode server detect the disconnect and abort generation.
    """

    def __init__(self, upstream: httpx.Response):
        super().__init__(upstream.aiter_raw(), media_type=upstream.headers.get("content-type"))
        self.upstream = upstream

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.upstream.aclose()


def _relay(response: httpx.Response) -> Response:
    return Response(response.content, response.status_code, media_type=response.headers.get("content-type"))
