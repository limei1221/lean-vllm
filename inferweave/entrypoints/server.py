"""Wires the engine, the app and uvicorn together."""

import uvicorn

from inferweave.engine.async_engine import AsyncLLMEngine
from inferweave.entrypoints.api_server import build_app


def run(args, engine_kwargs: dict):
    engine = AsyncLLMEngine.from_engine_args(args.model, **engine_kwargs)
    app = build_app(engine, args.served_model_name or args.model)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level)
    server = uvicorn.Server(config)
    # The CUDA context, KV cache and tensor-parallel children do not survive the
    # engine thread, so there is no in-process restart: shut down and let a
    # supervisor bring the process back.
    engine.on_death = lambda: setattr(server, "should_exit", True)
    server.run()
