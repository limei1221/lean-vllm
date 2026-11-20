"""Start the engine and serve the HTTP app with Uvicorn."""

import uvicorn

from lean_vllm.engine.async_llm import AsyncLLM
from lean_vllm.entrypoints.api_server import build_app


def run(args, engine_kwargs: dict):
    engine = AsyncLLM.from_engine_args(args.model, **engine_kwargs)  # starts the engine core in a separate process
    app = build_app(engine, args.served_model_name or args.model)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level)
    server = uvicorn.Server(config)
    # CUDA state and TP workers die with the engine, so stop the server and let a supervisor restart it.
    engine.on_death = lambda: setattr(server, "should_exit", True)
    server.run()
