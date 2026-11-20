"""CLI for `lean-vllm serve` and `lean-vllm proxy`, with serve's engine flags generated from `Config`."""

import argparse
from dataclasses import MISSING, fields

from lean_vllm.config import Config

# These fields have no generated flags: model is positional; the others are loaded
# from the checkpoint or parsed from JSON settings.
INTERNAL = {"model", "hf_config", "eos", "kv_transfer", "speculative"}


def add_engine_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("engine")
    for field in fields(Config):
        default = field.default
        if field.name in INTERNAL or default is MISSING or default is None:
            continue  # excluded field, or no default value to infer its type from
        flag = "--" + field.name.replace("_", "-")
        if isinstance(default, bool):
            group.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            group.add_argument(flag, type=type(default), default=default)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="lean-vllm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("model", help="path to a local model directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--served-model-name", default=None, help="the id reported by /v1/models")
    serve.add_argument("--log-level", default="info")
    add_engine_args(serve)
    proxy = subparsers.add_parser("proxy", help="split each request between prefill and decode servers")
    proxy.add_argument("--prefill", nargs="+", required=True, help="prefill server URLs, kv_role producer or both")
    proxy.add_argument("--decode", nargs="+", required=True, help="decode server URLs, kv_role consumer or both")
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=int, default=8000)
    proxy.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    if args.command == "proxy":
        import uvicorn

        from lean_vllm.entrypoints.disagg_proxy import build_proxy_app

        uvicorn.run(
            build_proxy_app(args.prefill, args.decode), host=args.host, port=args.port, log_level=args.log_level
        )
        return

    from lean_vllm.entrypoints.server import run  # load the `serve` extra's dependencies after parsing arguments

    engine_kwargs = {
        field.name: getattr(args, field.name)
        for field in fields(Config)
        if field.name not in INTERNAL and hasattr(args, field.name)
    }
    run(args, engine_kwargs)


if __name__ == "__main__":
    main()
