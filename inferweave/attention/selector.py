import os

from inferweave.attention.abstract import AttentionBackend
from inferweave.attention.flash_backend import FlashAttentionBackend
from inferweave.attention.torch_backend import TorchAttention

ENV_VAR = "INFERWEAVE_ATTENTION_BACKEND"

# TorchAttention is last and always available, so resolution cannot fail.
BACKENDS: tuple[type[AttentionBackend], ...] = (
    FlashAttentionBackend,
    TorchAttention,
)


def get_attention_backend(name: str | None = None) -> type[AttentionBackend]:
    """Resolve a backend: explicit name, then $INFERWEAVE_ATTENTION_BACKEND, then first available."""
    name = name or os.getenv(ENV_VAR)
    if name:
        by_name = {backend.get_name(): backend for backend in BACKENDS}
        if name not in by_name:
            raise ValueError(f"unknown attention backend {name!r}, expected one of {sorted(by_name)}")
        backend = by_name[name]
        if not backend.is_available():
            raise RuntimeError(f"attention backend {name!r} was requested but is not available on this machine")
        return backend
    return next(backend for backend in BACKENDS if backend.is_available())
