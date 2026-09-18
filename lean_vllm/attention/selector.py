from lean_vllm import envs
from lean_vllm.attention.abstract import AttentionBackend
from lean_vllm.attention.flash_backend import FlashAttention3Backend
from lean_vllm.attention.flashmla_backend import FlashMLABackend
from lean_vllm.attention.torch_backend import TorchAttention

# TorchAttention is last and always available, so resolution cannot fail.
BACKENDS: tuple[type[AttentionBackend], ...] = (
    FlashAttention3Backend,
    TorchAttention,
)
# Tried first for MLA models only, so a plain model never reports a kernel it does not run.
MLA_BACKENDS: tuple[type[AttentionBackend], ...] = (
    FlashMLABackend,
)


def get_attention_backend(name: str | None = None, mla: bool = False) -> type[AttentionBackend]:
    """Resolve a backend: explicit name, then $LEAN_VLLM_ATTENTION_BACKEND, then first available."""
    name = name or envs.LEAN_VLLM_ATTENTION_BACKEND
    if name:
        by_name = {backend.get_name(): backend for backend in (*MLA_BACKENDS, *BACKENDS)}
        if name not in by_name:
            raise ValueError(f"unknown attention backend {name!r}, expected one of {sorted(by_name)}")
        backend = by_name[name]
        if not backend.is_available():
            raise RuntimeError(f"attention backend {name!r} was requested but is not available on this machine")
        return backend
    candidates = (*MLA_BACKENDS, *BACKENDS) if mla else BACKENDS
    return next(backend for backend in candidates if backend.is_available())
