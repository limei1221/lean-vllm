from lean_vllm import envs
from lean_vllm.attention.abstract import AttentionBackend, LayerSpec
from lean_vllm.attention.flash_backend import FlashAttention3Backend
from lean_vllm.attention.flashinfer_backend import FlashInferBackend
from lean_vllm.attention.flashmla_backend import FlashMLABackend
from lean_vllm.attention.torch_backend import TorchAttention

# Automatic selection tries available, compatible backends in this order. FlashMLA supports only MLA layers.
# FA3 also supports MLA, with the layer expanding latents into keys and values even during decode.
# TorchAttention is the last fallback, but still checks the layer's dtype and head counts.
BACKENDS: tuple[type[AttentionBackend], ...] = (
    FlashMLABackend,
    FlashAttention3Backend,
    FlashInferBackend,
    TorchAttention,
)


def get_attention_backend(spec: LayerSpec, name: str | None = None) -> type[AttentionBackend]:
    """Return a backend class for one layer.

    Use a nonempty name first, then LEAN_VLLM_ATTENTION_BACKEND. A requested backend raises ValueError if
    unknown or RuntimeError if unavailable or incompatible. With neither name set, return the first available,
    compatible backend in BACKENDS; raise RuntimeError with each backend's reasons if none qualifies.
    """
    name = name or envs.LEAN_VLLM_ATTENTION_BACKEND
    if name:
        by_name = {backend.get_name(): backend for backend in BACKENDS}
        if name not in by_name:
            raise ValueError(f"unknown attention backend {name!r}, expected one of {sorted(by_name)}")
        backend = by_name[name]
        reasons = _unsupported(backend, spec)
        if reasons:
            raise RuntimeError(
                f"attention backend {name!r} was requested but cannot serve {spec}: {'; '.join(reasons)}"
            )
        return backend
    rejected = []
    for backend in BACKENDS:
        reasons = _unsupported(backend, spec)
        if not reasons:
            return backend
        rejected.append(f"{backend.get_name()}: {'; '.join(reasons)}")
    raise RuntimeError(f"no attention backend can serve {spec}: {' | '.join(rejected)}")


def _unsupported(backend: type[AttentionBackend], spec: LayerSpec) -> list[str]:
    if not backend.is_available():
        return ["not available on this machine"]
    return backend.validate(spec)
