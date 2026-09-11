import os
from dataclasses import dataclass
from transformers import AutoConfig


# Which kinds of step may replay a graph. Full covers pure decode, piecewise the
# prefill and mixed steps that have to leave attention outside the capture.
FULL_MODES = ("full", "full_and_piecewise")
PIECEWISE_MODES = ("piecewise", "full_and_piecewise")
CUDAGRAPH_MODES = ("none",) + FULL_MODES + ("piecewise",)


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    kvcache_memory_gb: float = 2.0    # cpu/mps only; cuda uses gpu_memory_utilization
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    cudagraph_mode: str = "full_and_piecewise"    # none | full | piecewise | full_and_piecewise
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    enable_chunked_prefill: bool = True    # off is the pre-M2 shape, kept for the A/B
    scheduling_policy: str = "fcfs"    # or "priority"
    max_waiting_requests: int = 0      # 0 is unlimited
    request_timeout: float = 0.0       # seconds a request may wait unscheduled; 0 is none
    long_prefill_token_threshold: int = 0    # per-step token cap for one prompt; 0 is none

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size > 0 and self.kvcache_block_size % 16 == 0, \
            "kvcache_block_size must be a positive multiple of 16"
        assert self.cudagraph_mode in CUDAGRAPH_MODES, f"unknown cudagraph_mode {self.cudagraph_mode!r}"
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
