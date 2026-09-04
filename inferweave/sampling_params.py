from dataclasses import dataclass, field


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0    # 0 is greedy
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_token_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        assert self.temperature >= 0
