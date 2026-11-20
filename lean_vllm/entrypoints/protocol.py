"""OpenAI-compatible request and response bodies. Unsupported fields are a 400, never silently ignored."""

from time import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Unsupported fields map to their allowed neutral values and rejection reasons.
# None, [] and {} are also accepted for every field listed here.
UNSUPPORTED = {
    "min_p": ((0.0,), "lean-vLLM samples with temperature, top-k and top-p only"),
    "best_of": ((1,), "n > 1 is out of scope"),
    "logprobs": ((False, 0), "the sampler does not return logprobs"),
    "top_logprobs": ((0,), "the sampler does not return logprobs"),
    "presence_penalty": ((0.0,), "penalties are not implemented"),
    "frequency_penalty": ((0.0,), "penalties are not implemented"),
    "repetition_penalty": ((1.0,), "penalties are not implemented"),
    "seed": ((), "the sampler does not take a per-request seed"),
    "logit_bias": ((), "logit processors are not implemented"),
    "tools": ((), "tool calling is not implemented"),
    "tool_choice": (("none",), "tool calling is not implemented"),
    "functions": ((), "tool calling is not implemented"),
    "echo": ((False,), "the prompt is never echoed back"),
    "suffix": ((), "infilling is not implemented"),
}


def _asks_for_it(value, neutral: tuple) -> bool:
    return value is not None and value != [] and value != {} and value not in neutral


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = False


class BaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0)  # 0 is greedy
    top_p: float | None = Field(default=None, gt=0, le=1)  # None disables top-p filtering, like 1.0
    top_k: int | None = Field(default=None, ge=-1)  # None, 0 or -1 disables top-k filtering
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    n: int = 1
    # Not in the OpenAI schema, so clients send these as extra body fields.
    ignore_eos: bool = False
    priority: int = 0
    kv_transfer_params: dict[str, Any] | None = None  # set by the proxy to coordinate prefill and decode

    @model_validator(mode="before")
    @classmethod
    def _refuse_unsupported(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for name, (neutral, reason) in UNSUPPORTED.items():
            if name not in data:
                continue
            if _asks_for_it(data[name], neutral):
                raise ValueError(f"{name} is not supported: {reason}")
            del data[name]  # remove the neutral value before extra="forbid" rejects the field
        return data

    @model_validator(mode="after")
    def _check_n(self) -> "BaseRequest":
        if self.n != 1:
            raise ValueError("n must be 1")
        return self

    @property
    def stop_strings(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)

    @property
    def include_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage


class CompletionRequest(BaseRequest):
    prompt: str | list[int]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseRequest):
    messages: list[ChatMessage] = Field(min_length=1)


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "lean-vllm"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


def usage(num_prompt_tokens: int, num_completion_tokens: int) -> UsageInfo:
    return UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_completion_tokens,
        total_tokens=num_prompt_tokens + num_completion_tokens,
    )


# Chunks serialize with exclude_unset, so fields they must carry have no defaults: one left out raises, not vanishes.


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    finish_reason: str | None
    logprobs: None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo
    # Set by a prefill instance; the proxy forwards it so decode knows where to pull the KV cache.
    kv_transfer_params: dict[str, Any] | None = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo | None = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
    kv_transfer_params: dict[str, Any] | None = None


class DeltaMessage(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None = None
