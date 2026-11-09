from typing import Annotated, Literal

from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass

PLACEHOLDER_TOKEN_ID = -1  # padding after a row's last token, or a draft token not yet known to the host


@dataclass(config=ConfigDict(extra="forbid"), slots=True)
class SpeculativeConfig:
    """Options for --speculative-config, using vLLM's JSON format. Only the checkpoint's own MTP layers draft."""

    method: Literal["mtp", "deepseek_mtp"] = "mtp"  # both names select the checkpoint's own MTP layers
    # Draft tokens per request per step; draft i uses MTP layer i % num_mtp_layers.
    num_speculative_tokens: Annotated[int, Field(ge=1)] = 1

    @classmethod
    def parse(cls, value: "str | dict | SpeculativeConfig") -> "SpeculativeConfig":
        if isinstance(value, SpeculativeConfig):
            return value
        adapter = TypeAdapter(cls)
        return adapter.validate_json(value or "{}") if isinstance(value, str) else adapter.validate_python(value)


def split_sampled(rows: list[list[int]], num_speculative_tokens: int) -> tuple[list[list[int]], list[list[int]]]:
    """Split sampled rows into two lists: new tokens per row and drafts per row for the next step.

    Each row starts with num_speculative_tokens + 1 token slots, padded with -1, followed by draft tokens.
    Drops the -1 padding from the new tokens and keeps the drafts unchanged.
    """
    width = num_speculative_tokens + 1
    return [[t for t in row[:width] if t >= 0] for row in rows], [row[width:] for row in rows]
