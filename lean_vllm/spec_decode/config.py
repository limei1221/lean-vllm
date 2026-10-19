from typing import Annotated, Literal

from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass

PLACEHOLDER_TOKEN_ID = -1  # past a row's last token, and a draft the host does not know yet


@dataclass(config=ConfigDict(extra="forbid"), slots=True)
class SpeculativeConfig:
    """vLLM's --speculative-config, as JSON. Only a checkpoint's own MTP layers draft."""

    method: Literal["mtp", "deepseek_mtp"] = "mtp"  # vLLM's names for a checkpoint's own MTP layers
    # Drafts per request per step; past the MTP layers' count they repeat.
    num_speculative_tokens: Annotated[int, Field(ge=1)] = 1

    @classmethod
    def parse(cls, value: "str | dict | SpeculativeConfig") -> "SpeculativeConfig":
        if isinstance(value, SpeculativeConfig):
            return value
        adapter = TypeAdapter(cls)
        return adapter.validate_json(value or "{}") if isinstance(value, str) else adapter.validate_python(value)


def split_sampled(rows: list[list[int]], num_speculative_tokens: int) -> tuple[list[list[int]], list[list[int]]]:
    """A speculative step's rows, as each one's new tokens and its drafts for the next step.

    The runner sends each row as num_speculative_tokens + 1 columns of tokens, -1 past the last, then the drafts.
    """
    width = num_speculative_tokens + 1
    return [[t for t in row[:width] if t >= 0] for row in rows], [row[width:] for row in rows]
