"""Incremental detokenization must never split a multi-byte character."""

import os

import pytest

from inferweave.utils.detokenizer import IncrementalDetokenizer

MODEL = os.path.expanduser(os.getenv("INFERWEAVE_TEST_MODEL", "~/huggingface/Qwen3-0.6B"))
SPECIAL = 256


class ByteTokenizer:
    """Byte-level tokenizer, so every non-ASCII character spans several tokens."""

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        tokens = []
        for i in ids:
            if i < SPECIAL:
                tokens.append(bytes([i]))
            elif not skip_special_tokens:
                tokens.append(b"<|s|>")    # real tokenizers render a marker, not a byte
        return tokens

    def convert_tokens_to_string(self, tokens):
        return b"".join(tokens).decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


@pytest.fixture
def detokenizer():
    return IncrementalDetokenizer(ByteTokenizer(), [])


def deltas(detokenizer, text: str) -> list[str]:
    return [detokenizer.decode(i) for i in ByteTokenizer().encode(text)]


class TestIncrementalDetokenizer:

    def test_ascii_arrives_one_character_at_a_time(self, detokenizer):
        assert deltas(detokenizer, "hey") == ["h", "e", "y"]

    def test_two_byte_character_is_held_back_until_complete(self, detokenizer):
        assert deltas(detokenizer, "é") == ["", "é"]

    def test_four_byte_character_is_held_back_until_complete(self, detokenizer):
        assert deltas(detokenizer, "🙂") == ["", "", "", "🙂"]

    def test_deltas_concatenate_to_the_whole_string(self, detokenizer):
        text = "héllo 🙂 wörld"
        assert "".join(deltas(detokenizer, text)) == text
        assert detokenizer.text == text

    def test_replacement_character_is_never_emitted(self, detokenizer):
        assert "�" not in "".join(deltas(detokenizer, "naïve 🚀"))

    def test_special_tokens_are_skipped(self, detokenizer):
        assert detokenizer.decode(SPECIAL) == ""

    def test_special_tokens_are_kept_when_asked(self):
        detokenizer = IncrementalDetokenizer(ByteTokenizer(), [], skip_special_tokens=False)
        assert detokenizer.decode(ord("!")) == "!"
        assert detokenizer.decode(SPECIAL) == "<|s|>"

    def test_prompt_tokens_are_context_not_output(self):
        tokenizer = ByteTokenizer()
        detokenizer = IncrementalDetokenizer(tokenizer, tokenizer.encode("prompt"))
        assert deltas(detokenizer, "!") == ["!"]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, use_fast=True)


@pytest.mark.skipif(not os.path.isdir(MODEL), reason=f"no tokenizer at {MODEL}")
class TestAgainstRealTokenizer:
    """The fake proves the offset logic; this proves the contract against transformers."""

    @pytest.mark.parametrize("text", ["Hello, world!", "café 🙂 naïve", "日本語のテキスト", "def f(x):\n    return x"])
    def test_matches_a_full_decode(self, tokenizer, text):
        token_ids = tokenizer.encode(text)
        detokenizer = IncrementalDetokenizer(tokenizer, [])
        assert "".join(detokenizer.decode(i) for i in token_ids) == tokenizer.decode(token_ids)
