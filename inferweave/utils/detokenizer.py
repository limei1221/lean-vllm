# Window of already-decoded tokens kept as context, so a piece that only makes
# sense after its predecessors still decodes correctly. vLLM uses 5.
INITIAL_OFFSET = 5


class IncrementalDetokenizer:
    """Decodes one token at a time without re-decoding the prefix.

    Holds back output that ends mid-character, so a multi-byte character split
    across tokens is never emitted as U+FFFD.
    """

    def __init__(self, tokenizer, prompt_token_ids: list[int], skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.tokens = tokenizer.convert_ids_to_tokens(prompt_token_ids)
        self.prefix_offset = max(len(self.tokens) - INITIAL_OFFSET, 0)
        self.read_offset = len(self.tokens)
        self.text = ""

    def decode(self, token_id: int) -> str:
        """Append one token and return the text it completes, possibly empty."""
        self.tokens.extend(
            self.tokenizer.convert_ids_to_tokens([token_id], skip_special_tokens=self.skip_special_tokens)
        )
        prefix = self.tokenizer.convert_tokens_to_string(self.tokens[self.prefix_offset:self.read_offset])
        whole = self.tokenizer.convert_tokens_to_string(self.tokens[self.prefix_offset:])
        if len(whole) <= len(prefix) or whole.endswith("�"):
            return ""    # incomplete character; wait for the next token
        delta = whole[len(prefix):]
        self.prefix_offset, self.read_offset = self.read_offset, len(self.tokens)
        self.text += delta
        return delta
