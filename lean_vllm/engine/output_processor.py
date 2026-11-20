"""Track each request's detokenizer and optional output stream.

LLMEngine uses this for offline text generation, and AsyncLLM uses it for server requests.
The engine core leaves detokenization to its client.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from transformers import PreTrainedTokenizerFast

from lean_vllm.engine.output import RequestOutput
from lean_vllm.utils.detokenizer import FastIncrementalDetokenizer


class AsyncStream:
    """Queue one request's outputs for its async reader. Write and read on the same event loop."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()

    def put(self, item: RequestOutput | Exception):
        self._queue.put_nowait(item)

    async def __aiter__(self) -> AsyncIterator[RequestOutput]:
        while True:
            item = await self._queue.get()
            if isinstance(item, Exception):
                raise item
            yield item
            if item.finished:
                return


@dataclass(slots=True)
class RequestState:
    detokenizer: FastIncrementalDetokenizer | None  # None without a tokenizer: output text stays unchanged
    stream: AsyncStream | None  # None to return outputs in a list instead of sending them to a stream


class OutputProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerFast | None):
        self.tokenizer = tokenizer
        self.request_states: dict[str, RequestState] = {}

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        skip_special_tokens: bool,
        stream: AsyncStream | None = None,
    ):
        detokenizer = None
        if self.tokenizer is not None:
            detokenizer = FastIncrementalDetokenizer(self.tokenizer, prompt_token_ids, skip_special_tokens)
        self.request_states[request_id] = RequestState(detokenizer, stream)

    def abort_request(self, request_id: str) -> RequestState | None:
        """Pop the request's state, if any; a stream gets a final output so its reader isn't left waiting."""
        state = self.request_states.pop(request_id, None)
        if state is not None and state.stream is not None:
            state.stream.put(RequestOutput(request_id, [], finished=True, finish_reason="abort"))
        return state

    def process_outputs(self, outputs: list[RequestOutput]) -> list[RequestOutput]:
        """Decode tracked outputs in place, then stream or return them; finished requests' state is dropped."""
        unstreamed = []
        for output in outputs:
            state = self.request_states.get(output.request_id)
            if state is None:
                continue  # requests aborted or failed since the step
            if state.detokenizer is not None:
                output.text = "".join(state.detokenizer.decode(token_id) for token_id in output.token_ids)
            if output.finished:
                del self.request_states[output.request_id]
            if state.stream is not None:
                state.stream.put(output)
            else:
                unstreamed.append(output)
        return unstreamed

    def fail_all(self, error: Exception):
        """Clear all request state and queue the error in each stream to be raised when the reader reaches it."""
        states, self.request_states = self.request_states, {}
        for state in states.values():
            if state.stream is not None:
                state.stream.put(error)
