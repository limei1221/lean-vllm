import torch


class SampledTokens:
    """Hold one step's sampled tokens for later retrieval.

    On CUDA, copy tokens to CPU on a separate stream. tolist() waits for that
    copy to finish without waiting for later steps on the compute stream.
    """

    _copy_stream: torch.cuda.Stream | None = None

    @classmethod
    def _get_copy_stream(cls) -> torch.cuda.Stream:
        if cls._copy_stream is None:
            cls._copy_stream = torch.cuda.Stream()
        return cls._copy_stream

    def __init__(self, tokens: torch.Tensor, device: torch.device):
        self._tokens: list | None = None
        if device.type != "cuda":
            self._device_tokens = tokens
            self._event = None
            return
        stream = self._get_copy_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Hold the device tensor until the copy runs, so record_stream() isn't needed.
            self._device_tokens = tokens
            self._host_tokens = torch.empty_like(tokens, device="cpu", pin_memory=True)
            self._host_tokens.copy_(tokens, non_blocking=True)
        self._event = torch.cuda.Event(blocking=True)
        self._event.record(stream)

    def device_tokens(self) -> torch.Tensor:
        """Return the original tensor so the next step can use it without a host round trip."""
        return self._device_tokens

    def tolist(self) -> list:
        """Return and cache the tokens as a Python list.

        Each sampling row holds one token ID, or, from a speculative step, a list of
        verified tokens padded with -1 followed by the drafts (see spec_decode.split_sampled).
        """
        if self._tokens is None:
            if self._event is None:
                self._tokens = self._device_tokens.tolist()
            else:
                self._event.synchronize()
                self._tokens = self._host_tokens.tolist()
        return self._tokens
