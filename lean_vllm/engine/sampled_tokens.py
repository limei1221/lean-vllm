import torch


class SampledTokens:
    """One launched step's sampled tokens, fetched without awaiting later steps.

    tolist() on a device tensor is a blocking copy on the default stream, so it
    waits for everything queued ahead of it -- including the next step, once the
    pipeline has launched it. The copy goes on its own stream instead and the
    wait is on its event, so awaiting step k-1 never waits for step k.
    """

    _copy_stream: torch.cuda.Stream | None = None

    @classmethod
    def _get_copy_stream(cls) -> torch.cuda.Stream:
        if cls._copy_stream is None:
            cls._copy_stream = torch.cuda.Stream()
        return cls._copy_stream

    def __init__(self, tokens: torch.Tensor, device: torch.device):
        self._tokens: list[int] | None = None
        if device.type != "cuda":
            self._device_tokens = tokens
            self._event = None
            return
        stream = self._get_copy_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Keep the device tensor alive until the copy has run.
            # No record_stream() on tokens: safe only because self holds this
            # reference past the event, keeping the allocator from reusing it early.
            self._device_tokens = tokens
            self._host_tokens = torch.empty_like(tokens, device="cpu", pin_memory=True)
            self._host_tokens.copy_(tokens, non_blocking=True)
        self._event = torch.cuda.Event(blocking=True)
        self._event.record(stream)

    def device_tokens(self) -> torch.Tensor:
        """The tensor itself, for filling the next step's inputs without a host round trip."""
        return self._device_tokens

    def tolist(self) -> list[int]:
        if self._tokens is None:
            if self._event is None:
                self._tokens = self._device_tokens.tolist()
            else:
                self._event.synchronize()
                self._tokens = self._host_tokens.tolist()
        return self._tokens
