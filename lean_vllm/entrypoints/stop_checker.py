class StopChecker:
    """Checks streamed text for stop strings, holding back a tail where a match could still begin."""

    def __init__(self, stop: list[str]):
        self.stop = [s for s in stop if s]
        self.hold = max((len(s) for s in self.stop), default=1) - 1
        self.buffer = ""
        self.matched = False

    def push(self, text: str) -> str:
        """Returns text safe to emit, cut before the earliest stop string. Sets `matched` if one is found."""
        if not self.stop:
            return text
        self.buffer += text
        cut = min((i for i in (self.buffer.find(s) for s in self.stop) if i >= 0), default=-1)
        if cut >= 0:
            self.matched = True
            emit, self.buffer = self.buffer[:cut], ""
            return emit
        if not self.hold:
            emit, self.buffer = self.buffer, ""
            return emit
        emit, self.buffer = self.buffer[: -self.hold], self.buffer[-self.hold :]
        return emit

    def flush(self) -> str:
        """Returns and clears the held text. Call when generation ends without a stop string match."""
        emit, self.buffer = self.buffer, ""
        return emit
