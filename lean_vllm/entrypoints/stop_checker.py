class StopChecker:
    """Applies `stop` strings to streamed text.

    String matching needs detokenized text, so it lives in the frontend rather
    than the engine. A stop string can span token boundaries, so the tail is
    held back up to the longest stop string minus one character — a deliberate
    small addition to streaming latency.
    """

    def __init__(self, stop: list[str]):
        self.stop = [s for s in stop if s]
        self.hold = max((len(s) for s in self.stop), default=1) - 1
        self.buffer = ""
        self.matched = False

    def push(self, text: str) -> str:
        """The text safe to emit now. Sets `matched` when a stop string hits."""
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
        emit, self.buffer = self.buffer[:-self.hold], self.buffer[-self.hold:]
        return emit

    def flush(self) -> str:
        """Generation ended with no match, so the held tail was real output."""
        emit, self.buffer = self.buffer, ""
        return emit
