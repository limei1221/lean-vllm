import heapq
from abc import ABC, abstractmethod
from collections import deque

from lean_vllm.engine.sequence import Sequence


class SchedulingPolicy(ABC):
    """Owns the waiting queue and picks preemption victims."""

    @staticmethod
    def create(name: str) -> "SchedulingPolicy":
        policies = {policy.name: policy for policy in (Fcfs, Priority)}
        if name not in policies:
            raise ValueError(f"unknown scheduling policy {name!r}, expected one of {sorted(policies)}")
        return policies[name]()

    @abstractmethod
    def add(self, seq: Sequence):
        """A newly arrived request."""

    @abstractmethod
    def requeue(self, seq: Sequence):
        """A preempted request, which has already waited once."""

    @abstractmethod
    def peek(self) -> Sequence: ...

    @abstractmethod
    def pop(self) -> Sequence: ...

    @abstractmethod
    def remove(self, seq: Sequence): ...

    @abstractmethod
    def victim(self, running: deque[Sequence]) -> Sequence:
        """Which running sequence gives up its blocks first."""


class Fcfs(SchedulingPolicy):
    """Arrival order, with preempted sequences going back to the front."""

    name = "fcfs"

    def __init__(self):
        self.queue: deque[Sequence] = deque()

    def add(self, seq):
        self.queue.append(seq)

    def requeue(self, seq):
        self.queue.appendleft(seq)

    def peek(self):
        return self.queue[0]

    def pop(self):
        return self.queue.popleft()

    def remove(self, seq):
        self.queue.remove(seq)

    def victim(self, running):
        return running[-1]    # newest, so the oldest work is preserved

    def __len__(self):
        return len(self.queue)

    def __iter__(self):
        return iter(self.queue)

    def __contains__(self, seq):
        return seq in self.queue


class Priority(SchedulingPolicy):
    """Client-supplied priority, lower first, arrival time breaking ties.

    A preempted sequence re-enters on its own merits: the ordering already
    prefers whoever the client said matters, so there is no front to jump.
    """

    name = "priority"

    def __init__(self):
        self.heap: list[tuple[int, float, int, Sequence]] = []

    @staticmethod
    def _key(seq: Sequence):
        return (seq.priority, seq.arrival_time, seq.seq_id, seq)

    def add(self, seq):
        heapq.heappush(self.heap, self._key(seq))

    requeue = add

    def peek(self):
        return self.heap[0][-1]

    def pop(self):
        return heapq.heappop(self.heap)[-1]

    def remove(self, seq):
        self.heap.remove(self._key(seq))
        heapq.heapify(self.heap)

    def victim(self, running):
        return max(running, key=lambda seq: (seq.priority, seq.arrival_time))

    def __len__(self):
        return len(self.heap)

    def __iter__(self):
        return (entry[-1] for entry in self.heap)

    def __contains__(self, seq):
        return any(entry[-1] is seq for entry in self.heap)
