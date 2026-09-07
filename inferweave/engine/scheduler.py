from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

from inferweave.config import Config
from inferweave.engine.sequence import Sequence, SequenceStatus
from inferweave.engine.block_manager import BlockManager
from inferweave.engine.policy import SchedulingPolicy


@dataclass(slots=True)
class SchedulerOutput:
    """What one step should run. A sequence carries its own num_scheduled_tokens."""
    scheduled: list[Sequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    dropped: list[Sequence] = field(default_factory=list)    # finished without ever sampling
    # Counted while scheduling: postprocess() clears num_scheduled_tokens.
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_queried_blocks: int = 0    # prefix cache, counted at admission
    num_cached_blocks: int = 0

    def __bool__(self):
        return bool(self.scheduled)


class QueueFull(Exception):
    """The waiting queue is at max_waiting_requests. The server answers 429."""


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.max_waiting_requests = config.max_waiting_requests
        self.request_timeout = config.request_timeout
        self.max_num_partial_prefills = config.max_num_partial_prefills
        self.long_prefill_token_threshold = config.long_prefill_token_threshold
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting = SchedulingPolicy.create(config.scheduling_policy)
        self.running: deque[Sequence] = deque()
        self.seqs: dict[str, Sequence] = {}    # live requests, for abort

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if self.max_waiting_requests and len(self.waiting) >= self.max_waiting_requests:
            raise QueueFull(f"{len(self.waiting)} requests already waiting")
        self.seqs[seq.request_id] = seq
        self.waiting.add(seq)

    def abort(self, request_id: str) -> bool:
        """Drop a request between steps. Returns False if it already finished."""
        seq = self.seqs.pop(request_id, None)
        if seq is None:
            return False
        queue = self.waiting if seq.status == SequenceStatus.WAITING else self.running
        queue.remove(seq)    # both queues expose remove()
        self.block_manager.deallocate(seq)
        self._finish(seq, "abort")
        return True

    def schedule(self) -> SchedulerOutput:
        """One token budget per step, running sequences first so decode is never starved."""
        dropped = self._expire_waiting()
        if not self.enable_chunked_prefill:
            output = self._schedule_whole_prompts()
            output.dropped = dropped + output.dropped
            if output:
                return output    # prefill-only step, as before M2
            dropped = output.dropped    # nothing to run, but the drops still owe an output
        output = SchedulerOutput(dropped=dropped)
        budget = self.max_num_batched_tokens
        still_running: deque[Sequence] = deque()

        while self.running:
            seq = self.running.popleft()
            if budget <= 0 or len(output.scheduled) >= self.max_num_seqs:
                still_running.append(seq)    # left untouched this step
                continue
            if seq.num_cached_tokens >= seq.num_prompt_tokens:    # decoding, so the cache grows
                if not self._make_room(seq, still_running, output):
                    continue
                self.block_manager.may_append(seq)
            budget -= self._schedule(seq, budget, output)
            still_running.append(seq)
        self.running = still_running

        # Admitting new work while under memory pressure would only preempt again.
        if not output.preempted:
            while self.waiting and len(output.scheduled) < self.max_num_seqs and budget > 0:
                seq = self.waiting.peek()
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                if self._would_chunk(seq, num_cached_blocks, budget) and self._partial_prefills_full():
                    break
                budget -= self._admit(seq, num_cached_blocks, budget, output)

        return output

    def _expire_waiting(self) -> list[Sequence]:
        """Drop requests that have waited past request_timeout without ever running.

        Only ones that never ran: a preempted sequence has tokens to show for
        itself, and shedding it would throw that work away for nothing.
        """
        if not self.request_timeout:
            return []
        deadline = perf_counter() - self.request_timeout
        expired = [
            seq for seq in self.waiting
            if seq.first_scheduled_time is None and seq.arrival_time < deadline
        ]
        for seq in expired:
            self.waiting.remove(seq)
            self._drop(seq, "timeout")
        return expired

    def _schedule_whole_prompts(self) -> SchedulerOutput:
        """Chunked prefill disabled: whole prompts only, and never mixed with decode."""
        output = SchedulerOutput()
        budget = self.max_num_batched_tokens
        while self.waiting and len(output.scheduled) < self.max_num_seqs:
            seq = self.waiting.peek()
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                break
            num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            if num_tokens > self.max_num_batched_tokens:
                self.waiting.pop()    # will never fit in one step, and splitting is off
                self._drop(seq, "capacity", output)
                continue
            if num_tokens > budget:
                break
            budget -= self._admit(seq, num_cached_blocks, budget, output)
        return output

    def _admit(self, seq: Sequence, num_cached_blocks: int, budget: int, output: SchedulerOutput) -> int:
        """Move the head of the waiting queue into the running set."""
        self.waiting.pop()
        self.block_manager.allocate(seq, num_cached_blocks)
        output.num_queried_blocks += seq.num_blocks
        output.num_cached_blocks += num_cached_blocks
        seq.status = SequenceStatus.RUNNING
        num_tokens = self._schedule(seq, budget, output)
        self.running.append(seq)
        return num_tokens

    def _schedule(self, seq: Sequence, budget: int, output: SchedulerOutput) -> int:
        """Give seq its share of the budget: a prompt chunk, or one decoded token."""
        seq.is_prefill = seq.num_cached_tokens < seq.num_prompt_tokens
        num_tokens = min(seq.num_tokens - seq.num_cached_tokens, budget) if seq.is_prefill else 1
        if seq.is_prefill and self.long_prefill_token_threshold:
            num_tokens = min(num_tokens, self.long_prefill_token_threshold)
        seq.num_scheduled_tokens = num_tokens
        if seq.first_scheduled_time is None:
            seq.first_scheduled_time = perf_counter()
        output.scheduled.append(seq)
        if seq.is_prefill:
            output.num_prefill_tokens += num_tokens
        else:
            output.num_decode_tokens += 1
        return num_tokens

    def _would_chunk(self, seq: Sequence, num_cached_blocks: int, budget: int) -> bool:
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
        return num_tokens > budget or 0 < self.long_prefill_token_threshold < num_tokens

    def _partial_prefills_full(self) -> bool:
        if not self.max_num_partial_prefills:
            return False
        partial = sum(1 for seq in self.running if seq.num_cached_tokens < seq.num_prompt_tokens)
        return partial >= self.max_num_partial_prefills

    def _make_room(self, seq: Sequence, still_running: deque[Sequence], output: SchedulerOutput) -> bool:
        """Free blocks for one more decoded token. False if seq itself gave way.

        A prompt chunk needs no room: its blocks were all taken at admission.
        """
        while not self.block_manager.can_append(seq):
            if self.running:
                victim = self.waiting.victim(self.running)
                self.running.remove(victim)
                self._preempt(victim, output)
            elif still_running or output.scheduled:
                self._preempt(seq, output)
                return False
            else:
                # Alone in the cache and still short of a block: it can never fit.
                self.block_manager.deallocate(seq)
                self._drop(seq, "capacity", output)
                return False
        return True

    def _preempt(self, seq: Sequence, output: SchedulerOutput):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_preemptions += 1
        self.block_manager.deallocate(seq)
        self.waiting.requeue(seq)
        output.preempted.append(seq)

    def _finish(self, seq: Sequence, reason: str):
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        seq.finish_time = perf_counter()

    def _drop(self, seq: Sequence, reason: str, output: SchedulerOutput | None = None):
        self._finish(seq, reason)
        del self.seqs[seq.request_id]
        if output is not None:
            output.dropped.append(seq)    # the caller is still owed a final output

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[Sequence]:
        """Returns the sequences that produced a token; a partial prefill produces none."""
        stepped = []
        tokens = iter(token_ids)    # only the rows that sampled produced one
        for seq in seqs:
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_tokens:
                continue    # prompt not finished, so no logits for this sequence
            token_id = next(tokens)
            seq.append_token(token_id)
            if seq.first_token_time is None:
                seq.first_token_time = perf_counter()
            stepped.append(seq)
            if (token_id == self.eos and not seq.ignore_eos) or token_id in seq.stop_token_ids:
                reason = "stop"    # ignore_eos covers the eos token only, not client stop tokens
            elif seq.num_completion_tokens == seq.max_tokens:
                reason = "length"
            else:
                continue
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
            self._drop(seq, reason)
        return stepped
