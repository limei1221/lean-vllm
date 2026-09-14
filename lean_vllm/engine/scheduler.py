from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

from lean_vllm.config import Config
from lean_vllm.engine.sequence import Sequence, SequenceStatus
from lean_vllm.engine.block_manager import BlockManager
from lean_vllm.engine.policy import SchedulingPolicy


@dataclass(slots=True)
class SchedulerOutput:
    """What one step should run. A sequence carries its own num_scheduled_tokens."""
    scheduled: list[Sequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    dropped: list[Sequence] = field(default_factory=list)    # finished without ever sampling
    # Counted while scheduling: advance() clears num_scheduled_tokens.
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_queried_blocks: int = 0    # prefix cache, counted at admission
    num_cached_blocks: int = 0

    def __bool__(self):
        return bool(self.scheduled)


@dataclass(slots=True)
class LaunchedRow:
    """One sampling row of a launched step. A requeue after launch voids its token."""
    seq: Sequence
    num_preemptions: int


class QueueFull(Exception):
    """The waiting queue is at max_waiting_requests. The server answers 429."""


class InvalidRequest(Exception):
    """The request can never run as asked. The server answers 400."""


class DuplicateRequestId(InvalidRequest):
    """Another request with this id is still in flight, and seqs holds only one."""


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.max_waiting_requests = config.max_waiting_requests
        self.request_timeout = config.request_timeout
        self.long_prefill_token_threshold = config.long_prefill_token_threshold
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size, config.enable_prefix_caching
        )
        self.waiting = SchedulingPolicy.create(config.scheduling_policy)
        self.running: deque[Sequence] = deque()
        self.seqs: dict[str, Sequence] = {}    # live requests, for abort

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if seq.request_id in self.seqs:
            raise DuplicateRequestId(f"{seq.request_id} is already in flight")
        if self.max_waiting_requests and len(self.waiting) >= self.max_waiting_requests:
            raise QueueFull(f"{len(self.waiting)} requests already waiting")
        self.seqs[seq.request_id] = seq
        self.waiting.add(seq)

    def abort(self, request_id: str, reason: str = "abort") -> bool:
        """Drop a request between steps. Returns False if it already finished."""
        seq = self.seqs.pop(request_id, None)
        if seq is None:
            return False
        queue = self.waiting if seq.status == SequenceStatus.WAITING else self.running
        queue.remove(seq)    # both queues expose remove()
        seq.drop_pending()
        self.block_manager.deallocate(seq)
        self._finish(seq, reason)
        return True

    def schedule(self) -> SchedulerOutput:
        """One token budget per step, running sequences first so decode is never starved."""
        dropped = self._expire_waiting()
        if not self.enable_chunked_prefill:
            output = self._schedule_whole_prompts()
            output.dropped = dropped + output.dropped
            if output:
                return output    # prefill-only step
            dropped = output.dropped    # nothing to run, but the drops still owe an output
        output = SchedulerOutput(dropped=dropped)
        budget = self.max_num_batched_tokens
        still_running: deque[Sequence] = deque()
        # Most urgent first, so a victim is never one already served this step.
        self.running = self.waiting.by_urgency(self.running)

        while self.running:
            seq = self.running.popleft()
            if budget <= 0 or len(output.scheduled) >= self.max_num_seqs:
                still_running.append(seq)    # left untouched this step
                continue
            if seq.num_planned_tokens - seq.num_prompt_tokens >= seq.max_tokens:
                still_running.append(seq)    # its reserved tokens already reach the limit
                continue
            if not seq.is_prefill:    # decoding, so the cache grows
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
                if seq.num_blocks > len(self.block_manager.blocks):
                    self.waiting.pop()    # impossible even with the entire cache free
                    self._drop(seq, "capacity", output)
                    continue
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                budget -= self._admit(seq, num_cached_blocks, budget, output)

        return output

    def _expire_waiting(self) -> list[Sequence]:
        """Drop requests that waited past request_timeout without ever running.

        Preempted sequences are kept: shedding them would waste their work.
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
            if seq.num_blocks > len(self.block_manager.blocks):
                self.waiting.pop()
                self._drop(seq, "capacity", output)
                continue
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
        # A preempted request also prefills its generated suffix until it samples again.
        num_tokens = min(seq.num_tokens - seq.num_cached_tokens, budget) if seq.is_prefill else 1
        if seq.is_prefill and self.enable_chunked_prefill and self.long_prefill_token_threshold:
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

    def _make_room(self, seq: Sequence, still_running: deque[Sequence], output: SchedulerOutput) -> bool:
        """Free blocks for one more decoded token. False if seq itself gave way.

        A prefill chunk needs none: its blocks were all taken at admission.
        """
        while not self.block_manager.can_append(seq):
            if self.running:
                victim = self.waiting.victim(self.running)
                self.running.remove(victim)
                self._preempt(victim, output)
            elif still_running or output.scheduled:
                self._preempt(seq, output)
                return False
            elif seq.num_pending_tokens:
                still_running.append(seq)    # its in-flight token may stop it; decide after reconcile
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
        seq.drop_pending()    # the in-flight token is discarded and recomputed
        self.block_manager.deallocate(seq)
        self.waiting.requeue(seq)
        output.preempted.append(seq)

    def _finish(self, seq: Sequence, reason: str):
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        seq.finish_time = perf_counter()

    def _drop(self, seq: Sequence, reason: str, output: SchedulerOutput | None = None):
        self._finish(seq, reason)
        seq.drop_pending()
        del self.seqs[seq.request_id]
        if output is not None:
            output.dropped.append(seq)    # the caller is still owed a final output

    def advance(self, seqs: list[Sequence]) -> list[LaunchedRow]:
        """Move bookkeeping forward with no token values. Returns the sampling rows.

        Rows follow the sampler's order, so reconcile() can zip them with token ids.
        """
        rows = []
        for seq in seqs:
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # Before the skip, so a chunked prefill publishes each block as it lands.
            self.block_manager.hash_blocks(seq, seq.num_cached_tokens)
            if seq.num_cached_tokens < seq.num_planned_tokens:
                continue    # prefill or recomputation unfinished, so this row samples nothing
            seq.is_prefill = False
            seq.reserve_token()
            rows.append(LaunchedRow(seq, seq.num_preemptions))
        return rows

    def reconcile(self, rows: list[LaunchedRow], token_ids: list[int]) -> list[Sequence]:
        """Commit the sampled tokens and run the stop checks. Returns the rows that produced one."""
        stepped = []
        for row, token_id in zip(rows, token_ids):
            seq = row.seq
            if seq.is_finished or seq.num_preemptions != row.num_preemptions:
                continue    # aborted, finished or requeued since the launch; the token is void
            seq.commit_token(token_id)
            if seq.first_token_time is None:
                seq.first_token_time = perf_counter()    # when the token reaches the host, not at launch
            stepped.append(seq)
            if (token_id == self.eos and not seq.ignore_eos) or token_id in seq.stop_token_ids:
                reason = "stop"    # ignore_eos covers the eos token only, not client stop tokens
            elif seq.num_completion_tokens == seq.max_tokens:
                reason = "length"
            else:
                continue
            seq.drop_pending()    # a later step may already have reserved one
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
            self._drop(seq, reason)
        return stepped
