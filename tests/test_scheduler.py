"""Pins the scheduler's behaviour before Project 1 rewrites it. See docs/online-serving.md.

Several tests assert limitations rather than features. Each says so, and names
the milestone that is expected to change it.
"""

import pytest

from inferweave.sampling_params import SamplingParams

FOREVER = SamplingParams(max_tokens=64, ignore_eos=True)


def prompt(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


class TestGeneration:

    def test_every_request_returns_max_tokens(self, make_engine):
        engine = make_engine()
        seqs = [engine.add(prompt(8, s * 100), SamplingParams(max_tokens=5, ignore_eos=True)) for s in range(4)]
        outputs = engine.run_to_completion()
        assert len(outputs) == 4
        assert all(len(outputs[seq.request_id]) == 5 for seq in seqs)

    def test_eos_stops_generation_and_is_kept(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=64))
        engine.model_runner.eos_after[seq.request_id] = 3
        outputs = engine.run_to_completion()
        assert len(outputs[seq.request_id]) == 4    # three tokens, then the eos itself

    def test_ignore_eos_runs_to_max_tokens(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=5, ignore_eos=True))
        engine.model_runner.eos_after[seq.request_id] = 1
        assert len(engine.run_to_completion()[seq.request_id]) == 5

    def test_finished_sequences_free_their_blocks(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        engine.run_to_completion()
        block_manager = engine.scheduler.block_manager
        assert not block_manager.used_block_ids
        assert len(block_manager.free_block_ids) == engine.config.num_kvcache_blocks


class TestStreaming:

    def test_a_token_is_reported_every_step(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        engine.step()    # prefill also samples the first token
        assert [(o.token_ids, o.finished) for o in engine.step()] == [([1001], False)]
        assert engine.step()[0].finished

    def test_a_partial_prefill_reports_nothing(self, make_engine):
        engine = make_engine(max_num_batched_tokens=16)
        engine.add(prompt(40), FOREVER)
        assert engine.step() == []
        assert engine.step() == []
        assert len(engine.step()) == 1    # third chunk completes the prompt


class TestFinishReason:

    def test_max_tokens_is_length(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=2, ignore_eos=True))
        engine.run_to_completion()
        assert seq.finish_reason == "length"

    def test_eos_is_stop(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=64))
        engine.model_runner.eos_after[seq.request_id] = 2
        engine.run_to_completion()
        assert seq.finish_reason == "stop"

    def test_a_client_stop_token_is_honoured_even_with_ignore_eos(self, make_engine):
        """ignore_eos covers the eos token only, as in vLLM."""
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=64, ignore_eos=True, stop_token_ids=[1002]))
        engine.run_to_completion()
        assert seq.finish_reason == "stop" and seq.completion_token_ids == [1000, 1001, 1002]


class TestMetrics:

    def test_timestamps_are_ordered(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        engine.run_to_completion()
        m = seq.metrics()
        assert m.arrival_time <= m.first_scheduled_time <= m.first_token_time <= m.finish_time
        assert m.queue_time >= 0 and m.ttft >= 0 and m.e2e >= m.ttft
        assert m.num_prompt_tokens == 8 and m.num_completion_tokens == 3

    def test_tpot_needs_more_than_one_token(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=1, ignore_eos=True))
        engine.run_to_completion()
        assert seq.metrics().tpot is None

    def test_preemptions_are_counted(self, make_engine):
        engine = make_engine(num_kvcache_blocks=2, kvcache_block_size=8)
        engine.add(prompt(8), FOREVER)
        second = engine.add(prompt(8, 100), FOREVER)
        engine.step()
        engine.step()
        assert second.num_preemptions == 1


class TestAbort:

    def test_aborting_a_waiting_request_frees_nothing_and_drops_it(self, make_engine):
        engine = make_engine()
        first = engine.add(prompt(8), FOREVER)
        second = engine.add(prompt(8, 100), FOREVER)
        assert engine.scheduler.abort(second.request_id)
        engine.step()
        assert [request_id for request_id, _ in engine.model_runner.batches[0][1]] == [first.request_id]
        assert second.finish_reason == "abort"

    def test_aborting_a_running_request_frees_its_blocks(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(16), FOREVER)
        engine.step()
        blocks = list(seq.block_table)
        assert engine.scheduler.abort(seq.request_id)
        assert seq.block_table == []
        block_manager = engine.scheduler.block_manager
        assert not block_manager.used_block_ids.intersection(blocks)
        assert engine.is_finished()

    def test_aborting_an_unknown_request_is_a_no_op(self, make_engine):
        engine = make_engine()
        assert not engine.scheduler.abort("nope")

    def test_aborting_a_finished_request_is_a_no_op(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=2, ignore_eos=True))
        engine.run_to_completion()
        assert not engine.scheduler.abort(seq.request_id)


class TestBatching:

    def test_prefill_batches_many_sequences(self, make_engine):
        engine = make_engine()
        for s in range(4):
            engine.add(prompt(8, s * 100), FOREVER)
        engine.step()
        is_prefill, batch = engine.model_runner.batches[0]
        assert is_prefill and [n for _, n in batch] == [8, 8, 8, 8]

    def test_decode_schedules_one_token_per_sequence(self, make_engine):
        engine = make_engine()
        for s in range(3):
            engine.add(prompt(8, s * 100), FOREVER)
        engine.step()
        engine.step()
        is_prefill, batch = engine.model_runner.batches[1]
        assert not is_prefill and [n for _, n in batch] == [1, 1, 1]

    def test_max_num_seqs_caps_the_batch(self, make_engine):
        engine = make_engine(max_num_seqs=2)
        for s in range(5):
            engine.add(prompt(8, s * 100), FOREVER)
        engine.step()
        assert len(engine.model_runner.batches[0][1]) == 2

    def test_arriving_prefill_starves_running_decodes(self, make_engine):
        """The reason this project exists: prefill returns first, unconditionally."""
        engine = make_engine()
        running = engine.add(prompt(8), FOREVER)
        engine.step()
        engine.add(prompt(8, 100), FOREVER)
        engine.step()
        is_prefill, batch = engine.model_runner.batches[1]
        assert is_prefill
        assert running.request_id not in [request_id for request_id, _ in batch]    # M2 mixes them


class TestChunkedPrefill:

    def test_long_prompt_is_split_across_steps(self, make_engine):
        engine = make_engine(max_num_batched_tokens=16)
        seq = engine.add(prompt(40), FOREVER)
        engine.step()
        assert engine.model_runner.batches[0] == (True, [(seq.request_id, 16)])
        assert seq in engine.scheduler.waiting    # not running until prefill completes
        engine.step()
        engine.step()
        assert [n for _, n in engine.model_runner.batches[2][1]] == [8]
        assert seq in engine.scheduler.running

    def test_only_the_first_sequence_may_be_chunked(self, make_engine):
        """Limitation: leftover budget goes unused rather than partly prefilling. M2 fixes."""
        engine = make_engine(max_num_batched_tokens=20)
        engine.add(prompt(8), FOREVER)
        engine.add(prompt(40, 100), FOREVER)
        engine.step()
        assert [n for _, n in engine.model_runner.batches[0][1]] == [8]    # 12 tokens of budget wasted


class TestPrefixCache:

    def test_repeated_prefix_is_not_recomputed(self, make_engine):
        engine = make_engine()
        first = engine.add(prompt(16), FOREVER)
        engine.step()
        second = engine.add(prompt(16), FOREVER)
        engine.step()
        # One full block hits; the trailing block is never a candidate, so 8 of 16 recompute.
        assert [n for _, n in engine.model_runner.batches[1][1]] == [8]
        assert second.block_table[0] == first.block_table[0]
        assert engine.scheduler.block_manager.blocks[first.block_table[0]].ref_count == 2


class TestPreemption:

    def test_victim_is_preempted_and_requeued(self, make_engine):
        engine = make_engine(num_kvcache_blocks=2, kvcache_block_size=8)
        first = engine.add(prompt(8), FOREVER)
        second = engine.add(prompt(8, 100), FOREVER)
        engine.step()
        engine.step()
        assert second.block_table == [] and second.num_cached_tokens == 0    # recompute, not swap
        assert second in engine.scheduler.waiting
        assert engine.model_runner.batches[1][1] == [(first.request_id, 1)]

    def test_preempted_sequence_still_completes(self, make_engine):
        engine = make_engine(num_kvcache_blocks=3, kvcache_block_size=8)
        for s in range(2):
            engine.add(prompt(8, s * 100), SamplingParams(max_tokens=12, ignore_eos=True))
        outputs = engine.run_to_completion()
        assert [len(o) for o in outputs.values()] == [12, 12]

    def test_preempting_the_last_sequence_crashes(self, make_engine):
        """Known bug: an empty decode batch trips `assert scheduled_seqs`. M2 turns this green."""
        engine = make_engine(num_kvcache_blocks=1, kvcache_block_size=8)
        engine.add(prompt(8), FOREVER)
        engine.step()
        with pytest.raises(AssertionError):
            engine.step()
