"""Batch preparation on CPU, including the sequence state sent to TP workers."""

import pickle

import pytest
import torch

from lean_vllm.engine.model_runner import ModelRunner
from lean_vllm.engine.sequence import Sequence
from lean_vllm.sampling_params import SamplingParams
from lean_vllm.utils.context import get_context, reset_context


@pytest.fixture
def runner():
    # Skip model loading and process-group startup; exercise real batch preparation.
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner.device = torch.device("cpu")
    runner.block_size = 8
    yield runner
    reset_context()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("decode", [False, True])
def test_batch_preparation_on_tensor_parallel_ranks(runner, rank, decode):
    seq = Sequence([10, 11, 12], SamplingParams(temperature=0.5))
    seq.num_scheduled_tokens = 3
    if decode:
        seq.append_token(13)
        seq.num_cached_tokens = 3
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
    runner.rank = rank
    if rank:
        seq = pickle.loads(pickle.dumps(seq))

    ids, positions, temperatures, is_prefill = runner.prepare_batch([seq])

    assert ids.tolist() == ([13] if decode else [10, 11, 12])
    assert positions.tolist() == ([3] if decode else [0, 1, 2])
    assert is_prefill == (not decode)
    if decode:
        assert get_context().logits_indices is None
    else:
        assert get_context().logits_indices.tolist() == [2]
    if rank:
        assert temperatures is None
    else:
        assert temperatures.tolist() == [0.5]


@pytest.mark.parametrize("chunked", [False, True])
def test_preemption_recomputes_the_generated_suffix(runner, make_engine, chunked):
    engine = make_engine(num_kvcache_blocks=6, enable_chunked_prefill=chunked)
    first = engine.add(list(range(13)), SamplingParams(max_tokens=16, ignore_eos=True))
    second = engine.add(list(range(100, 108)), SamplingParams(max_tokens=16, ignore_eos=True))
    # The first request needs another block while the second is mid-block.
    for _ in range(16):
        engine.step()
    assert first.is_finished
    assert second.num_preemptions == 1
    assert second.num_tokens == 20

    scheduled = engine.scheduler.schedule().scheduled
    ids, positions, temperatures, is_prefill = runner.prepare_batch(scheduled)

    assert is_prefill
    assert ids.tolist() == [1108, 1109, 1110, 1111]
    assert positions.tolist() == [16, 17, 18, 19]
    assert get_context().logits_indices.tolist() == [3]
    assert temperatures.tolist() == [1.0]
    stepped = engine.scheduler.postprocess(scheduled, [1112])
    assert stepped == [second]
    assert second.num_completion_tokens == 13
    assert second.num_cached_tokens == 20
    engine.run_to_completion()
    assert second.finish_reason == "length"
    assert not engine.scheduler.block_manager.used_block_ids


def test_recomputed_suffix_stays_prefill_across_chunks(runner, make_engine):
    engine = make_engine(num_kvcache_blocks=2, max_num_batched_tokens=2)
    seq = engine.add([10, 11], SamplingParams(max_tokens=8, ignore_eos=True))
    # A requeued sequence retains generated tokens, but may have no cached prefix.
    for token in [20, 21, 22, 23, 24, 25, 26]:
        seq.append_token(token)
    seq.num_preemptions = 1

    for expected_ids, expected_positions in [
        ([10, 11], [0, 1]),
        ([20, 21], [2, 3]),
        ([22, 23], [4, 5]),
        ([24, 25], [6, 7]),
        ([26], [8]),
    ]:
        scheduled = engine.scheduler.schedule().scheduled
        assert scheduled == [seq]
        ids, positions, temperatures, is_prefill = runner.prepare_batch(scheduled)
        assert is_prefill
        assert ids.tolist() == expected_ids
        assert positions.tolist() == expected_positions
        assert len(seq.block_table) == 2    # replay uses the blocks reserved at admission
        last = expected_positions == [8]
        assert temperatures.tolist() == ([1.0] if last else [])
        assert get_context().logits_indices.tolist() == ([0] if last else [])
        stepped = engine.scheduler.postprocess(scheduled, [27] if last else [])
        assert stepped == ([seq] if last else [])

    assert seq.completion_token_ids == [20, 21, 22, 23, 24, 25, 26, 27]
    assert seq.finish_reason == "length"
    assert engine.is_finished()
    assert not engine.scheduler.block_manager.used_block_ids
