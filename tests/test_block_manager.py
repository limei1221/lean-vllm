"""The free queue's eviction order: what the cache gives up first when blocks run short."""

import pytest

from lean_vllm.engine.block_manager import BlockManager, FreeBlockQueue
from lean_vllm.engine.sequence import Sequence
from lean_vllm.sampling_params import SamplingParams

BLOCK_SIZE = 4


@pytest.fixture
def manager():
    Sequence.block_size = BLOCK_SIZE
    return BlockManager(num_blocks=6, block_size=BLOCK_SIZE)


def cache(manager: BlockManager, token_ids: list[int]) -> list[int]:
    """Admit a sequence, hash its full blocks, release it, and report the blocks it held."""
    seq = Sequence(token_ids, SamplingParams())
    manager.allocate(seq, manager.can_allocate(seq))
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    manager.hash_blocks(seq)
    block_table = list(seq.block_table)
    manager.deallocate(seq)
    return block_table


class TestFreeQueue:

    def test_taking_a_block_out_of_the_middle_leaves_the_order_alone(self):
        """A prefix hit takes its blocks back from wherever they sit."""
        queue = FreeBlockQueue(range(5))
        queue.remove(2)
        assert list(queue) == [0, 1, 3, 4]
        queue.append(2)
        assert list(queue) == [0, 1, 3, 4, 2]
        assert queue.popleft() == 0


class TestEvictionOrder:

    def test_blocks_that_cache_nothing_are_spent_first(self, manager):
        cache(manager, list(range(12)))    # blocks 0, 1 and 2, then released
        assert list(manager.free_block_ids) == [3, 4, 5, 2, 1, 0]

    def test_a_sequence_gives_up_its_deepest_block_first(self, manager):
        """Which is what makes a shared prefix outlive the suffixes built on it."""
        held = cache(manager, list(range(12)))
        taken = [manager._allocate_block() for _ in range(6)]
        assert taken[3:] == list(reversed(held))

    def test_cached_contents_survive_until_the_block_is_claimed(self, manager):
        """Eviction is lazy: a freed block keeps its hash while it waits."""
        cache(manager, list(range(12)))
        surviving = []
        for _ in range(6):
            manager._allocate_block()
            surviving.append(len(manager.hash_to_block_id))
        assert surviving == [3, 3, 3, 2, 1, 0]
