from collections import OrderedDict

from lean_vllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class FreeKVCacheBlockQueue:
    """Free blocks, least recently released first.

    A prefix hit takes a block out of the middle in O(1).
    """

    def __init__(self, blocks: list[Block]):
        self._blocks = OrderedDict((block.block_id, block) for block in blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self._blocks)

    def popleft(self) -> Block:
        assert self._blocks, "no free blocks"
        return self._blocks.popitem(last=False)[1]

    def remove(self, block: Block):
        assert block.block_id in self._blocks
        del self._blocks[block.block_id]

    def append(self, block: Block):
        self._blocks[block.block_id] = block

    def get_all_free_blocks(self) -> list[Block]:
        return list(self._blocks.values())


class BlockManager:
    def __init__(
        self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True, recompute_last_hit: bool = False
    ):
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        # The MTP drafter's KV cache at a block's last position depends on the next token, outside the block.
        # Recompute the last matching cached block because that token may differ, as in vLLM's use_eagle logic.
        self.recompute_last_hit = recompute_last_hit
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.used_block_ids: set[int] = set()

    @property
    def usage(self) -> float:
        return len(self.used_block_ids) / len(self.blocks) if self.blocks else 0.0

    def _allocate_block(self) -> int:
        block = self.free_block_queue.popleft()
        block_id = block.block_id
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_queue.append(self.blocks[block_id])

    def _num_blocks(self, seq: Sequence, num_lookahead: int) -> int:
        """Return the number of blocks needed for existing tokens, pending tokens, and additional lookahead slots."""
        return (seq.num_planned_tokens + num_lookahead + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: Sequence, num_lookahead: int = 0) -> int:
        """Return the reusable prefix block count, or -1 if allocation cannot fit. Exclude seq's final block."""
        num_cached_blocks = 0
        num_new_blocks = self._num_blocks(seq, num_lookahead)
        last_hit_held = False
        for i in range(seq.num_blocks - 1) if self.enable_prefix_caching else ():
            block_id = self.hash_to_block_id.get(seq.block_hashes[i], -1)
            if block_id == -1 or self.blocks[block_id].token_ids != seq.block(i):
                break
            num_cached_blocks += 1
            last_hit_held = block_id in self.used_block_ids
            if last_hit_held:
                num_new_blocks -= 1  # already held, so it costs no free block
        if self.recompute_last_hit and num_cached_blocks:
            num_cached_blocks -= 1
            num_new_blocks += last_hit_held
        if self.free_block_queue.num_free_blocks < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int, num_lookahead: int = 0):
        assert not seq.block_table
        for i in range(num_cached_blocks):
            block_id = self.hash_to_block_id[seq.block_hashes[i]]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_queue.remove(block)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(num_cached_blocks, self._num_blocks(seq, num_lookahead)):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        seq.num_published_blocks = num_cached_blocks

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_published_blocks = 0  # the new block table has published nothing
        seq.block_table.clear()

    def _num_missing_blocks(self, seq: Sequence, num_lookahead: int) -> int:
        return max(self._num_blocks(seq, num_lookahead) - len(seq.block_table), 0)

    def can_append(self, seq: Sequence, num_lookahead: int = 0) -> bool:
        """Check for enough free blocks to cover seq's planned tokens plus num_lookahead draft slots."""
        return self.free_block_queue.num_free_blocks >= self._num_missing_blocks(seq, num_lookahead)

    def may_append(self, seq: Sequence, num_lookahead: int = 0):
        for _ in range(self._num_missing_blocks(seq, num_lookahead)):
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence, num_computed_tokens: int):
        """Publish full blocks once their KV is computed and token IDs committed (async or draft tokens can lag)."""
        if not self.enable_prefix_caching:
            return
        end = min(num_computed_tokens, seq.num_tokens) // self.block_size
        for i in range(seq.num_published_blocks, end):
            block = self.blocks[seq.block_table[i]]
            block.update(seq.block_hashes[i], seq.block(i))
            self.hash_to_block_id[block.hash] = block.block_id
        seq.num_published_blocks = max(seq.num_published_blocks, end)
