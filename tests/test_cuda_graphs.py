import torch

from lean_vllm.engine.cuda_graphs import full_graph_buffers, pad_to_bucket, replay_full_graph, smallest_fitting
from lean_vllm.utils.context import Context


def test_a_step_takes_the_smallest_size_that_holds_it():
    sizes = [1, 2, 4, 8]
    assert [smallest_fitting(sizes, n) for n in (1, 3, 4, 5, 8, 9)] == [1, 4, 4, 8, 8, None]
    assert smallest_fitting([], 1) is None


class Graph:
    def __init__(self, buffers: dict):
        self.buffers, self.seen = buffers, None

    def replay(self):
        self.seen = {name: buffer.clone() for name, buffer in self.buffers.items()}
        self.buffers["outputs"].copy_(self.buffers["input_ids"].unsqueeze(1).float())


def test_a_replay_pads_past_the_step_s_rows_with_rows_that_write_and_attend_nothing():
    buffers = full_graph_buffers(4, 2, 3, 64, 1, torch.float32, torch.device("cpu"))
    buffers["slot_mapping"].fill_(7)  # a larger step's, from before
    graph = Graph(buffers)
    replanned = []

    class Backend:
        @staticmethod
        def before_full_graph_replay(context, batch_size):
            replanned.append(batch_size)

    context = Context(
        slot_mapping=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
        context_lens=torch.tensor([5, 9], dtype=torch.int32),
        block_tables=torch.tensor([[2, 0, 0, 9], [3, 4, 0, 9]], dtype=torch.int32),  # one lookahead column past
    )
    out = replay_full_graph(graph, buffers, context, 4, [Backend], input_ids=torch.arange(4), positions=torch.ones(4))
    seen = graph.seen
    assert seen["slot_mapping"].tolist() == [10, 11, 20, 21, -1, -1, -1, -1]
    assert seen["context_lens"].tolist() == [5, 9, 0, 0]
    assert seen["block_tables"].tolist() == [[2, 0, 0], [3, 4, 0], [0, 0, 0], [0, 0, 0]]
    assert seen["positions"][:4].tolist() == [1, 1, 1, 1]
    assert replanned == [4]
    assert out.flatten().tolist() == [0, 1, 2, 3]


def test_a_padded_step_keeps_its_own_token_count_for_attention():
    buffers = dict(input_ids=torch.zeros(8, dtype=torch.int64), positions=torch.zeros(8, dtype=torch.int64))
    context = Context()
    input_ids, positions = pad_to_bucket(
        buffers, 4, context, input_ids=torch.tensor([5, 6, 7]), positions=torch.tensor([0, 1, 2])
    )
    assert input_ids.tolist() == [5, 6, 7, 0] and positions.tolist() == [0, 1, 2, 0]
    assert (context.piecewise_size, context.num_actual_tokens) == (4, 3)
