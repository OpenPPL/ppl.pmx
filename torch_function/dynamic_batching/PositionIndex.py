import torch


class PositionIndex(torch.autograd.Function):
    @staticmethod
    def symbolic(g, sequences: torch.Value,
                seqstarts: torch.Value, start_pos: torch.Value,
                max_seqlen: torch.Value):
        position_idx = g.op("opmx.dynamic_batching::PositionIndex",
                    sequences, seqstarts,
                    start_pos, max_seqlen)
        return position_idx.setTypeAs(sequences)


    @staticmethod
    def forward(self, sequences: torch.Tensor,
                seqstarts: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor):
        if torch.onnx.is_in_onnx_export():
            return torch.zeros_like(sequences)

        position_idx = torch.zeros_like(sequences)
        seqlens = seqstarts[1:] - seqstarts[:-1]
        for b, seqlen in enumerate(seqlens):
            seqbeg = seqstarts[b]
            seqend = seqstarts[b+1]
            position = start_pos[b]

            position_idx[seqbeg:seqend] = torch.arange(
                position, position + seqlen, dtype=torch.int64, device=sequences.device)
        return position_idx


def position_index(sequences: torch.Tensor,
                seqstarts: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor) -> torch.Tensor:
    return PositionIndex.apply(
                    sequences, seqstarts,
                    start_pos, max_seqlen)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()


        def forward(self, sequences: torch.Tensor,
                seqstarts: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor):
            return position_index(
                    sequences, seqstarts,
                    start_pos, max_seqlen)


    test_op1 = TestModule1()

    sequences = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64)
    seqstarts = torch.tensor([0, 1, 3, 9])
    start_pos = torch.tensor([0, 10, 100])
    max_seqlen = torch.tensor(6)

    out = test_op1.forward(sequences, seqstarts, start_pos, max_seqlen)
    print(out)

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (sequences, seqstarts, start_pos, max_seqlen),
        "PositionIndex.onnx", opset_version=11)
    
    print(model_str1)
