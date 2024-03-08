import torch


class InsertEmbedding(torch.autograd.Function):
    @staticmethod
    def symbolic(g, X: torch.Value, P: torch.Value,
                seqstarts: torch.Value, outstarts: torch.Value,
                patstarts: torch.Value, offset: torch.Value,
                max_outlen: torch.Value):
        return g.op("pmx.dynamic_batching::InsertEmbedding",
                    X, P, seqstarts,
                    outstarts, patstarts,
                    offset, max_outlen)


    @staticmethod
    def forward(self, X: torch.Tensor, P: torch.Tensor,
                seqstarts: torch.Tensor, outstarts: torch.Tensor,
                patstarts: torch.Tensor, offset: torch.Tensor,
                max_outlen: torch.Tensor):
        if torch.onnx.is_in_onnx_export():
            pat_batch = patstarts.numel() - 1
            if pat_batch == 1:
                in_batch = seqstarts.numel() - 1
                return torch.zeros((X.shape[0] + in_batch * P.shape[0], X.shape[1]),
                                   dtype=X.dtype).to(X.device)
            else:
                return torch.zeros((X.shape[0] + P.shape[0], X.shape[1]),
                                   dtype=X.dtype).to(X.device)

        boradcast_offset = offset.numel() == 1
        broadcast_pat = (patstarts.numel() - 1) == 1

        Y = torch.zeros(outstarts[-1], X.shape[1], dtype=X.dtype, device=X.device)

        seqlens = seqstarts[1:] - seqstarts[:-1]
        for b, seqlen in enumerate(seqlens):
            cur_offset = int(offset) if boradcast_offset else offset[b]
            seqbeg = seqstarts[b]
            seqend = seqstarts[b+1]
            patbeg = patstarts[0] if broadcast_pat else patstarts[b]
            patend = patstarts[1] if broadcast_pat else  patstarts[b+1]
            outbeg = outstarts[b]
            outend = outstarts[b+1]

            _X = X[seqbeg:seqend]
            X_1, X_2 = _X[:cur_offset], _X[cur_offset:]
            Y[outbeg:outend] = torch.concat([X_1, P[patbeg:patend], X_2], dim=-2)
        return Y


def insert_embedding(X: torch.Tensor, P: torch.Tensor,
                seqstarts: torch.Tensor, outstarts: torch.Tensor,
                patstarts: torch.Tensor, offset: torch.Tensor,
                max_outlen: torch.Tensor) -> torch.Tensor:
    return InsertEmbedding.apply(
                    X, P, seqstarts,
                    outstarts, patstarts,
                    offset, max_outlen)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()


        def forward(self, X: torch.Tensor, P: torch.Tensor,
                seqstarts: torch.Tensor, outstarts: torch.Tensor,
                patstarts: torch.Tensor, offset: torch.Tensor,
                max_outlen: torch.Tensor):
            return insert_embedding(
                    X, P, seqstarts,
                    outstarts, patstarts,
                    offset, max_outlen)


    test_op1 = TestModule1()

    input = torch.ones([2 * 8, 8]).cumsum(0).to(torch.float16)
    patch = torch.ones([4, 8]).cumsum(0).to(torch.float16)
    offset = torch.tensor([2, 4])
    seqstarts = torch.tensor([0, 8, 16])
    patstarts = torch.tensor([0, 4])
    outstarts = torch.tensor([0, 12, 24])
    max_outlen = torch.tensor(12)

    # out = test_op1.forward(input, patch, seqstarts, outstarts, patstarts, offset, max_outlen)
    # print(input)
    # print(patch)
    # print(offset)
    # print(out)

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input, patch, seqstarts, outstarts, patstarts, offset, max_outlen),
        "InsertEmbedding.onnx", opset_version=11)
    
    print(model_str1)
