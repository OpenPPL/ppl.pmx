import torch


class InsertEmbedding(torch.autograd.Function):
    @staticmethod
    def symbolic(g, X: torch.Value, P: torch.Value, offset: torch.Value):
        return g.op("pmx::InsertEmbedding", X, P, offset)


    @staticmethod
    def forward(self, X: torch.Tensor, P: torch.Tensor, offset: torch.Tensor):
        if torch.onnx.is_in_onnx_export():
            new_shape = X.shape
            new_shape[-2] = new_shape[-2] + P.shape[-2]
            return torch.zeros(new_shape, dtype=X.dtype).to(X.device)
        offset = int(offset)
        X_1, X_2 = X[:, :offset], X[:, offset:]
        if P.dim() == 2:
            Y = torch.concat([X_1, P.expand(X.shape[0], -1, -1), X_2], dim=-2)
        else:
            Y = torch.concat([X_1, P, X_2], dim=-2)
        return Y


def insert_embedding(X: torch.Tensor, P: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return InsertEmbedding.apply(X, P, offset)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()


        def forward(self, X: torch.Tensor, P: torch.Tensor, offset: torch.Tensor):
            return insert_embedding(X, P, offset)


    test_op1 = TestModule1()

    input = torch.ones([2, 8, 8]).cumsum(1).to(torch.float16)
    patch = torch.ones([4, 8]).cumsum(0).to(torch.float16)
    offset = torch.tensor(4)

    # out = test_op1.forward(input, patch, offset)
    # print(input)
    # print(patch)
    # print(offset)
    # print(out)

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input, patch, offset), "InsertEmbedding.onnx", opset_version=11)
    
    print(model_str1)
