import torch


class GELU(torch.autograd.Function):
    @staticmethod
    def symbolic(g, X: torch.Value, gate: torch.Value = None, approximate: bool = False):
        if gate is not None:
            Y = g.op("pmx::GELU", X, gate, approximate_i = approximate)
        else:
            Y = g.op("pmx::GELU", X, approximate_i = approximate)
        return Y.setTypeAs(X)


    @staticmethod
    def forward(self, X: torch.Tensor, gate: torch.Tensor = None, approximate: bool = False):
        if torch.onnx.is_in_onnx_export():
            return X
        Y = torch.nn.functional.gelu(X, approximate = "tanh" if approximate else "none")
        if gate is not None:
            return gate * Y
        return Y


def gelu(X: torch.Tensor, gate: torch.Tensor = None, approximate: bool = False):
    return GELU.apply(X, gate, approximate)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, approximate: bool = False) -> None:
            super().__init__()
            self.approximate = approximate


        def forward(self, X: torch.Tensor):
            return gelu(X, approximate=self.approximate)


    test_op1 = TestModule1(True)

    input = torch.ones([8, 4096])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "GELU1.onnx", opset_version=11)
    
    print(model_str1)
