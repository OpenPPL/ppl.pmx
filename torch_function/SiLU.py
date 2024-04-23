import torch


class SiLU(torch.autograd.Function):
    @staticmethod
    def symbolic(g, X: torch.Value, gate: torch.Value = None):
        if gate is not None:
            Y = g.op("opmx::SiLU", X, gate)
        else:
            Y = g.op("opmx::SiLU", X)
        return Y.setTypeAs(X)


    @staticmethod
    def forward(self, X: torch.Tensor, gate: torch.Tensor = None):
        if torch.onnx.is_in_onnx_export():
            return X
        Y = torch.nn.functional.silu(X)
        if gate is not None:
            return gate * Y
        return Y


def silu(X: torch.Tensor, gate: torch.Tensor = None):
    return SiLU.apply(X, gate)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def forward(self, X: torch.Tensor):
            return silu(X, X)


    test_op1 = TestModule1()

    input = torch.ones([8, 4096])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "SiLU1.onnx", opset_version=11)
    
    print(model_str1)
