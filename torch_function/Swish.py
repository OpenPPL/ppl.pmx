import torch


class Swish(torch.autograd.Function):
    @staticmethod
    def symbolic(g, X: torch.Value, gate: torch.Value = None, beta: float = 1.0):
        if gate is not None:
            Y = g.op("opmx::Swish", X, gate, beta_f = beta)
        else:
            Y = g.op("opmx::Swish", X, beta_f = beta)
        return Y.setTypeAs(X)


    @staticmethod
    def forward(self, X: torch.Tensor, gate: torch.Tensor = None, beta: float = 1.0):
        if torch.onnx.is_in_onnx_export():
            return X
        Y = X * torch.nn.functional.sigmoid(beta * X)
        if gate is not None:
            return gate * Y
        return Y


def swish(X: torch.Tensor, gate: torch.Tensor = None, beta: float = 1.0):
    return Swish.apply(X, gate, beta)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def forward(self, X: torch.Tensor):
            return swish(X, X, 0.141)


    test_op1 = TestModule1()

    input = torch.ones([8, 4096])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "Swish1.onnx", opset_version=11)
    
    print(model_str1)
