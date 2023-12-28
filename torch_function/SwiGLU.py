import torch


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def symbolic(g, X: torch.Value, beta: float = 1.0):
        Y = g.op("pmx::SwiGLU", X, beta_f = beta)
        return Y


    @staticmethod
    def forward(self, X: torch.Tensor, beta: float = 1.0):
        if torch.onnx.is_in_onnx_export():
            return torch.zeros(*X.shape[:-1], X.shape[-1]//2, dtype=X.dtype).to(X.device)
        x, g = torch.split(X, [X.shape[-1]//2, X.shape[-1]//2], -1)
        Y = g * x * torch.nn.functional.sigmoid(beta * x)
        return Y


def swiglu(X: torch.Tensor, beta: float = 1.0):
    return SwiGLU.apply(X, beta)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, beta: float = 1.0) -> None:
            super().__init__()
            self.beta = beta


        def forward(self, X: torch.Tensor):
            return swiglu(X, approximate=self.beta)


    test_op1 = TestModule1(True)

    input = torch.ones([8, 4096])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "SwiGLU1.onnx", opset_version=11)
    
    print(model_str1)
