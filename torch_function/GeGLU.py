import torch


class GeGLU(torch.autograd.Function):
    @staticmethod
    def symbolic(g, X: torch.Value, approximate: bool = False):
        Y = g.op("opmx::GeGLU", X, approximate_i = approximate)
        return Y


    @staticmethod
    def forward(self, X: torch.Tensor, approximate: bool = False):
        if torch.onnx.is_in_onnx_export():
            return torch.zeros(*X.shape[:-1], X.shape[-1]//2, dtype=X.dtype).to(X.device)
        x, g = torch.split(X, [X.shape[-1]//2, X.shape[-1]//2], -1)
        Y = g * torch.nn.functional.gelu(x, approximate = "tanh" if approximate else "none")
        return Y


def geglu(X: torch.Tensor, approximate: bool = False):
    return GeGLU.apply(X, approximate)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, approximate: bool = False) -> None:
            super().__init__()
            self.approximate = approximate


        def forward(self, X: torch.Tensor):
            return geglu(X, approximate=self.approximate)


    test_op1 = TestModule1(True)

    input = torch.ones([8, 4096])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "GeGLU1.onnx", opset_version=11)
    
    print(model_str1)
