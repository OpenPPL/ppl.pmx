import torch


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, weight: torch.Value,
        axis: int = -1, eps: float = 1e-5):
        Y = g.op("opmx::RMSNorm", X, weight,
                    axis_i = axis, eps_f = eps, skip_term_i = False)
        return Y.setTypeAs(X)


    @staticmethod
    def forward(
        self, X: torch.Tensor, weight: torch.Tensor,
        axis: int = -1, eps: float = 1e-5):
        if torch.onnx.is_in_onnx_export():
            return X
        x = X.float()
        mean_square = (x * x).mean(axis, keepdim=True)
        Y = x * torch.rsqrt(mean_square + eps)
        return Y.type_as(X) * weight


class SkipRMSNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, weight: torch.Value, SkipIn: torch.Value = None,
        axis: int = -1, eps: float = 1e-5):
        if SkipIn is None:
            Y, SkipOut = g.op("opmx::RMSNorm", X, weight,
                        axis_i = axis, eps_f = eps, skip_term_i = True,
                        outputs = 2)
        else:
            Y, SkipOut = g.op("opmx::RMSNorm", X, weight, SkipIn,
                        axis_i = axis, eps_f = eps, skip_term_i = True,
                        outputs = 2)
        return Y.setTypeAs(X), SkipOut.setTypeAs(X)


    @staticmethod
    def forward(
        self, X: torch.Tensor, weight: torch.Tensor, SkipIn: torch.Tensor = None,
        axis: int = -1, eps: float = 1e-5) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            return X, X
        if SkipIn is None:
            SkipOut = X
        else:
            SkipOut = X + SkipIn
        x = SkipOut.float()
        mean_square = x.pow(2).mean(axis, keepdim=True)
        Y = x * torch.rsqrt(mean_square + eps)
        Y = Y.type_as(X) * weight
        return Y, SkipOut


def rms_norm(X: torch.Tensor, weight: torch.Tensor,
        axis: int = -1, eps: float = 1e-5) -> torch.Tensor:
    return RMSNorm.apply(X, weight, axis, eps)


def skip_rms_norm(X: torch.Tensor, weight: torch.Tensor, SkipIn: torch.Tensor = None,
        axis: int = -1, eps: float = 1e-5) -> torch.Tensor:
    return SkipRMSNorm.apply(X, weight, SkipIn, axis, eps)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))


        def forward(self, X: torch.Tensor):
            return rms_norm(X, self.weight, -1, self.eps)


    class TestModule2(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))


        def forward(self, X: torch.Tensor, SkipIn: torch.Tensor):
            return skip_rms_norm(X, self.weight, SkipIn, -1, self.eps)


    test_op1 = TestModule1(4096, 1e-6)
    test_op2 = TestModule2(4096, 1e-6)

    input = torch.ones([8, 4096])
    skip = torch.ones([8, 4096])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "RMSNorm1.onnx", opset_version=11)
    model_str2 = torch.onnx.export_to_pretty_string(
        test_op2, (input, skip), "RMSNorm2.onnx", opset_version=11)

    print(model_str1)
    print(model_str2)
