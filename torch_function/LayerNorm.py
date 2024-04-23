import torch


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, weight: torch.Value, bias: torch.Value,
        elementwise_affine: bool = False, axis: int = -1, eps: float = 1e-5):
        if elementwise_affine:
            Y = g.op("opmx::LayerNorm", X, weight, bias,
                        elementwise_affine_i = True,
                        axis_i = axis, eps_f = eps, skip_term_i = False)
        else:
            Y = g.op("opmx::LayerNorm", X,
                        elementwise_affine_i = False,
                        axis_i = axis, eps_f = eps, skip_term_i = False)
        return Y.setTypeAs(X)


    @staticmethod
    def forward(
        self, X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
        elementwise_affine: bool = False, axis: int = -1, eps: float = 1e-5):
        if torch.onnx.is_in_onnx_export():
            return X
        x = X.float()
        mean = x.mean(axis, keepdim=True)
        var = x.var(axis, keepdim=True)
        Y = (x - mean) * torch.rsqrt(var + eps)
        Y = Y.type_as(X)
        if elementwise_affine:
            Y = Y * weight + bias
        return Y


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, weight: torch.Value,
        bias: torch.Value, SkipIn: torch.Value = None,
        elementwise_affine: bool = False, axis: int = -1, eps: float = 1e-5):
        if SkipIn is None:
            if elementwise_affine:
                Y, SkipOut = g.op("opmx::LayerNorm", X, weight, bias,
                            elementwise_affine_i = True,
                            axis_i = axis, eps_f = eps, skip_term_i = True,
                            outputs = 2)
            else:
                Y, SkipOut = g.op("opmx::LayerNorm", X,
                            elementwise_affine_i = False,
                            axis_i = axis, eps_f = eps, skip_term_i = True,
                            outputs = 2)
        else:
            Y, SkipOut = g.op("opmx::LayerNorm", X, weight, bias, SkipIn,
                        elementwise_affine_i = elementwise_affine,
                        axis_i = axis, eps_f = eps, skip_term_i = True,
                        outputs = 2)
        return Y.setTypeAs(X), SkipOut.setTypeAs(X)


    @staticmethod
    def forward(
        self, X: torch.Tensor, weight: torch.Tensor,
        bias: torch.Tensor, SkipIn: torch.Tensor,
        elementwise_affine: bool = False, axis: int = -1, eps: float = 1e-5):
        if torch.onnx.is_in_onnx_export():
            return X, X
        if SkipIn is None:
            SkipOut = X
        else:
            SkipOut = X + SkipIn
        x = SkipOut.float()
        mean = x.mean(axis, keepdim=True)
        var = x.var(axis, keepdim=True)
        Y = (x - mean) * torch.rsqrt(var + eps)
        Y = Y.type_as(X)
        if elementwise_affine:
            Y = Y * weight + bias
        return Y, SkipOut


def layer_norm(X: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None,
                axis: int = -1, eps: float = 1e-5) -> torch.Tensor:
    if weight is None or bias is None:
        elementwise_affine = False
    else:
        elementwise_affine = True
    return LayerNorm.apply(X, weight, bias, elementwise_affine, axis, eps)


def skip_layer_norm(X: torch.Tensor, weight: torch.Tensor = None,
                    bias: torch.Tensor = None, SkipIn: torch.Tensor = None,
                    axis: int = -1, eps: float = 1e-5) -> torch.Tensor:
    if weight is None or bias is None:
        W = torch.empty(0, device=X.device)
        B = torch.empty(0, device=X.device)
        elementwise_affine = False
    else:
        W = weight
        B = bias
        elementwise_affine = True
    return SkipLayerNorm.apply(X, W, B, SkipIn, elementwise_affine, axis, eps)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.bias = torch.nn.Parameter(torch.zeros(dim))


        def forward(self, X: torch.Tensor):
            return layer_norm(X, self.weight, self.bias, -1, self.eps)


    class TestModule2(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.bias = torch.nn.Parameter(torch.zeros(dim))


        def forward(self, X: torch.Tensor, SkipIn: torch.Tensor):
            return skip_layer_norm(X, self.weight, self.bias, SkipIn, -1, self.eps)


    test_op1 = TestModule1(4096, 1e-6)
    test_op2 = TestModule2(4096, 1e-6)

    input = torch.ones([8, 4096])
    skip = torch.ones([8, 4096])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "LayerNorm1.onnx", opset_version=11)
    model_str2 = torch.onnx.export_to_pretty_string(
        test_op2, (input, skip), "LayerNorm2.onnx", opset_version=11)

    print(model_str1)
    print(model_str2)
