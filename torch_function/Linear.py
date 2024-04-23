import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g: torch._C.Graph, X: torch.Value, W: torch.Value, B: torch.Value,
        in_features: int, out_features: int):
        print("symbolic")
        if B is not None:
            Y = g.op("opmx::Linear", X, W, B,
                    in_features_i = in_features,
                    out_features_i = out_features,
                    bias_term_i = True)
        else:
            Y = g.op("opmx::Linear", X, W,
                    in_features_i = in_features,
                    out_features_i = out_features,
                    bias_term_i = False)
        return Y


    @staticmethod
    def forward(
        self, X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, in_features: int, out_features: int):
        if torch.onnx.is_in_onnx_export():
            Y = torch.zeros(*X.shape[:-1], W.shape[0], dtype=W.dtype).to(X.device)
            return Y
        else:
            Y = F.linear(X, W, B)
            return Y


def linear(
    X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, in_features: int, out_features: int):
    return Linear.apply(X, W, B, in_features, out_features)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias_term: bool = True) -> None:
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            
            self.weight = nn.Parameter(torch.ones(self.out_features, self.in_features))
            if bias_term:
                self.bias = nn.Parameter(torch.zeros(self.out_features))
            else:
                self.register_parameter("bias", None)
        
        def forward(self, X: torch.Tensor):
            return linear(X, self.weight, self.bias, self.in_features, self.out_features)
        
    test_op1 = TestModule1(1024, 4096, True)
    input = torch.ones([8, 1024])
    
    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "Linear.onnx", opset_version=11)
    
    print(model_str1)
