import torch

class PixelUnShuffle(torch.autograd.Function):
    @staticmethod
    def symbolic(g, X: torch.Value, scale_factor: float,
                 data_layout: str='nhwc'):
        Y = g.op("opmx::PixelUnShuffle", X,
                 scale_factor_f = scale_factor,
                 data_layout_s = data_layout)
        return Y.setTypeAs(X)


    @staticmethod
    def forward(self, X: torch.Tensor, scale_factor: float, data_layout: str='nhwc'):
        if torch.onnx.is_in_onnx_export():
            n, h, w, c = X.size()
            Y = torch.zeros(n, h*scale_factor, w*scale_factor, c/(scale_factor*scale_factor))
            return Y.type_as(X)
        else:
            assert data_layout == 'nhwc'
            n, h, w, c = X.size()

            # N, H, W, C --> N, H, W * scale, C // scale
            X = X.view(n, h, int(w * scale_factor), int(c / scale_factor))

            # N, H, W * scale, C // scale --> N, W * scale, H, C // scale
            X = X.permute(0, 2, 1, 3).contiguous()

            # N, W * scale, H, C // scale --> N, W * scale, H * scale, C // (scale ** 2)
            X = X.view(n, int(w * scale_factor), int(h * scale_factor),
                       int(c / (scale_factor * scale_factor)))

            # N, W * scale, H * scale, C // (scale ** 2) --> N, H * scale, W * scale, C // (scale ** 2)
            Y = X.permute(0, 2, 1, 3).contiguous()
            return Y


def pixelunshuffle(X: torch.Tensor, scale_factor: float, data_layout: str='nhwc'):
    return PixelUnShuffle.apply(X, scale_factor, data_layout)


if __name__ == "__main__":
    class TestModule(torch.nn.Module):
        def __init__(self, scale_factor: float,
                     data_layout: str='nhwc') -> None:
            super().__init__()
            self.scale_factor = scale_factor
            self.data_layout = data_layout


        def forward(self, X: torch.Tensor):
            return pixelunshuffle(X, self.scale_factor, self.data_layout)

    scale_factor = 0.5
    inputs = torch.rand([1, 32, 32, 3200])
    test_op = TestModule(scale_factor)
    #out = test_op.forward(inputs)

    model_str = torch.onnx.export_to_pretty_string(
                test_op, (inputs), "PixelUnShuffle.onnx", opset_version=11
    )

    print (model_str)
