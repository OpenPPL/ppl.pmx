import torch

from typing import Sequence, Union


class Reshape(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input: torch.Value, shape: torch.Value):
        reshaped = g.op("pmx::Reshape", input, shape)
        return reshaped


    @staticmethod
    def forward(self, input: torch.Tensor, shape: torch.Tensor):
        _shape = shape.tolist()
        for i in range(shape.numel()):
            if _shape[i] == 0:
                _shape[i] = input.shape[i]
        return torch.reshape(input, _shape)


def reshape(input: torch.Tensor, shape: Sequence[Union[int, torch.SymInt]]) -> torch.Tensor:
    return Reshape.apply(input, torch.tensor(shape))


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()


        def forward(self, input: torch.Tensor, shape: Sequence[Union[int, torch.SymInt]]):
            return reshape(input, shape)


    test_op1 = TestModule1()

    input = torch.ones([8, 4096])
    shape = [0, 32, -1]

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input, shape), "Reshape1.onnx", opset_version=11)
    
    print(model_str1)
