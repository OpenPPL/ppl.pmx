import torch

from typing import Sequence, Union


class Pad(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input: torch.Value, padding: torch.Value,
                 value: torch.Value, mode: str='constant'):
        output = g.op("opmx::Pad", input, padding, value,
                      mode_s = mode)
        return output


    @staticmethod
    def forward(self, input: torch.Tensor, padding: torch.Tensor,
                value: torch.Tensor, mode: str='constant'):
        output = torch.nn.functional.pad(input, tuple(padding), mode, value.item())
        return output


def pad(input: torch.Tensor, padding: torch.Tensor,
        value: torch.Tensor, mode: str='constant') -> torch.Tensor:
    return Pad.apply(input, padding, value, mode)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, mode='constant') -> None:
            super().__init__()
            self.mode = mode


        def forward(self, input: torch.Tensor, padding: torch.Tensor, value: torch.Tensor):
            return pad(input, padding, value, self.mode)


    test_op1 = TestModule1()

    input = torch.ones([1, 32, 512, 512])
    padding = torch.tensor([0,1,0,1], dtype=torch.int32)
    value = torch.tensor(0)
    #pad = [0,1,0,1]

    output = test_op1.forward(input, padding, value)
    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input, padding, value), "pad.onnx", opset_version=11)

    print(model_str1)
