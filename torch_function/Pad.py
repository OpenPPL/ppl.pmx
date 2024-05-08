import torch

from typing import Sequence, Union


class Pad(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input: torch.Value, padding: torch.Value,
                 mode: str='constant', value: float=0.):
        output = g.op("opmx::Pad", input, padding,
                      mode_s = mode,
                      value_f = value)
        return output


    @staticmethod
    def forward(self, input: torch.Tensor, padding: torch.Tensor,
                mode: str='constant', value: float=0.):
        output = torch.nn.functional.pad(input, tuple(padding), mode, value)
        return output


def pad(input: torch.Tensor, padding: Sequence[Union[int, torch.SymInt]],
            mode: str='constant', value: float=0.) -> torch.Tensor:
    return Pad.apply(input, padding, mode, value)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, mode='constant', value: float=0.) -> None:
            super().__init__()
            self.mode = mode
            self.value = value


        def forward(self, input: torch.Tensor, padding: Sequence[Union[int, torch.SymInt]]):
            return pad(input, padding, self.mode, self.value)


    test_op1 = TestModule1()

    input = torch.ones([1, 32, 512, 512])
    padding = torch.tensor([0,1,0,1], dtype=torch.int32)
    #pad = [0,1,0,1]

    output = test_op1.forward(input, padding)
    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input, padding), "pad.onnx", opset_version=11)

    print(model_str1)
