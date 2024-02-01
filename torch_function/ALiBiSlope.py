import torch
import math


class ALiBiSlope(torch.autograd.Function):
    @staticmethod
    def symbolic(g, num_heads: int):
        slopes = g.op('pmx::ALiBiSlope',
                        num_heads_i = num_heads)
        return slopes


    @staticmethod
    def forward(ctx, num_heads: int):

        if torch.onnx.is_in_onnx_export():
            return torch.zeros((num_heads), dtype=torch.float32)


        def get_slopes(heads):
            tmp = []
            closest_power_of_2 = 2 ** math.floor(math.log2(heads))
            for n in range(1, closest_power_of_2+1):
                tmp.append(2**(-8 * n / closest_power_of_2))
            if closest_power_of_2 < heads:
                for n in range(1, 2*(heads-closest_power_of_2)+1, 2):
                    tmp.append(2**(-4 * n / closest_power_of_2))
            return tmp

        return torch.tensor(get_slopes(num_heads), dtype=torch.float32)


def alibi_slope(num_heads: int):
    return ALiBiSlope.apply(num_heads)


if __name__ == "__main__":
    class TestALiBiModule(torch.nn.Module):
        def __init__(self, num_heads: int):
            super().__init__()
            self.num_heads = num_heads


        def forward(self):
            return alibi_slope(self.num_heads)


    num_heads = 40

    alibi = TestALiBiModule(num_heads)

    res = alibi.forward()
    print(res)

    model_str_1 = torch.onnx.export_to_pretty_string(
        alibi, (), "alibi.onnx", output_names=["slopes"], opset_version=11)

    print(model_str_1)
