import torch
import math


torch2onnx_dtype = {torch.float16: 10,
                    torch.float32: 1}


class ALiBiSlope(torch.autograd.Function):
    @staticmethod
    def symbolic(g, num_heads: int, data_type: torch.dtype):
        data_type_onnx = torch2onnx_dtype[data_type]
        slopes = g.op('pmx::ALiBiSlope',
                        num_heads_i = num_heads,
                        data_type_i = data_type_onnx)
        return slopes


    @staticmethod
    def forward(ctx, num_heads: int, data_type: torch.dtype):

        if torch.onnx.is_in_onnx_export():
            return torch.zeros((num_heads), dtype=data_type)


        def get_slopes(heads):
            tmp = []
            closest_power_of_2 = 2 ** math.floor(math.log2(heads))
            for n in range(1, closest_power_of_2+1):
                tmp.append(2**(-8 * n / closest_power_of_2))
            if closest_power_of_2 < heads:
                for n in range(1, 2*(heads-closest_power_of_2)+1, 2):
                    tmp.append(2**(-4 * n / closest_power_of_2))
            return tmp

        return torch.tensor(get_slopes(num_heads), dtype=data_type)


def alibi_slope(num_heads: int, data_type: torch.dtype):
    return ALiBiSlope.apply(num_heads, data_type)


if __name__ == "__main__":
    class TestALiBiModule(torch.nn.Module):
        def __init__(self, num_heads: int, data_type: torch.dtype):
            super().__init__()
            self.num_heads = num_heads
            self.data_type = data_type


        def forward(self):
            return alibi_slope(self.num_heads, self.data_type)


    num_heads = 40
    data_type = torch.float16

    alibi = TestALiBiModule(num_heads, data_type)

    res = alibi.forward()
    print(res)

    model_str_1 = torch.onnx.export_to_pretty_string(
        alibi, (), "alibi.onnx", output_names=["slopes"], opset_version=11)

    print(model_str_1)
