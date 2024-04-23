import torch

import math
from typing import Optional


torch2onnx_dtype = {torch.float16: 10,
                    torch.float32: 1}


class ALiBiMask(torch.autograd.Function):
    @staticmethod
    def symbolic(g, seqstarts: torch.Value, kvstarts: torch.Value,
                 attention_mask: Optional[torch.Value], num_heads: int, data_type: torch.dtype):
        data_type_onnx = torch2onnx_dtype[data_type]
        if attention_mask is not None:
            alibi_mask = g.op('opmx.dynamic_batching::ALiBiMask',
                          seqstarts, kvstarts,
                          attention_mask,
                          num_heads_i = num_heads,
                          data_type_i = data_type_onnx)
        else:
            alibi_mask = g.op('opmx.dynamic_batching::ALiBiMask',
                          seqstarts, kvstarts,
                          num_heads_i = num_heads,
                          data_type_i = data_type_onnx)
        return alibi_mask


    @staticmethod
    def forward(ctx, seqstarts: torch.Tensor, kvstarts: torch.Tensor,
                attention_mask: Optional[torch.Tensor], num_heads: int, data_type: torch.dtype):

        if torch.onnx.is_in_onnx_export():
            return torch.zeros((num_heads, seqstarts[-1], kvstarts[-1]), dtype=data_type)


        def get_slopes(heads):
            tmp = []
            closest_power_of_2 = 2 ** math.floor(math.log2(heads))
            for n in range(1, closest_power_of_2+1):
                tmp.append(2**(-8 * n / closest_power_of_2))
            if closest_power_of_2 < heads:
                for n in range(1, 2*(heads-closest_power_of_2)+1, 2):
                    tmp.append(2**(-4 * n / closest_power_of_2))
            return tmp


        # pad last dim to compatible with flah attention
        last_dim = kvstarts[-1]
        padded_last_dim = (kvstarts[-1] + 15) // 16 * 16

        slopes = torch.tensor(get_slopes(num_heads), dtype=data_type)
        alibi_mask = torch.zeros((seqstarts[-1], padded_last_dim), dtype=data_type)

        seqlens = seqstarts[1:] - seqstarts[:-1]
        kvlens = kvstarts[1:] - kvstarts[:-1]
        for batch_idx, seqlen in enumerate(seqlens):
            kvlen = kvlens[batch_idx]
            seqbeg = seqstarts[batch_idx]
            seqend = seqstarts[batch_idx+1]
            kvbeg = kvstarts[batch_idx]
            kvend = kvstarts[batch_idx+1]

            tmp_alibi_mask = torch.full((seqlen, kvlen), float("-inf"), dtype=data_type)
            for i in range(seqlen-1, -1, -1):
                for j in range(kvlen):
                    mask = j - kvlen + 1 + (seqlen - 1 - i)
                    if mask <= 0:
                        tmp_alibi_mask[i][j] = mask
            alibi_mask[seqbeg:seqend, kvbeg:kvend] = tmp_alibi_mask

        # alibi_mask shape -> (num_heads, sum(seqlens), sum(kvlens))
        alibi_mask = alibi_mask.unsqueeze(0).expand(num_heads, -1, -1)
        alibi_mask = slopes.unsqueeze(1).unsqueeze(1) * alibi_mask

        if attention_mask is not None and attention_mask.numel() > 0:
            assert len(attention_mask.shape) == 2 or len(attention_mask.shape) == 3
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask.unsqueeze(0).expand(num_heads, -1, -1)
            alibi_mask[..., :last_dim] = (alibi_mask[..., :last_dim].to(attention_mask[..., :last_dim])
                                        + attention_mask[..., :last_dim])
        return alibi_mask


def alibi_mask(seqstarts: torch.Tensor, kvstarts: torch.Tensor,
                attention_mask: Optional[torch.Tensor],
                num_heads: int, data_type: torch.dtype):
    return ALiBiMask.apply(seqstarts, kvstarts, attention_mask, num_heads, data_type)


if __name__ == "__main__":
    class TestALiBiModule(torch.nn.Module):
        def __init__(self, num_heads: int, data_type: torch.dtype):
            super().__init__()
            self.num_heads = num_heads
            self.data_type = data_type


        def forward(self, seqstarts: torch.Tensor, kvstarts: torch.Tensor,
                    attention_mask: torch.Tensor = None):
            return alibi_mask(seqstarts, kvstarts, attention_mask, self.num_heads, self.data_type)


    num_heads = 40
    #seqstarts = torch.tensor([0, 8, 16, 24, 32])
    seqstarts = torch.tensor([0, 1, 2, 3, 4])
    kvstarts = torch.tensor([0, 8, 16, 24, 32])
    bz = 1
    data_type = torch.float16
    attention_mask = torch.zeros((seqstarts[-1], kvstarts[-1]), dtype=data_type)

    alibi = TestALiBiModule(num_heads, data_type)
    res = alibi.forward(seqstarts, kvstarts, attention_mask)

    model_str_1 = torch.onnx.export_to_pretty_string(
        alibi, (seqstarts, kvstarts, attention_mask), "alibi.onnx",
        input_names=["seqstarts", "kvstarts", "attention_mask"],
        output_names=["attention_with_alibi"], opset_version=11)

    model_str_2 = torch.onnx.export_to_pretty_string(
        alibi, (seqstarts, kvstarts), "alibi.onnx",
        input_names=["seqstarts", "kvstarts"],
        output_names=["attention_with_alibi"], opset_version=11)

    print(model_str_1)
    print(model_str_2)
