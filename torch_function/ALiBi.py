import torch

import math
from typing import Optional

torch2onnx_dtype = {torch.float16: 10,
                    torch.float32: 1}

class ALiBi(torch.autograd.Function):
    @staticmethod
    def symbolic(g, seqlen_q: torch.Value, seqlen_kv: torch.Value,
                 attention_mask: Optional[torch.Value], num_heads: int, data_type: torch.dtype):
        data_type_onnx = torch2onnx_dtype[data_type]
        if attention_mask is not None:
            alibi_mask = g.op('pmx::ALiBi',
                          seqlen_q, seqlen_kv,
                          attention_mask,
                          num_heads_i = num_heads,
                          data_type_i = data_type_onnx)
        else:
            alibi_mask = g.op('pmx::ALiBi',
                          seqlen_q, seqlen_kv,
                          num_heads_i = num_heads,
                          data_type_i = data_type_onnx)
        return alibi_mask

    @staticmethod
    def forward(ctx, seqlen_q: torch.Tensor, seqlen_kv: torch.Tensor,
                attention_mask: Optional[torch.Tensor], num_heads: int, data_type: torch.dtype):

        if torch.onnx.is_in_onnx_export():
            return torch.tensor([1, num_heads, seqlen_q, seqlen_kv], dtype=data_type)

        def _get_interleave(heads):
            tmp = []
            closest_power_of_2 = 2 ** math.floor(math.log2(heads))
            for n in range(1, closest_power_of_2+1):
                tmp.append(2**(-8 * n / closest_power_of_2))
            if closest_power_of_2 < heads:
                for n in range(1, 2*(heads-closest_power_of_2)+1, 2):
                    tmp.append(2**(-4 * n / closest_power_of_2))
            return tmp

        def _fill_with_neg_inf(t):
            """FP16-compatible function that fills a tensor with -inf."""
            return t.float().fill_(float("-inf")).type_as(t)

        slopes = torch.tensor(_get_interleave(num_heads), dtype=data_type)
        seqlen_q, seqlen_kv = int(seqlen_q), int(seqlen_kv)
        alibi_mask = _fill_with_neg_inf(torch.zeros([seqlen_q, seqlen_kv], dtype=data_type))

        for i in range(seqlen_q-1, -1, -1):
            for j in range(seqlen_kv):
                mask = j - seqlen_kv + 1 + (seqlen_q - 1 - i)
                if mask <= 0:
                    alibi_mask[i][j] = mask

        # alibi_mask shape -> (num_heads, seqlen_q, seqlen_kv)
        alibi_mask = alibi_mask.unsqueeze(0).expand(num_heads, -1, -1)
        alibi_mask = slopes.unsqueeze(1).unsqueeze(1) * alibi_mask

        if attention_mask is not None and attention_mask.numel() > 0:
            assert len(attention_mask.shape) == 2
            attention_mask =  attention_mask.unsqueeze(0).expand(num_heads, seqlen_q, seqlen_kv)
            alibi_mask = alibi_mask.to(attention_mask) + attention_mask
        alibi_mask = alibi_mask.unsqueeze(0)
        return alibi_mask


def alibi_position_embedding(seqlen_q: torch.Tensor, seqlen_kv: torch.Tensor,
                             attention_mask: Optional[torch.Tensor], num_heads: int, data_type: torch.dtype):
    return ALiBi.apply(seqlen_q, seqlen_kv, attention_mask, num_heads, data_type)

if __name__ == "__main__":
    class TestALiBiModule(torch.nn.Module):
        def __init__(self, num_heads: int, data_type: torch.dtype):
            super().__init__()
            self.num_heads = num_heads
            self.data_type = data_type

        def forward(self, seqlen_q: torch.Tensor, seqlen_kv: torch.Tensor,
                    attention_mask: torch.Tensor = None):
            return alibi_position_embedding(seqlen_q, seqlen_kv, attention_mask, self.num_heads, self.data_type)

    num_heads = 40
    seqlen_q = torch.tensor(2)
    seqlen_kv = torch.tensor(5)
    bz = 1
    data_type = torch.float16
    attention_mask = torch.randn(seqlen_q, seqlen_kv, dtype=data_type)

    alibi = TestALiBiModule(num_heads, data_type)
    res = alibi.forward(seqlen_q, seqlen_kv, attention_mask)

    model_str_1 = torch.onnx.export_to_pretty_string(
        alibi, (seqlen_q, seqlen_kv, attention_mask), "alibi.onnx",
        input_names=["seqlen_q", "seqlen_kv", "attention_mask"],
        output_names=["attention_with_alibi"], opset_version=11)


    model_str_2 = torch.onnx.export_to_pretty_string(
        alibi, (seqlen_q, seqlen_kv), "alibi.onnx",
        input_names=["seqlen_q", "seqlen_kv"],
        output_names=["attention_with_alibi"], opset_version=11)

    print (model_str_1)
    print (model_str_2)
