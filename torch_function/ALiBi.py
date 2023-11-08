import torch
import math

class ALiBi(torch.autograd.Function):
    @staticmethod
    def symbolic(g, attention_mask: torch.Value, src_len: torch.Value,
                 tgt_len: torch.Value, num_heads: int):
        attention_with_alibi = g.op('pmx::ALiBi',
                                    attention_mask,
                                    src_len,
                                    tgt_len,
                                    num_heads_i = num_heads)
        return attention_with_alibi

    @staticmethod
    def forward(ctx, attention_mask: torch.Tensor, src_len: torch.Tensor,
                tgt_len: torch.Tensor, num_heads: int):
        if torch.onnx.is_in_onnx_export():
            return attention_mask
        assert src_len == tgt_len

        def _get_interleave(n):
            def _get_interleave_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return _get_interleave_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    _get_interleave_power_of_2(closest_power_of_2)
                    + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        def _fill_with_neg_inf(t):
            """FP16-compatible function that fills a tensor with -inf."""
            return t.float().fill_(float("-inf")).type_as(t)

        slopes = torch.Tensor(_get_interleave(num_heads))
        max_pos = max(src_len.item(), tgt_len.item())
        #max_pos = torch.max(src_len, tgt_len)
        position_point = torch.arange(max_pos) - max_pos + 1
        position_point = position_point.unsqueeze(0).unsqueeze(0).expand(num_heads, -1, -1)
        diag = torch.diag(position_point[0])
        position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
        alibi = alibi.view(num_heads, 1, max_pos)
        alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
        alibi_mask = alibi_mask.unsqueeze(0) + alibi
        attention_with_alibi = attention_mask + alibi_mask

        return attention_with_alibi

def alibi_position_embedding(attention_mask: torch.Tensor, src_len: torch.Tensor,
                             tgt_len: torch.Tensor, num_heads: int):
    return ALiBi.apply(attention_mask, src_len, tgt_len, num_heads)

if __name__ == "__main__":
    class TestALiBiModule(torch.nn.Module):
        def __init__(self, num_heads: int):
            super().__init__()
            self.num_heads = num_heads

        def forward(self, attention_mask: torch.Tensor, src_len: torch.Tensor,
                    tgt_len: torch.Tensor):
            return alibi_position_embedding(attention_mask, src_len, tgt_len, self.num_heads)

    num_heads = 40
    src_len = 4096
    tgt_len = 4096
    bz = 1
    attention_mask = torch.randn(bz, 40, 4096, 4096)

    alibi = TestALiBiModule(num_heads)
    res = alibi.forward(attention_mask, torch.tensor(src_len), torch.tensor(tgt_len))
    model_str = torch.onnx.export_to_pretty_string(
        alibi, (attention_mask, src_len, tgt_len), 'alibi.onnx',
        input_names=["attention_mask", "src_len", "tgt_len"],
        output_names=["attention_with_alibi"], opset_version=11)
    print (model_str)
