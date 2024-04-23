import torch

from typing import Optional


class MultiHeadAttention(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query: torch.Value, key: torch.Value, value: torch.Value, attn_mask: Optional[torch.Value],
                num_heads: int, head_dim: int, is_causal: bool = True, is_alibi: bool = False, num_kv_heads: int = 0):
        # g: GraphContext, defined in onnx/_internal/jit_utils.py
        if attn_mask is not None:
            output = g.op('opmx::MultiHeadAttention',
                query, key, value, attn_mask,
                num_heads_i=num_heads,
                head_dim_i=head_dim,
                is_causal_i=is_causal,
                is_alibi_i=is_alibi,
                num_kv_heads_i=num_kv_heads)
        else:
            output = g.op('opmx::MultiHeadAttention',
                query, key, value,
                num_heads_i=num_heads,
                head_dim_i=head_dim,
                is_causal_i=is_causal,
                is_alibi_i=is_alibi,
                num_kv_heads_i=num_kv_heads)
        return output.setTypeAs(query)


    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor],
                num_heads: int, head_dim: int, is_causal: bool = True, is_alibi: bool = False, num_kv_heads: int = 0):
        if attn_mask is not None and attn_mask.numel() > 0:
            assert attn_mask.dim() == 2 or attn_mask.dim() == 4 or attn_mask.dim() == 3, "attn_mask.dim() is {}".format(attn_mask.dim())
            if attn_mask.dim() == 4 or attn_mask.dim() == 3:
                #assert query.shape[0] == attn_mask.shape[0], "{} is not equal to {}".format(query.shape[0], attn_mask.shape[0])
                assert num_heads == attn_mask.shape[-3], "{} is not equal to {}".format(num_heads, attn_mask.shape[1])
            assert query.shape[1] == attn_mask.shape[-2], "{} is not equal to {}".format(query.shape[1], attn_mask.shape[-2])
            assert key.shape[1] <= attn_mask.shape[-1], "{} is bigger than {}".format(key.shape[1], attn_mask.shape[-1])

        if torch.onnx.is_in_onnx_export():
            return query

        _query, _key, _value = query, key, value
        bsz = _query.shape[0]
        seqlen_q = _query.shape[1]
        seqlen_kv = _key.shape[1]

        _num_kv_heads = num_kv_heads
        if _num_kv_heads == 0:
            _num_kv_heads = num_heads
        if _num_kv_heads != num_heads:
            assert num_heads % _num_kv_heads == 0, "{} is not divisible by {}".format(num_heads, _num_kv_heads)
            num_rep = num_heads // _num_kv_heads
            _key = (_key[:, :, :, None, :]
                    .expand(bsz, seqlen_kv, _num_kv_heads, num_rep, head_dim)
                    .reshape(bsz, seqlen_kv, num_heads, head_dim)
            )
            _value = (_value[:, :, :, None, :]
                    .expand(bsz, seqlen_kv, _num_kv_heads, num_rep, head_dim)
                    .reshape(bsz, seqlen_kv, num_heads, head_dim)
            )

        if is_causal and seqlen_q > 1:
            causal_mask = torch.zeros((1, 1, seqlen_q, seqlen_kv), device=_query.device, dtype=_query.dtype)
            causal_mask[..., -seqlen_q:] = float("-inf")
            causal_mask[..., -seqlen_q:] = torch.triu(causal_mask[..., -seqlen_q:], diagonal=1)
        else:
            causal_mask = None

        if is_alibi:
            if __name__ == "__main__":
                from ALiBiMask import alibi_mask
            else:
                from .ALiBiMask import alibi_mask

        _query = _query.transpose(1, 2)
        _key = _key.transpose(1, 2)
        _value = _value.transpose(1, 2)
        scores = torch.matmul(_query, _key.transpose(2, 3)) / torch.math.sqrt(head_dim)
        if causal_mask is not None:
            scores = scores + causal_mask
        if attn_mask is not None and attn_mask.numel() > 0:
            scores = scores + attn_mask.to(scores.device)[..., :seqlen_kv]
        if is_alibi:
            scores = scores + alibi_mask(
                torch.tensor(seqlen_q),
                torch.tensor(seqlen_kv),
                None,
                num_heads=num_heads,
                data_type=_query.dtype).to(device=scores.device)[..., :seqlen_kv]
        scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(_query)
        output = torch.matmul(scores, _value)
        output = output.transpose(1, 2).contiguous()

        return output


def multi_head_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor],
                num_heads: int, head_dim: int, is_causal: bool = True, is_alibi: bool = False, num_kv_heads: int = 0) -> torch.Tensor:
    return MultiHeadAttention.apply(query, key, value, attn_mask,
                                    num_heads, head_dim, is_causal,
                                    is_alibi, num_kv_heads)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, is_causal: bool = True) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.is_causal = is_causal


        def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor = None):
            return multi_head_attention(query, key, value, attn_mask,
                                    num_heads=self.num_heads,
                                    head_dim=self.head_dim,
                                    is_causal=self.is_causal,
                                    is_alibi=True,
                                    num_kv_heads=self.num_kv_heads)


    bs = 2
    seqlen = 38
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    q = torch.randn(bs, seqlen, num_heads, head_dim)
    k = torch.randn(bs, seqlen, num_kv_heads, head_dim)
    v = torch.randn(bs, seqlen, num_kv_heads, head_dim)

    attn_mask = torch.randn(num_heads, seqlen, seqlen)

    test_op1 = TestModule1(num_heads, num_kv_heads, head_dim, True)

    output = test_op1(q, k, v, attn_mask)

    model_str1 = torch.onnx.export_to_pretty_string(
       test_op1, (q, k, v), "MultiHeadAttention1.onnx",
       input_names=["query", "key", "value"], output_names=["attention_output"], opset_version=11)
    model_str2 = torch.onnx.export_to_pretty_string(
       test_op1, (q, k, v, attn_mask), "MultiHeadAttention1.onnx",
       input_names=["query", "key", "value", "mask"], output_names=["attention_output"], opset_version=11)

    print(model_str1)
    print(model_str2)
