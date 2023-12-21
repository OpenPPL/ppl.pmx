import torch

from typing import Optional


class MultiHeadAttention(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query: torch.Value, key: torch.Value, value: torch.Value,
                 seqstarts: torch.Value, kvstarts: torch.Value, decoding_batches: torch.Value,
                 max_seqlen: torch.Value, max_kvlen: torch.Value,
                 attn_mask: Optional[torch.Value], num_heads: int, head_dim: int,
                 is_causal: bool = True, num_kv_heads: int = 0):
        # g: GraphContext, defined in onnx/_internal/jit_utils.py
        if attn_mask is not None:
            output = g.op('pmx.dynamic_batching::MultiHeadAttention',
                query, key, value, seqstarts,
                kvstarts, decoding_batches,
                max_seqlen, max_kvlen, attn_mask,
                num_heads_i=num_heads,
                head_dim_i=head_dim,
                is_causal_i=is_causal,
                num_kv_heads_i=num_kv_heads)
        else:
            output = g.op('pmx.dynamic_batching::MultiHeadAttention',
                query, key, value, seqstarts,
                kvstarts, decoding_batches,
                max_seqlen, max_kvlen,
                num_heads_i=num_heads,
                head_dim_i=head_dim,
                is_causal_i=is_causal,
                num_kv_heads_i=num_kv_heads)
        return output.setTypeAs(query)


    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, decoding_batches: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                attn_mask: Optional[torch.Tensor], num_heads: int, head_dim: int,
                is_causal: bool = True, num_kv_heads: int = 0):
        if torch.onnx.is_in_onnx_export():
            return query

        if attn_mask is not None and attn_mask.numel() > 0:
            assert 2 == attn_mask.dim() or 3 == attn_mask.dim(), "attn_mask.dim is {}".format(attn_mask.dim())
            # assert kvstarts[-1] == attn_mask.shape[-1], "{} vs. {}".format(kvstarts[-1], attn_mask.shape[-1])
            # assert seqstarts[-1] == attn_mask.shape[-2], "{} vs. {}".format(seqstarts[-1], attn_mask.shape[-2])

        __query, __key, __value = query, key, value

        _num_kv_heads = num_kv_heads
        if _num_kv_heads == 0:
            _num_kv_heads = num_heads
        if _num_kv_heads != num_heads:
            assert num_heads % _num_kv_heads == 0, "{} is not divisible by {}".format(num_heads, _num_kv_heads)
            num_rep = num_heads // _num_kv_heads
            __key = torch.repeat_interleave(__key, dim=-2, repeats=num_rep)
            __value = torch.repeat_interleave(__value, dim=-2, repeats=num_rep)

        output = torch.zeros_like(__query)

        seqlens = seqstarts[1:] - seqstarts[:-1]
        kvlens = kvstarts[1:] - kvstarts[:-1]
        for b, seqlen in enumerate(seqlens):
            kvlen = kvlens[b]
            seqbeg = seqstarts[b]
            seqend = seqstarts[b+1]
            kvbeg = kvstarts[b]
            kvend = kvstarts[b+1]

            if is_causal and seqlen > 1 and b >= decoding_batches.item():
                assert seqlen == kvlen, "{} is not equal to {}".format(seqlen, kvlen)
                causal_mask = torch.full((1, seqlen, kvlen), float("-inf"), device=__query.device)
                causal_mask = torch.triu(causal_mask, diagonal=1).type_as(__query)
            else:
                causal_mask = None

            _query = __query[seqbeg:seqend].transpose(0, 1)
            _key = __key[kvbeg:kvend].transpose(0, 1)
            _value = __value[kvbeg:kvend].transpose(0, 1)
            scores = torch.matmul(_query, _key.transpose(1, 2)) / torch.math.sqrt(head_dim)
            if causal_mask is not None:
                scores = scores + causal_mask
            if attn_mask is not None and attn_mask.numel() > 0:
                # scores (num_heads, seqlen, kvlen)
                scores = scores + attn_mask[..., seqbeg:seqend, kvbeg:kvend].to(scores.device)
            scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(_query)
            output[seqbeg:seqend] = torch.matmul(scores, _value).transpose(0, 1).contiguous()

        return output


def multi_head_attention(
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, decoding_batches: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                attn_mask: Optional[torch.Tensor], num_heads: int, head_dim: int,
                is_causal: bool = True, num_kv_heads: int = 0) -> torch.Tensor:
    return MultiHeadAttention.apply(query, key, value, seqstarts, kvstarts, decoding_batches,
                                    max_seqlen, max_kvlen, attn_mask,
                                    num_heads, head_dim, is_causal, num_kv_heads)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, num_heads: int, head_dim: int, is_causal: bool = True) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.is_causal = is_causal


        def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    seqstarts: torch.Tensor, kvstarts: torch.Tensor, decoding_batches: torch.Tensor,
                    max_seqlen: torch.Tensor, max_kvlen: torch.Tensor, attn_mask: torch.Tensor = None):
            return multi_head_attention(query, key, value, seqstarts, kvstarts, decoding_batches,
                                    max_seqlen, max_kvlen, attn_mask,
                                    self.num_heads, self.head_dim, self.is_causal)


    bs = 2
    seqlen = 16
    num_heads = 32
    head_dim = 128

    q = torch.randn(bs * seqlen, num_heads, head_dim)
    k = torch.randn(bs * seqlen, num_heads, head_dim)
    v = torch.randn(bs * seqlen, num_heads, head_dim)

    attn_mask = torch.randn(bs * seqlen, bs * seqlen)

    seqstarts = torch.tensor([0, seqlen, seqlen], dtype=torch.int64).cumsum(dim=0)
    decoding_batches = torch.tensor([0], dtype=torch.int64)

    max_seqlen = torch.tensor([seqlen])

    test_op1 = TestModule1(num_heads, head_dim, True)

    test_op1.forward(q, k, v, seqstarts, seqstarts, decoding_batches, max_seqlen, max_seqlen, attn_mask)

    model_str1 = torch.onnx.export_to_pretty_string(
       test_op1, (q, k, v, seqstarts, seqstarts, decoding_batches, max_seqlen, max_seqlen), "MultiHeadAttention1.onnx",
       input_names=["query", "key", "value", "seqstarts", "kvstarts", "decoding_batches", "max_seqlen", "max_kvlen"],
       output_names=["attention_output"], opset_version=11)
    model_str2 = torch.onnx.export_to_pretty_string(
       test_op1, (q, k, v, seqstarts, seqstarts, decoding_batches, max_seqlen, max_seqlen, attn_mask), "MultiHeadAttention1.onnx",
       input_names=["query", "key", "value", "seqstarts", "kvstarts", "decoding_batches", "max_seqlen", "max_kvlen", "attn_mask"],
       output_names=["attention_output"], opset_version=11)

    print(model_str1)
    print(model_str2)
