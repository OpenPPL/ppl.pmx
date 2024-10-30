import torch

from typing import Optional

if __name__ == "__main__":
    from KeyValueCache import key_value_cache
    from MultiHeadAttention import multi_head_attention
else:
    from .KeyValueCache import key_value_cache
    from .MultiHeadAttention import multi_head_attention


class MultiHeadCacheAttention(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query: torch.Value, current_key: torch.Value, current_value: torch.Value,
                 start_pos: torch.Value, cache: torch.Value, scale: Optional[torch.Value],
                 attn_mask: Optional[torch.Value], num_heads: int, head_dim: int,
                 is_causal: bool = True, is_alibi: bool = False, softmax_scale: float = 0,
                 num_kv_heads: int = 0, num_layer: int = 1, layer_idx: int = 0,
                 quant_bit: int = 0, quant_group: int = 8, cache_layout: int = 0):
        # g: GraphContext, defined in onnx/_internal/jit_utils.py
        if attn_mask is not None:
            output = g.op('opmx::MultiHeadCacheAttention',
                query, current_key, current_value,
                start_pos, cache, scale, attn_mask,
                num_heads_i=num_heads,
                head_dim_i=head_dim,
                is_causal_i=is_causal,
                is_alibi_i=is_alibi,
                softmax_scale_f=softmax_scale,
                num_kv_heads_i=num_kv_heads,
                num_layer_i=num_layer,
                layer_idx_i=layer_idx,
                quant_bit_i=quant_bit,
                quant_group_i=quant_group,
                cache_layout_i=cache_layout)
        elif scale is not None:
            output = g.op('opmx.dynamic_batching::MultiHeadCacheAttention',
                query, current_key, current_value,
                start_pos, cache, scale,
                num_heads_i=num_heads,
                head_dim_i=head_dim,
                is_causal_i=is_causal,
                is_alibi_i=is_alibi,
                softmax_scale_f=softmax_scale,
                num_kv_heads_i=num_kv_heads,
                num_layer_i=num_layer,
                layer_idx_i=layer_idx,
                quant_bit_i=quant_bit,
                quant_group_i=quant_group,
                cache_layout_i=cache_layout)
        else:
            output = g.op('opmx.dynamic_batching::MultiHeadCacheAttention',
                query, current_key, current_value,
                start_pos, cache,
                num_heads_i=num_heads,
                head_dim_i=head_dim,
                is_causal_i=is_causal,
                is_alibi=is_alibi,
                softmax_scale_f=softmax_scale,
                num_kv_heads_i=num_kv_heads,
                num_layer_i=num_layer,
                layer_idx_i=layer_idx,
                quant_bit_i=quant_bit,
                quant_group_i=quant_group,
                cache_layout_i=cache_layout)
        return output.setTypeAs(query)


    @staticmethod
    def forward(ctx, query: torch.Tensor, current_key: torch.Tensor, current_value: torch.Tensor,
                 start_pos: torch.Tensor, cache: torch.Tensor, scale: Optional[torch.Tensor],
                 attn_mask: Optional[torch.Tensor], num_heads: int, head_dim: int,
                 is_causal: bool = True, is_alibi: bool = False, softmax_scale: float = 0,
                 num_kv_heads: int = 0, num_layer: int = 1, layer_idx: int = 0,
                 quant_bit: int = 0, quant_group: int = 8, cache_layout: int = 0):
        if torch.onnx.is_in_onnx_export():
            return query

        key, value = key_value_cache(
            current_key, current_value,
            start_pos, cache, scale, num_layer, layer_idx,
            quant_bit, quant_group, 1,
            cache_layout)

        if is_alibi:
            if __name__ == "__main__":
                from ALiBiMask import alibi_mask
            else:
                from .ALiBiMask import alibi_mask
            attn_mask = alibi_mask(
                torch.tensor(query.shape[1]),
                torch.tensor(key.shape[1]),
                attn_mask,
                num_heads=num_heads,
                data_type=query.dtype)
        
        output = multi_head_attention(
            query, key, value, attn_mask,
            num_heads, head_dim,
            is_causal, False,
            softmax_scale, num_kv_heads)

        return output


def multi_head_cache_attention(
                query: torch.Tensor, current_key: torch.Tensor, current_value: torch.Tensor,
                start_pos: torch.Tensor, cache: torch.Tensor, scale: Optional[torch.Tensor],
                attn_mask: Optional[torch.Tensor], num_heads: int, head_dim: int,
                is_causal: bool = True, is_alibi: bool = False, softmax_scale: float = 0,
                num_kv_heads: int = 0, num_layer: int = 1, layer_idx: int = 0,
                quant_bit: int = 0, quant_group: int = 8, cache_layout: int = 0) -> torch.Tensor:
    if attn_mask is not None and scale is None:
        _scale = torch.empty(0, device=query.device)
    else:
        _scale = scale
    return MultiHeadCacheAttention.apply(query, current_key, current_value, start_pos, 
                                         cache, _scale, attn_mask, num_heads, head_dim,
                                         is_causal, is_alibi, softmax_scale, num_kv_heads, num_layer,
                                         layer_idx, quant_bit, quant_group, cache_layout)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, num_heads: int, head_dim: int, is_causal: bool = True,
                     num_layer: int = 1, layer_idx: int = 0,
                     quant_bit: int = 0, quant_group: int = 8) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.is_causal = is_causal
            self.num_layer = num_layer
            self.layer_idx = layer_idx
            self.quant_bit = quant_bit
            self.quant_group = quant_group


        def forward(self, query: torch.Tensor, current_key: torch.Tensor, current_value: torch.Tensor,
                start_pos: torch.Tensor, cache: torch.Tensor, scale: torch.Tensor = None,
                attn_mask: torch.Tensor = None):
            return multi_head_cache_attention(
                                        query, current_key, current_value, start_pos, cache, scale, attn_mask,
                                        self.num_heads, self.head_dim, self.is_causal, True, 0, 0,
                                        self.num_layer, self.layer_idx, self.quant_bit, self.quant_group)


    bs = 2
    seqlen = 16
    kvlen = 32
    num_heads = 32
    head_dim = 128

    num_layer = 2
    layer_idx = 1
    quant_group = 8
    quant_bit = 8

    q = torch.randn(bs, seqlen, num_heads, head_dim)
    k = torch.randn(bs, seqlen, num_heads, head_dim)
    v = torch.randn(bs, seqlen, num_heads, head_dim)

    attn_mask = torch.randn(bs, num_heads, seqlen, seqlen)

    cache = torch.zeros([bs, num_layer, 2, kvlen, num_heads, head_dim], dtype=torch.int8)
    scale = torch.zeros([bs, num_layer, 2, kvlen, num_heads, head_dim // quant_group])
    start_pos = torch.tensor([0], dtype=torch.int64)

    test_op1 = TestModule1(num_heads, head_dim, True, num_layer, layer_idx, quant_bit, quant_group)
    test_op2 = TestModule1(num_heads, head_dim, True, num_layer, layer_idx, 0, quant_group)

    test_op1.forward(q, k, v, start_pos, cache, scale, attn_mask)
    
    model_str1 = torch.onnx.export_to_pretty_string(
       test_op1, (q, k, v, start_pos, cache, scale),
       "MultiHeadAttention1.onnx", opset_version=11)
    model_str2 = torch.onnx.export_to_pretty_string(
       test_op1, (q, k, v, start_pos, cache, scale, attn_mask),
       "MultiHeadAttention2.onnx", opset_version=11)
    
    cache = cache.to(q)
    model_str3 = torch.onnx.export_to_pretty_string(
       test_op2, (q, k, v, start_pos, cache, None, attn_mask),
       "MultiHeadAttention3.onnx", opset_version=11)

    print(model_str1)
    print(model_str2)
    print(model_str3)
