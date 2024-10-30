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
                 seqstarts: torch.Value, kvstarts: torch.Value, cachestarts: torch.Value,
                 start_pos: torch.Value, decoding_batches: torch.Value,
                 max_seqlen: torch.Value, max_kvlen: torch.Value,
                 cache: torch.Value, scale: Optional[torch.Value],
                 attn_mask: Optional[torch.Value],
                 num_heads: int, head_dim: int,
                 is_causal: bool = True,
                 is_alibi: bool = False,
                 softmax_scale: float = 0,
                 num_kv_heads: int = 0,
                 num_layer: int = 1, layer_idx: int = 0,
                 quant_bit: int = 0, quant_group: int = 8,
                 cache_mode: int = 0, cache_layout: int = 0,
                 page_size: int = 128):
        # g: GraphContext, defined in onnx/_internal/jit_utils.py
        if attn_mask is not None:
            output = g.op('opmx.dynamic_batching::MultiHeadCacheAttention',
                query, current_key, current_value,
                seqstarts, kvstarts, cachestarts,
                start_pos, decoding_batches,
                max_seqlen, max_kvlen,
                cache, scale, attn_mask,
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
                cache_mode_i=cache_mode,
                cache_layout_i=cache_layout,
                page_size_i=page_size)
        elif scale is not None:
            output = g.op('opmx.dynamic_batching::MultiHeadCacheAttention',
                query, current_key, current_value,
                seqstarts, kvstarts, cachestarts,
                start_pos, decoding_batches,
                max_seqlen, max_kvlen,
                cache, scale,
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
                cache_mode_i=cache_mode,
                cache_layout_i=cache_layout,
                page_size_i=page_size)
        else:
            output = g.op('opmx.dynamic_batching::MultiHeadCacheAttention',
                query, current_key, current_value,
                seqstarts, kvstarts, cachestarts,
                start_pos, decoding_batches,
                max_seqlen, max_kvlen, cache,
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
                cache_mode_i=cache_mode,
                cache_layout_i=cache_layout,
                page_size_i=page_size)
        return output.setTypeAs(query)


    @staticmethod
    def forward(ctx, query: torch.Tensor, current_key: torch.Tensor, current_value: torch.Tensor,
                 seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                 start_pos: torch.Tensor, decoding_batches: torch.Tensor,
                 max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                 cache: torch.Tensor, scale: Optional[torch.Tensor],
                 attn_mask: Optional[torch.Tensor],
                 num_heads: int, head_dim: int,
                 is_causal: bool = True,
                 is_alibi: bool = False,
                 softmax_scale: float = 0,
                 num_kv_heads: int = 0,
                 num_layer: int = 1, layer_idx: int = 0,
                 quant_bit: int = 0, quant_group: int = 8,
                 cache_mode: int = 0, cache_layout: int = 0,
                 page_size: int = 128):
        if torch.onnx.is_in_onnx_export():
            return query

        key, value = key_value_cache(
            current_key, current_value,
            seqstarts, kvstarts, cachestarts,
            start_pos, max_seqlen, max_kvlen,
            cache, scale, num_layer, layer_idx,
            quant_bit, quant_group, 1,
            cache_mode, cache_layout, page_size)

        if is_alibi:
            if __name__ == "__main__":
                from ALiBiMask import alibi_mask
            else:
                from .ALiBiMask import alibi_mask
            attn_mask = alibi_mask(seqstarts, kvstarts, attn_mask,
                                   num_heads, query.dtype)

        output = multi_head_attention(
            query, key, value, seqstarts,
            kvstarts, decoding_batches,
            max_seqlen, max_kvlen, attn_mask,
            num_heads, head_dim,
            is_causal, False,
            softmax_scale, num_kv_heads)

        return output


def multi_head_cache_attention(
                query: torch.Tensor, current_key: torch.Tensor, current_value: torch.Tensor,
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                start_pos: torch.Tensor, decoding_batches: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                cache: torch.Tensor, scale: Optional[torch.Tensor],
                attn_mask: Optional[torch.Tensor],
                num_heads: int, head_dim: int,
                is_causal: bool = True,
                is_alibi: bool = False,
                softmax_scale: float = 0, 
                num_kv_heads: int = 0,
                num_layer: int = 1, layer_idx: int = 0,
                quant_bit: int = 0, quant_group: int = 8,
                cache_mode: int = 0, cache_layout: int = 0,
                page_size: int = 128) -> torch.Tensor:
    if attn_mask is not None and scale is None:
        _scale = torch.empty(0, device=query.device)
    else:
        _scale = scale
    return MultiHeadCacheAttention.apply(query, current_key, current_value, seqstarts, kvstarts,
                                        cachestarts, start_pos, decoding_batches,
                                        max_seqlen, max_kvlen, cache, _scale,
                                        attn_mask, num_heads, head_dim,
                                        is_causal, is_alibi, softmax_scale,
                                        num_kv_heads, num_layer,
                                        layer_idx, quant_bit, quant_group,
                                        cache_mode, cache_layout,
                                        page_size)


if __name__ == "__main__":
    class TestModule(torch.nn.Module):
        def __init__(self,
                     num_heads: int, num_kv_heads: int, head_dim: int,
                     is_causal: bool = True, is_alibi: bool = False,
                     num_layer: int = 1, layer_idx: int = 0,
                     quant_bit: int = 0, quant_group: int = 8,
                     cache_mode: int = 0, cache_layout: int = 0,
                     page_size: int = 128) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.is_causal = is_causal
            self.is_alibi = is_alibi
            self.num_layer = num_layer
            self.layer_idx = layer_idx
            self.quant_bit = quant_bit
            self.quant_group = quant_group
            self.cache_mode = cache_mode
            self.cache_layout = cache_layout
            self.page_size = page_size


        @torch.inference_mode()
        def forward(self,
                    query: torch.Tensor, current_key: torch.Tensor, current_value: torch.Tensor,
                    seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                    start_pos: torch.Tensor, decoding_batches: torch.Tensor,
                    max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                    cache: torch.Tensor, scale: torch.Tensor = None,
                    attn_mask: torch.Tensor = None):
            return multi_head_cache_attention(
                    query, current_key, current_value, seqstarts, kvstarts, cachestarts,
                    start_pos, decoding_batches, max_seqlen, max_kvlen, cache, scale, attn_mask,
                    self.num_heads, self.head_dim, self.is_causal, self.is_alibi, 0, self.num_kv_heads,
                    self.num_layer, self.layer_idx, self.quant_bit, self.quant_group,
                    self.cache_mode, self.cache_layout, self.page_size)

    
    def dump_tensor(X: torch.Tensor, name: str):
        shape_str = "" if X.dim == 0 else str(X.shape[0])
        for d in X.shape[1:]:
            shape_str = shape_str + "_" + str(d)
        type_dict = {
            torch.float: "fp32",
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.int8: "int8",
            torch.int64: "int64",
        }
        filename = "{}-{}-{}.bin".format(name, shape_str, type_dict[X.dtype])
        X.cpu().numpy().tofile(filename)


    batch = 2
    seqlen = 4
    genlen = 8
    num_heads = 32
    num_kv_heads = 4
    head_dim = 128
    num_layer = 2
    layer_idx = 1
    page_size = 128
    cache_layout = 3
    quant_group = 8

    tensor_type = torch.float
    # tensor_type = torch.float16

    q = torch.randn(batch * seqlen, num_heads, head_dim, dtype=tensor_type)
    k = torch.randn(batch * seqlen, num_kv_heads, head_dim, dtype=tensor_type)
    v = torch.randn(batch * seqlen, num_kv_heads, head_dim, dtype=tensor_type)
    attn_mask = torch.randn(batch * seqlen, batch * genlen, dtype=tensor_type)
    seqstarts = torch.tensor([0, seqlen, seqlen], dtype=torch.int64).cumsum(dim=0)
    decoding_batches = torch.tensor([0], dtype=torch.int64)


    for quant_bit in [0, 8]:
        for cache_mode in [0, 1]:
            if quant_bit == 0:
                cache_dtype = tensor_type
            if quant_bit == 8:
                cache_dtype = torch.int8
            if cache_mode == 0:
                cache = torch.zeros([num_layer, 2, num_kv_heads, batch * genlen, head_dim], dtype=cache_dtype)
                scale = torch.zeros([num_layer, 2, num_kv_heads, batch * genlen, head_dim // quant_group], dtype=tensor_type)
                cachestarts = torch.arange(0, batch * genlen, genlen, dtype=torch.int64)
            if cache_mode == 1:
                cache = torch.zeros([num_layer, 2, num_kv_heads, (genlen + page_size - 1) // page_size * page_size * batch, head_dim], dtype=cache_dtype)
                scale = torch.zeros([num_layer, 2, num_kv_heads, (genlen + page_size - 1) // page_size * page_size * batch, head_dim // quant_group], dtype=tensor_type)
                cachestarts = torch.tensor([[0 * page_size], [1 * page_size]], dtype=torch.int64)

            attnetion = TestModule(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                is_causal=True,
                is_alibi=False,
                num_layer=num_layer,
                layer_idx=layer_idx,
                quant_bit=quant_bit,
                quant_group=quant_group,
                cache_mode=cache_mode,
                cache_layout=cache_layout,
                page_size=page_size
            )

            _cache, _scale, _attn_mask = cache, scale, attn_mask

            # _cache, _scale, _attn_mask = cache.cuda(), scale.cuda(), attn_mask.cuda()

            for step in [0, 1]:
                if step == 0:
                    start_pos = torch.full([batch], 0, dtype=torch.int64)
                    max_seqlen = torch.tensor([seqlen])
                    max_kvlen = torch.tensor([seqlen])
                if step == 1:
                    start_pos = torch.full([batch], seqlen, dtype=torch.int64)
                    max_seqlen = torch.tensor([seqlen])
                    max_kvlen = torch.tensor([genlen])

                kvstarts = torch.zeros([batch + 1], dtype=torch.int64)
                kvstarts[1:] = start_pos.cumsum(0)
                kvstarts = kvstarts + seqstarts

                _q, _k, _v, _start_pos = q, k, v, start_pos
                _seqstarts, _kvstarts, _cachestarts = seqstarts, kvstarts, cachestarts

                # _q, _k, _v, _start_pos = q.cuda(), k.cuda(), v.cuda(), start_pos.cuda()
                # _seqstarts, _kvstarts, _cachestarts = seqstarts.cuda(), kvstarts.cuda(), cachestarts.cuda()

                # dump_tensor(_q, f"q_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(_k, f"k_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(_v, f"v_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(_seqstarts, f"seqstarts_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(_kvstarts, f"kvstarts_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(_cachestarts, f"cachestarts_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(_start_pos, f"start_pos_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(decoding_batches, f"decodeing_batches_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(max_seqlen, f"max_seqlen_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(max_kvlen, f"max_kvlen_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(_cache, f"cache_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(_scale, f"scale_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(_attn_mask, f"attn_mask_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")

                output = attnetion.forward(
                    _q, _k, _v, _seqstarts, _kvstarts,
                    _cachestarts, _start_pos, decoding_batches,
                    max_seqlen, max_kvlen, _cache, _scale, _attn_mask)

                # dump_tensor(output, f"output_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(_cache, f"cache_output_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")
                # dump_tensor(_scale, f"scale_output_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}")

                model_str = torch.onnx.export_to_pretty_string(
                    attnetion,
                    (_q, _k, _v, _seqstarts, _kvstarts,
                    _cachestarts, _start_pos, decoding_batches,
                    max_seqlen, max_kvlen, _cache, _scale, _attn_mask),
                    f"mhca_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}.onnx",
                    opset_version=11)
                print(model_str)

                # torch.onnx.export(
                #     attnetion.cpu(),
                #     (_q, _k, _v, _seqstarts, _kvstarts,
                #     _cachestarts, _start_pos, decoding_batches,
                #     max_seqlen, max_kvlen, _cache, _scale, _attn_mask),
                #     f"mhca_qb{quant_bit}_qg{quant_group}_cm{cache_mode}_cl{cache_layout}_step{step}.onnx",
                #     output_names=["output"],
                #     do_constant_folding=True,
                #     opset_version=11)

