import torch
import torch.distributed as dist
import torch.nn.functional as F

from typing import Optional

if __name__ == "__main__":
    from KeyValueCache import key_value_cache
    from MultiHeadAttention import multi_head_attention
else:
    from .KeyValueCache import key_value_cache
    from .MultiHeadAttention import multi_head_attention


class TensorParallelFusedMultiLatentCacheAttention(torch.autograd.Function):
    @staticmethod
    def symbolic(g, hidden_states: torch.Value,
                q_a_weight: torch.Value, q_norm_weight: torch.Value, q_b_weight: torch.Value,
                kv_a_weight: torch.Value, kv_norm_weight: torch.Value,
                k_b_weight: torch.Value, v_b_weight: torch.Value, o_weight: torch.Value,
                rotary_sin: torch.Value, rotary_cos: torch.Value,
                seqstarts: torch.Value, kvstarts: torch.Value, cachestarts: torch.Value,
                start_pos: torch.Value, decoding_batches: torch.Value,
                max_seqlen: torch.Value, max_kvlen: torch.Value,
                cache: torch.Value, scale: Optional[torch.Value],
                attn_mask: Optional[torch.Value],
                proc_group: dist.ProcessGroup,
                num_heads: int, hidden_dim: int,
                q_lora_rank: int, kv_lora_rank: int,
                head_dim: int, rotray_dim: int,
                is_causal: bool = True,
                is_interleaved_rotary: bool = True,
                softmax_scale: float = 0, 
                num_kv_heads: int = 0,
                vo_head_dim: int = 0,
                num_layer: int = 1, layer_idx: int = 0,
                quant_bit: int = 0, quant_group: int = 8,
                cache_mode: int = 0, cache_layout: int = 0,
                page_size: int = 128):
        # g: GraphContext, defined in onnx/_internal/jit_utils.py
        if attn_mask is not None:
            output = g.op('opmx.dynamic_batching::TensorParallelFusedMultiLatentCacheAttention',
                hidden_states,
                q_a_weight, q_norm_weight, q_b_weight,
                kv_a_weight, kv_norm_weight,
                k_b_weight, v_b_weight, o_weight,
                rotary_sin, rotary_cos,
                seqstarts, kvstarts, cachestarts,
                start_pos, decoding_batches,
                max_seqlen, max_kvlen,
                cache, scale, attn_mask,
                num_heads_i=num_heads,
                hidden_dim_i=hidden_dim,
                q_lora_rank_i=q_lora_rank,
                kv_lora_rank_i=kv_lora_rank,
                head_dim_i=head_dim,
                rotray_dim_i=rotray_dim,
                is_causal_i=is_causal,
                is_interleaved_rotary_i=is_interleaved_rotary,
                softmax_scale_f=softmax_scale,
                num_kv_heads_i=num_kv_heads,
                vo_head_dim_i=vo_head_dim,
                num_layer_i=num_layer,
                layer_idx_i=layer_idx,
                quant_bit_i=quant_bit,
                quant_group_i=quant_group,
                cache_mode_i=cache_mode,
                cache_layout_i=cache_layout,
                page_size_i=page_size)
        if scale is not None:
            output = g.op('opmx.dynamic_batching::TensorParallelFusedMultiLatentCacheAttention',
                hidden_states,
                q_a_weight, q_norm_weight, q_b_weight,
                kv_a_weight, kv_norm_weight,
                k_b_weight, v_b_weight, o_weight,
                rotary_sin, rotary_cos,
                seqstarts, kvstarts, cachestarts,
                start_pos, decoding_batches,
                max_seqlen, max_kvlen,
                cache, scale,
                num_heads_i=num_heads,
                hidden_dim_i=hidden_dim,
                q_lora_rank_i=q_lora_rank,
                kv_lora_rank_i=kv_lora_rank,
                head_dim_i=head_dim,
                rotray_dim_i=rotray_dim,
                is_causal_i=is_causal,
                is_interleaved_rotary_i=is_interleaved_rotary,
                softmax_scale_f=softmax_scale,
                num_kv_heads_i=num_kv_heads,
                vo_head_dim_i=vo_head_dim,
                num_layer_i=num_layer,
                layer_idx_i=layer_idx,
                quant_bit_i=quant_bit,
                quant_group_i=quant_group,
                cache_mode_i=cache_mode,
                cache_layout_i=cache_layout,
                page_size_i=page_size)
        else:
            output = g.op('opmx.dynamic_batching::TensorParallelFusedMultiLatentCacheAttention',
                hidden_states,
                q_a_weight, q_norm_weight, q_b_weight,
                kv_a_weight, kv_norm_weight,
                k_b_weight, v_b_weight, o_weight,
                rotary_sin, rotary_cos,
                seqstarts, kvstarts, cachestarts,
                start_pos, decoding_batches,
                max_seqlen, max_kvlen,
                cache,
                num_heads_i=num_heads,
                hidden_dim_i=hidden_dim,
                q_lora_rank_i=q_lora_rank,
                kv_lora_rank_i=kv_lora_rank,
                head_dim_i=head_dim,
                rotray_dim_i=rotray_dim,
                is_causal_i=is_causal,
                is_interleaved_rotary_i=is_interleaved_rotary,
                softmax_scale_f=softmax_scale,
                num_kv_heads_i=num_kv_heads,
                vo_head_dim_i=vo_head_dim,
                num_layer_i=num_layer,
                layer_idx_i=layer_idx,
                quant_bit_i=quant_bit,
                quant_group_i=quant_group,
                cache_mode_i=cache_mode,
                cache_layout_i=cache_layout,
                page_size_i=page_size)
        return output.setTypeAs(hidden_states)


    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor,
                q_a_weight: torch.Tensor, q_norm_weight: torch.Tensor, q_b_weight: torch.Tensor,
                kv_a_weight: torch.Tensor, kv_norm_weight: torch.Tensor,
                k_b_weight: torch.Tensor, v_b_weight: torch.Tensor, o_weight: torch.Tensor,
                rotary_sin: torch.Tensor, rotary_cos: torch.Tensor,
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                start_pos: torch.Tensor, decoding_batches: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                cache: torch.Tensor, scale: Optional[torch.Tensor],
                attn_mask: Optional[torch.Tensor],
                proc_group: dist.ProcessGroup,
                num_heads: int, hidden_dim: int,
                q_lora_rank: int, kv_lora_rank: int,
                head_dim: int, rotray_dim: int,
                is_causal: bool = True,
                is_interleaved_rotary: bool = True,
                softmax_scale: float = 0,
                num_kv_heads: int = 0,
                vo_head_dim: int = 0,
                num_layer: int = 1, layer_idx: int = 0,
                quant_bit: int = 0, quant_group: int = 8,
                cache_mode: int = 0, cache_layout: int = 0,
                page_size: int = 128):
        if torch.onnx.is_in_onnx_export():
            return hidden_states
        
        num_local_heads = num_heads
        num_local_kv_heads = num_kv_heads if num_kv_heads > 0 else num_heads
        if proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
            world_size = torch.distributed.get_world_size(proc_group)
            num_local_heads = num_heads // world_size
            num_local_kv_heads = num_kv_heads // world_size

        qk_nope_head_dim = head_dim - rotray_dim
        qk_rope_head_dim = rotray_dim
        _vo_head_dim = vo_head_dim
        if vo_head_dim == 0:
            _vo_head_dim = head_dim


        def rms_norm(
            X: torch.Tensor, weight: torch.Tensor,
            axis: int = -1, eps: float = 1e-6):
            x = X.float()
            mean_square = x.pow(2).mean(axis, keepdim=True)
            Y = x * torch.rsqrt(mean_square + eps)
            return Y.type_as(X) * weight
        
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary(x, seqstarts, start_pos, sin, cos, rotary_interleaved):
            # x (b*s, h, d)
            # sin, cos (>=max_kvlen, d)
            for b, position in enumerate(start_pos):
                seqbeg = seqstarts[b]
                seqlen = seqstarts[b+1] - seqstarts[b]
                _cos = cos[position:position+seqlen].unsqueeze(1) # (s, 1, d)
                _sin = sin[position:position+seqlen].unsqueeze(1) # (s, 1, d)

                _x = x[seqbeg:seqbeg+seqlen] # (s, h ,d)
                h, d = _x.shape[-2], _x.shape[-1]
                if rotary_interleaved:
                    _x = _x.view(-1, h, d // 2, 2).transpose(-1, -2).reshape(-1, h, d)
                _x = (_x * _cos) + (rotate_half(_x) * _sin)
                if rotary_interleaved:
                    _x = _x.view(-1, h, 2, d // 2).transpose(-1, -2).reshape(-1, h, d)
                x[seqbeg:seqbeg+seqlen] = _x
            return x

        if q_lora_rank == 0:
            q = F.linear(hidden_states, q_a_weight)
        else:
            q = F.linear(rms_norm(F.linear(hidden_states, q_a_weight), q_norm_weight), q_b_weight)
        q = q.view(-1, num_local_heads, qk_nope_head_dim + qk_rope_head_dim) # (b*s, h, qk_n + qk_r)

        compressed_kv = F.linear(hidden_states, kv_a_weight)
        compressed_kv = compressed_kv.view(-1, 1, kv_lora_rank + qk_rope_head_dim)
        compressed_kv[..., :kv_lora_rank] = rms_norm(compressed_kv[..., :kv_lora_rank], kv_norm_weight)

        if rotray_dim > 0 and rotary_sin.numel() > 0 and rotary_cos.numel() > 0:
            q[..., -qk_rope_head_dim:] = apply_rotary(
                q[..., -qk_rope_head_dim:], seqstarts, start_pos,
                rotary_sin, rotary_cos, is_interleaved_rotary)
            compressed_kv[..., -qk_rope_head_dim:] = apply_rotary(
                compressed_kv[..., -qk_rope_head_dim:], seqstarts, start_pos,
                rotary_sin, rotary_cos, is_interleaved_rotary)

        # CC method, TODO: ACC method
        # compressed kv没有V部分占用，kv cache数据排布需要修改
        compressed_kv, _ = key_value_cache(
            compressed_kv, None,
            seqstarts, kvstarts, cachestarts,
            start_pos, max_seqlen, max_kvlen,
            cache, scale, num_layer, layer_idx,
            quant_bit, quant_group, 1,
            cache_mode, cache_layout, page_size)

        compressed_kv, k_pe = torch.split( # (b*s, 1, kv_lora + qk_r)
            compressed_kv, [kv_lora_rank, qk_rope_head_dim], dim=-1
        )
        compressed_kv = compressed_kv.view(-1, kv_lora_rank)
        # 真实现的时候可以修改LDA，欸，Torch你在干嘛？
        k = F.linear(compressed_kv, k_b_weight).view(-1, num_local_kv_heads, qk_nope_head_dim)
        k = torch.cat([k, k_pe.expand(k_pe.shape[0], k.shape[1], k_pe.shape[2])], dim=-1) # (b*s, h, qk_n + qk_r)
        v = F.linear(compressed_kv, v_b_weight).view(-1, num_local_kv_heads, _vo_head_dim) # (b*s, h, vo_d)

        output_parallel = multi_head_attention(
            q, k, v, seqstarts,
            kvstarts, decoding_batches,
            max_seqlen, max_kvlen, attn_mask,
            num_local_heads, head_dim,
            is_causal, False, softmax_scale, num_local_kv_heads) # (bs, h, vo_d)

        output_parallel = F.linear(output_parallel.reshape(-1, num_heads * vo_head_dim), o_weight)

        if proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
            torch.distributed.all_reduce(output_parallel, group=proc_group)

        return output_parallel


def tensor_parallel_fused_multi_head_cache_attention(
                hidden_states: torch.Tensor,
                q_a_weight: torch.Tensor, q_norm_weight: Optional[torch.Tensor], q_b_weight: Optional[torch.Tensor],
                kv_a_weight: torch.Tensor, kv_norm_weight: torch.Tensor,
                k_b_weight: torch.Tensor, v_b_weight: torch.Tensor, o_weight: torch.Tensor,
                rotary_sin: Optional[torch.Tensor], rotary_cos: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                start_pos: torch.Tensor, decoding_batches: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                cache: torch.Tensor, scale: Optional[torch.Tensor],
                attn_mask: Optional[torch.Tensor],
                proc_group: dist.ProcessGroup,
                num_heads: int, hidden_dim: int,
                q_lora_rank: int, kv_lora_rank: int,
                head_dim: int, rotray_dim: int,
                is_causal: bool = True,
                is_interleaved_rotary: bool = True,
                softmax_scale: float = 0,
                num_kv_heads: int = 0,
                vo_head_dim: int = 0,
                num_layer: int = 1, layer_idx: int = 0,
                quant_bit: int = 0, quant_group: int = 8,
                cache_mode: int = 0, cache_layout: int = 0,
                page_size: int = 128) -> torch.Tensor:
    if attn_mask is not None and scale is None:
        _scale = torch.empty(0, device=hidden_states.device)
    else:
        _scale = scale
    _rotary_sin = rotary_sin
    _rotary_cos = rotary_cos
    _q_b_weight = q_b_weight
    _q_lora_rank = q_lora_rank
    _q_norm_weight = q_norm_weight
    if rotary_sin is None:
        _rotary_sin = torch.empty(0)
    if rotary_cos is None:
        _rotary_cos = torch.empty(0)
    if q_b_weight is None:
        _q_b_weight = torch.empty(0)
        _q_lora_rank = 0
    if q_norm_weight is None:
        _q_norm_weight = torch.empty(0)
    return TensorParallelFusedMultiLatentCacheAttention.apply(
        hidden_states, q_a_weight, _q_norm_weight, _q_b_weight,
        kv_a_weight, kv_norm_weight,
        k_b_weight, v_b_weight, o_weight,
        _rotary_sin, _rotary_cos,
        seqstarts, kvstarts, cachestarts,
        start_pos, decoding_batches,
        max_seqlen, max_kvlen,
        cache, _scale, attn_mask,
        proc_group,
        num_heads, hidden_dim,
        _q_lora_rank, kv_lora_rank,
        head_dim, rotray_dim,
        is_causal, is_interleaved_rotary,
        softmax_scale,
        num_kv_heads, vo_head_dim,
        num_layer, layer_idx,
        quant_bit, quant_group,
        cache_mode, cache_layout,
        page_size)


if __name__ == "__main__":
    class TestModule(torch.nn.Module):
        def __init__(self,
                    num_heads: int, hidden_dim: int,
                    q_lora_rank: int, kv_lora_rank: int,
                    head_dim: int, rotray_dim: int,
                    is_causal: bool = True,
                    is_interleaved_rotary: bool = True,
                    num_kv_heads: int = 0,
                    vo_head_dim: int = 0,
                    num_layer: int = 1, layer_idx: int = 0,
                    quant_bit: int = 0, quant_group: int = 8,
                    cache_mode: int = 0, cache_layout: int = 0,
                    page_size: int = 128) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.is_causal = is_causal
            self.q_lora_rank = q_lora_rank
            self.kv_lora_rank = kv_lora_rank
            self.head_dim = head_dim
            self.vo_head_dim = vo_head_dim
            self.rotray_dim = rotray_dim
            self.is_interleaved_rotary = is_interleaved_rotary
            self.num_layer = num_layer
            self.layer_idx = layer_idx
            self.quant_bit = quant_bit
            self.quant_group = quant_group
            self.cache_mode = cache_mode
            self.cache_layout = cache_layout
            self.page_size = page_size

            if q_lora_rank == 0:
                self.q_a_weight = torch.nn.Parameter(torch.randn(num_heads * head_dim, hidden_dim))
                self.register_parameter("q_norm_weight", None)
                self.register_parameter("q_b_weight", None)
            else:
                self.q_a_weight = torch.nn.Parameter(torch.randn(q_lora_rank, hidden_dim))
                self.q_norm_weight = torch.nn.Parameter(torch.randn(q_lora_rank))
                self.q_b_weight = torch.nn.Parameter(torch.randn(num_heads * head_dim, q_lora_rank))

            self.kv_a_weight = torch.nn.Parameter(torch.randn(kv_lora_rank + rotray_dim, hidden_dim))
            self.kv_norm_weight = torch.nn.Parameter(torch.randn(kv_lora_rank))
            self.k_b_weight = torch.nn.Parameter(torch.randn(num_heads * (head_dim - rotray_dim), kv_lora_rank))
            self.v_b_weight = torch.nn.Parameter(torch.randn(num_heads * vo_head_dim, kv_lora_rank))

            self.o_weight = torch.nn.Parameter(torch.randn(hidden_dim, num_heads * vo_head_dim))


        @torch.inference_mode()
        def forward(self,
                    hidden_states: torch.Tensor,
                    rotary_sin: Optional[torch.Tensor], rotary_cos: Optional[torch.Tensor],
                    seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                    start_pos: torch.Tensor, decoding_batches: torch.Tensor,
                    max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                    cache: torch.Tensor, scale: Optional[torch.Tensor]):
            return tensor_parallel_fused_multi_head_cache_attention(
                    hidden_states, self.q_a_weight, self.q_norm_weight, self.q_b_weight,
                    self.kv_a_weight, self.kv_norm_weight,
                    self.k_b_weight, self.v_b_weight, self.o_weight,
                    rotary_sin, rotary_cos,
                    seqstarts, kvstarts, cachestarts,
                    start_pos, decoding_batches,
                    max_seqlen, max_kvlen,
                    cache, scale, None,
                    None,
                    self.num_heads, self.hidden_dim, self.q_lora_rank, self.kv_lora_rank,
                    self.head_dim, self.rotray_dim,
                    self.is_causal, self.is_interleaved_rotary,
                    self.num_kv_heads, self.vo_head_dim, 
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
    hidden_dim = 5120
    num_layer = 2
    layer_idx = 1
    cache_layout = 3

    num_heads = 128
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    q_load_rank = 1536
    qk_nope_head_dim = 128
    v_head_dim = 128

    tensor_type = torch.float
    # tensor_type = torch.float16

    hidden_states = torch.randn(batch * seqlen, hidden_dim, dtype=tensor_type)
    seqstarts = torch.tensor([0, seqlen, seqlen], dtype=torch.int64).cumsum(dim=0)
    decoding_batches = torch.tensor([0], dtype=torch.int64)
    start_pos = torch.full([batch], 0, dtype=torch.int64)
    rope_sin = torch.randn(genlen, qk_rope_head_dim, dtype=tensor_type)

    cache = torch.zeros([num_layer, 1, 1, batch * genlen, kv_lora_rank + qk_rope_head_dim], dtype=tensor_type)
    cachestarts = torch.arange(0, batch * genlen, genlen, dtype=torch.int64)

    test_op1 = TestModule(num_heads, hidden_dim, q_load_rank, kv_lora_rank, qk_nope_head_dim + qk_rope_head_dim, qk_rope_head_dim,
                     True, True, num_heads, v_head_dim, num_layer, layer_idx, 0, 8, 0, cache_layout)

    output = test_op1.forward(hidden_states, rope_sin, rope_sin, seqstarts, seqstarts, cachestarts, start_pos, decoding_batches, seqlen, seqlen, cache, None)

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (hidden_states, rope_sin, rope_sin, seqstarts, seqstarts, cachestarts, start_pos, decoding_batches, seqlen, seqlen, cache, None),
        "TensorParallelFusedMultiLatentCacheAttention1.onnx", opset_version=11)

    print(model_str1)

