import sys
import os

import torch
from torch import nn
import torch.distributed as dist

from typing import Mapping, Any, Optional

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

import torch_function as PMX
from ModelParams import ModelParams
import ModelUtils
from ModelParallel import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding

TensorDumper = ModelUtils.__TensorDumper__()

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return PMX.rms_norm(x, self.weight, -1, self.eps)


class SkipRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, skip):
        return PMX.skip_rms_norm(x, self.weight, skip, -1, self.eps)


class Attention(nn.Module):
    def __init__(
            self,
            args: ModelParams,
            layer_id: int,
            friendly_gqa: bool,
            fused_qkv: bool,
            fused_kvcache: bool,
            attn_wqkv_bias_term: bool,
            attn_wo_bias_term: bool,
            rotary_dim: int,
            proc_group: dist.ProcessGroup):
        super().__init__()

        world_size = 1 if proc_group is None else proc_group.size()

        self.num_kv_heads = args.num_heads if args.num_kv_heads is None else args.num_kv_heads
        self.num_local_heads = args.num_heads // world_size
        self.num_local_kv_heads = self.num_kv_heads // world_size
        self.num_local_kv_repeats = self.num_local_heads // self.num_local_kv_heads
        self.head_dim = args.hidden_dim // args.num_heads
        self.rotary_dim = rotary_dim
        self.num_layers = args.num_layers
        self.layer_id = layer_id
        self.cache_quant_bit = args.cache_quant_bit
        self.cache_quant_group = args.cache_quant_group
        self.cache_layout = args.cache_layout
        self.cache_mode = args.cache_mode

        self.friendly_gqa = friendly_gqa
        self.fused_qkv = fused_qkv
        self.fused_kvcache = fused_kvcache
        self.auto_causal = args.auto_causal

        if self.fused_qkv:
            self.wqkv = ColumnParallelLinear(
                proc_group, args.hidden_dim, args.hidden_dim + 2 * self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
        else:
            self.wq = ColumnParallelLinear(
                proc_group, args.hidden_dim, args.hidden_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
            self.wk = ColumnParallelLinear(
                proc_group, args.hidden_dim, self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
            self.wv = ColumnParallelLinear(
                proc_group, args.hidden_dim, self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
        self.wo = RowParallelLinear(
            proc_group, args.hidden_dim, args.hidden_dim,
            bias_term=attn_wo_bias_term, input_is_parallel=True)


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                decoding_batches: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                kv_cache: torch.Tensor, kv_scale: torch.Tensor = None):
        expanded_shape = (0, -1, self.head_dim)
        if self.fused_qkv:
            xqkv = self.wqkv(x)
            xqkv = PMX.reshape(xqkv, expanded_shape)
            split_size = (self.num_local_heads, self.num_local_kv_heads, self.num_local_kv_heads)
            # TensorDumper.dump(xqkv, "layer{}_reshaped_xqkv".format(self.layer_id))
            xq, xk, xv = torch.split(xqkv, split_size, -2)
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            xq = PMX.reshape(xq, expanded_shape)
            xk = PMX.reshape(xk, expanded_shape)
            xv = PMX.reshape(xv, expanded_shape)
        # TensorDumper.dump(xq, "layer{}_reshaped_xq".format(self.layer_id))
        # TensorDumper.dump(xk, "layer{}_reshaped_xk".format(self.layer_id))
        # TensorDumper.dump(xv, "layer{}_reshaped_xv".format(self.layer_id))


        xq, xk = PMX.dynamic_batching.rotary_position_embedding(
                                        xq, xk, seqstarts,
                                        start_pos, max_seqlen, 
                                        self.rotary_dim)
        # TensorDumper.dump(xq, "layer{}_rotary_position_embedding_out_xq".format(self.layer_id))
        # TensorDumper.dump(xk, "layer{}_rotary_position_embedding_out_xk".format(self.layer_id))

        if self.fused_kvcache:
            attn = PMX.dynamic_batching.multi_head_cache_attention(
                xq, xk, xv, seqstarts, kvstarts,
                cachestarts, start_pos,
                decoding_batches,
                max_seqlen, max_kvlen,
                kv_cache, kv_scale,
                attn_mask=attn_mask,
                num_heads=self.num_local_heads,
                head_dim=self.head_dim,
                is_causal=self.auto_causal,
                num_kv_heads=self.num_local_kv_heads,
                num_layer=self.num_layers,
                layer_idx=self.layer_id,
                quant_bit=self.cache_quant_bit,
                quant_group=self.cache_quant_group,
                cache_mode=self.cache_mode,
                cache_layout=self.cache_layout)
        else:
            keys, values = PMX.dynamic_batching.key_value_cache(
                                            xk, xv, seqstarts, kvstarts,
                                            cachestarts, start_pos,
                                            max_seqlen, max_kvlen,
                                            kv_cache, kv_scale,
                                            num_layer=self.num_layers,
                                            layer_idx=self.layer_id,
                                            quant_bit=self.cache_quant_bit,
                                            quant_group=self.cache_quant_group,
                                            num_repeat=self.num_local_kv_repeats if self.friendly_gqa else 1,
                                            cache_mode=self.cache_mode,
                                            cache_layout=self.cache_layout)
            # TensorDumper.dump(kv_cache, "layer{}_modified_kv_cache".format(self.layer_id))
            # TensorDumper.dump(kv_scale, "layer{}_modified_kv_scale".format(self.layer_id))
            # TensorDumper.dump(keys, "layer{}_key_value_cache_out_keys".format(self.layer_id))
            # TensorDumper.dump(values, "layer{}_key_value_cache_out_values".format(self.layer_id))
            attn = PMX.dynamic_batching.multi_head_attention(
                                            xq, keys, values,
                                            seqstarts, kvstarts,
                                            decoding_batches,
                                            max_seqlen, max_kvlen,
                                            attn_mask=attn_mask,
                                            num_heads=self.num_local_heads,
                                            head_dim=self.head_dim,
                                            is_causal=self.auto_causal,
                                            num_kv_heads=0 if self.friendly_gqa else self.num_local_kv_heads)
        attn = PMX.reshape(attn, (0, -1))
        # TensorDumper.dump(attn, "layer{}_multi_head_attention_out".format(self.layer_id))

        output = self.wo(attn)
        # TensorDumper.dump(output, "layer{}_reshaped_wo_out".format(self.layer_id))

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelParams,
        layer_id: int,
        fused_ffn_glu: bool,
        linear_bias_term: bool,
        proc_group: dist.ProcessGroup
    ):
        super().__init__()
        self.layer_id = layer_id
        self.fused_ffn_glu = fused_ffn_glu

        if self.fused_ffn_glu:
            self.wu = ColumnParallelLinear(
                proc_group, args.hidden_dim, 2 * args.intermediate_dim,
                bias_term=linear_bias_term, gather_output=False)
        else:
            self.w1 = ColumnParallelLinear(
                proc_group, args.hidden_dim, args.intermediate_dim,
                bias_term=linear_bias_term, gather_output=False)
            self.w3 = ColumnParallelLinear(
                proc_group, args.hidden_dim, args.intermediate_dim,
                bias_term=linear_bias_term, gather_output=False)
        self.w2 = RowParallelLinear(
            proc_group, args.intermediate_dim, args.hidden_dim,
            bias_term=linear_bias_term, input_is_parallel=True)


    def forward(self, x):
        if self.fused_ffn_glu:
            x13 = self.wu(x)
            # TensorDumper.dump(x13, "layer{}_ffn_wu".format(self.layer_id))
            x13 = PMX.swiglu(x13)
        else:
            x1 = self.w1(x)
            # TensorDumper.dump(x1, "layer{}_ffn_w1".format(self.layer_id))
            x3 = self.w3(x)
            # TensorDumper.dump(x3, "layer{}_ffn_w3".format(self.layer_id))
            x13 = PMX.silu(x1, x3)
        # TensorDumper.dump(x13, "layer{}_ffn_mul_silu".format(self.layer_id))
        output = self.w2(x13)
        # TensorDumper.dump(output, "layer{}_ffn_w2".format(self.layer_id))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int,
                 args: ModelParams,
                 friendly_gqa: bool,
                 fused_qkv: bool,
                 fused_kvcache: bool,
                 fused_ffn_glu: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 rotary_dim: int,
                 proc_group: dist.ProcessGroup):
        super().__init__()
        self.attention = Attention(args,
                                   layer_id,
                                   friendly_gqa,
                                   fused_qkv,
                                   fused_kvcache,
                                   attn_wqkv_bias_term,
                                   attn_wo_bias_term,
                                   rotary_dim=rotary_dim,
                                   proc_group=proc_group)
        self.feed_forward = FeedForward(args, 
                                        layer_id,
                                        fused_ffn_glu,
                                        ffn_linear_bias_term,
                                        proc_group=proc_group)

        self.layer_id = layer_id
        self.attention_norm = SkipRMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.ffn_norm = SkipRMSNorm(args.hidden_dim, eps=args.norm_eps)


    def forward(self, x: torch.Tensor, skip: torch.Tensor, attn_mask: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                decoding_batches: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                kv_cache: torch.Tensor, kv_sacle: torch.Tensor = None):
        norm, x = self.attention_norm(x, skip)
        # TensorDumper.dump(norm, "layer{}_attention_norm_out".format(self.layer_id))
        # TensorDumper.dump(x, "layer{}_attention_norm_skip_out".format(self.layer_id))
        attn = self.attention.forward(norm, attn_mask, seqstarts, kvstarts,
                                      cachestarts, decoding_batches,
                                      start_pos, max_seqlen, max_kvlen,
                                      kv_cache, kv_sacle)
        norm, h = self.ffn_norm(x, attn)
        # TensorDumper.dump(norm, "layer{}_ffn_norm_out".format(self.layer_id))
        # TensorDumper.dump(h, "layer{}_ffn_norm_skip_out".format(self.layer_id))
        ffn = self.feed_forward.forward(norm)
        return h, ffn


class Transformer(nn.Module):
    def __init__(self, params: ModelParams,
                 friendly_gqa: bool,
                 fused_qkv: bool,
                 fused_kvcache: bool,
                 fused_ffn_glu: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 rotary_dim: int,
                 proc_group: dist.ProcessGroup):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.num_layers
        self.proc_group = proc_group
        self.fused_qkv = fused_qkv
        self.fused_kvcache = fused_kvcache
        self.fused_ffn_glu = fused_ffn_glu

        world_size = 1 if proc_group is None else proc_group.size()
        num_kv_heads = params.num_heads if params.num_kv_heads is None else params.num_kv_heads
        num_local_heads = params.num_heads // world_size
        num_local_kv_heads = num_kv_heads // world_size
        head_dim = params.hidden_dim // params.num_heads
        self.local_q_dim = num_local_heads * head_dim
        self.local_kv_dim = num_local_kv_heads * head_dim
        self.local_imm_dim = params.intermediate_dim // world_size 

        self.tok_embeddings = ParallelEmbedding(proc_group, params.vocab_size, params.hidden_dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_layers):
            self.layers.append(TransformerBlock(
                layer_id, params,
                friendly_gqa,
                fused_qkv,
                fused_kvcache,
                fused_ffn_glu,
                attn_wqkv_bias_term,
                attn_wo_bias_term,
                ffn_linear_bias_term,
                rotary_dim,
                proc_group=proc_group))

        self.norm = SkipRMSNorm(params.hidden_dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(proc_group, params.hidden_dim, params.vocab_size, bias_term=False)


    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor,
                cachestarts: torch.Tensor, decoding_batches: torch.Tensor,
                start_pos: torch.Tensor, max_seqlen: torch.Tensor,  max_kvlen: torch.Tensor,
                kv_cache: torch.Tensor, kv_scale: torch.Tensor = None):
        h = self.tok_embeddings(tokens)
        # TensorDumper.dump(h, "emb_out")

        _kv_scale = kv_scale
        TensorDumper.dump(tokens, "token_ids")
        if attn_mask is not None:
            TensorDumper.dump(attn_mask, "attn_mask")
        if self.fused_kvcache and attn_mask is not None:
            if kv_scale is None: # mount an empty scale for friendly exporting
                _kv_scale = torch.empty(0, dtype=h.dtype)
        TensorDumper.dump(seqstarts, "seqstarts")
        TensorDumper.dump(kvstarts, "kvstarts")
        TensorDumper.dump(cachestarts, "cachestarts")
        TensorDumper.dump(decoding_batches, "decoding_batches")
        TensorDumper.dump(start_pos, "start_pos")
        TensorDumper.dump(max_seqlen, "max_seqlen")
        TensorDumper.dump(max_kvlen, "max_kvlen")
        TensorDumper.dump(kv_cache, "kv_cache")
        if kv_scale is not None:
            TensorDumper.dump(kv_scale, "kv_scale")

        norm = None
        for layer in self.layers:
            h, norm = layer(h, norm, attn_mask, seqstarts, kvstarts, cachestarts,
                            decoding_batches, start_pos, max_seqlen, max_kvlen,
                            kv_cache, _kv_scale)

        h, norm = self.norm(h, norm)
        # TensorDumper.dump(h, "last_rms_norm")
        gathered_h = torch.index_select(h, 0, seqstarts[1:] - 1)
        # TensorDumper.dump(gathered_h, "gathered_h")
        output = self.output(gathered_h)  # only compute last logits
        # TensorDumper.dump(output, "logits_before_cast")
        output = output.float()
        TensorDumper.dump(output, "logits")
        return output
    

    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, Any]):
        loaded_params = set()
        model_params = {key: value for key, value in self.named_parameters()}

        for key, value in state_dict.items():
            module_name, param_name = key.rsplit(".", 1)

            if key in model_params:
                self.get_submodule(module_name)._parameters[param_name][:] = value
                loaded_params.add(key)
                print(f'Loaded: {key} -> {key}[{value.shape}]')

            try:
                if self.fused_qkv:
                    if 'attention.wq' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('wq', 'wqkv')
                        self.get_submodule(module_name)._parameters[param_name][
                            :self.local_q_dim] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    elif 'attention.wk' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('wk', 'wqkv')
                        self.get_submodule(module_name)._parameters[param_name][
                            self.local_q_dim:self.local_q_dim + self.local_kv_dim] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    elif 'attention.wv' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('wv', 'wqkv')
                        self.get_submodule(module_name)._parameters[param_name][
                            self.local_q_dim + self.local_kv_dim:
                            self.local_q_dim + self.local_kv_dim * 2] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                if self.fused_ffn_glu:
                    if 'feed_forward.w1' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('w1', 'wu')
                        self.get_submodule(module_name)._parameters[param_name][
                            :self.local_imm_dim] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    if 'feed_forward.w3' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('w3', 'wu')
                        self.get_submodule(module_name)._parameters[param_name][
                            self.local_imm_dim:] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
            except AttributeError as e:
                raise Exception(f'Failed to inject model weight {key}, can not find corresponding layer.')
        
        for key in state_dict:
            if key not in loaded_params:
                print(f'{key} is not loaded.')
