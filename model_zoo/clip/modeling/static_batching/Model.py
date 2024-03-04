import sys
import os

import torch
from torch import nn
import torch.distributed as dist

from typing import Mapping, Any, Optional

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

import torch_function as PMX
from ModelParams import ModelParams, VisionModelParams
import ModelUtils
from ModelParallel import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from ModelLayers import Linear

TensorDumper = ModelUtils.__TensorDumper__()

class LayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return PMX.layer_norm(x, self.weight, self.bias, -1, self.eps)

class SkipLayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, X: torch.Tensor, SkipIn: torch.Tensor):
        return PMX.skip_layer_norm(X, self.weight, self.bias, SkipIn, -1, self.eps)


class VisionEmbeddings(torch.nn.Module):
    def __init__(self, hidden_dim: int, image_size: int, patch_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_positions = (self.image_size // self.patch_size) ** 2 + 1

        self.cls_emb_weight = nn.Parameter(torch.randn(self.hidden_dim))
        self.patch_emb_weight  = nn.Parameter(torch.randn([self.hidden_dim, 3, patch_size, patch_size]))
        self.pos_emb_weight = nn.Parameter(torch.randn(self.num_positions, self.hidden_dim))

    def forward(self, pixel_values: torch.Tensor):
        return PMX.vision_embedding(pixel_values, self.cls_emb_weight, self.patch_emb_weight, self.pos_emb_weight, self.hidden_dim, self.image_size, self.patch_size)


class Attention(nn.Module):
    def __init__(
            self,
            args: ModelParams,
            layer_id: int,
            friendly_gqa: bool,
            fused_qkv: bool,
            attn_wqkv_bias_term: bool,
            attn_wo_bias_term: bool,
            proc_group: dist.ProcessGroup):
        super().__init__()

        world_size = 1 if proc_group is None else proc_group.size()

        self.num_kv_heads = args.num_heads if args.num_kv_heads is None else args.num_kv_heads
        self.num_local_heads = args.num_heads // world_size
        self.num_local_kv_heads = self.num_kv_heads // world_size
        self.num_local_kv_repeats = self.num_local_heads // self.num_local_kv_heads
        self.head_dim = args.hidden_dim // args.num_heads
        self.num_layers = args.num_layers
        self.layer_id = layer_id
        self.friendly_gqa = friendly_gqa
        self.fused_qkv = fused_qkv
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


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        expanded_shape = (0, 0, -1, self.head_dim)
        if self.fused_qkv:
            xqkv = self.wqkv(x)
            xqkv = PMX.reshape(xqkv, expanded_shape)
            # TensorDumper.dump(xqkv, "layer{}_reshaped_xqkv".format(self.layer_id))
            split_size = (self.num_local_heads, self.num_local_kv_heads, self.num_local_kv_heads)
            xq, xk, xv = torch.split(xqkv, split_size, -2)
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            xq = PMX.reshape(xq, expanded_shape)
            xk = PMX.reshape(xk, expanded_shape)
            xv = PMX.reshape(xv, expanded_shape)
        # TensorDumper.dump(xq, "layer{}_reshaped_xq".format(self.layer_id))
        # TensorDumper.dump(xk, "layer{}_reshaped_xk".format(self.layer_id))
        # TensorDumper.dump(xv, "layer{}_reshaped_xv".format(self.layer_id))


        attn = PMX.multi_head_attention(xq, xk, xv,
                                        attn_mask=attn_mask,
                                        num_heads=self.num_local_heads,
                                        head_dim=self.head_dim,
                                        is_causal=self.auto_causal,
                                        num_kv_heads=0 if self.friendly_gqa else self.num_local_kv_heads)
        # TensorDumper.dump(attn, "layer{}_multi_head_attention_out".format(self.layer_id))

        output = self.wo(PMX.reshape(attn, (0, 0, -1)))
        # TensorDumper.dump(output, "layer{}_reshaped_wo_out".format(self.layer_id))

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelParams,
        layer_id: int,
        linear_bias_term: bool,
        proc_group: dist.ProcessGroup
    ):
        super().__init__()
        self.layer_id = layer_id

        self.w1 = ColumnParallelLinear(
            proc_group, args.hidden_dim, args.intermediate_dim,
            bias_term=linear_bias_term, gather_output=False)
        self.w2 = RowParallelLinear(
            proc_group, args.intermediate_dim, args.hidden_dim,
            bias_term=linear_bias_term, input_is_parallel=True)


    def forward(self, x):
        x1 = self.w1(x)
        # x1 = PMX.quickgelu(x1)
        x1 = PMX.swish(x1, beta=1.702)
        # TensorDumper.dump(x1, "layer{}_ffn_w1".format(self.layer_id))
        output = self.w2(x1)
        # TensorDumper.dump(output, "layer{}_ffn_w2".format(self.layer_id))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int,
                 args: ModelParams,
                 friendly_gqa: bool,
                 fused_qkv: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 proc_group: dist.ProcessGroup):
        super().__init__()
        self.attention = Attention(args,
                                   layer_id,
                                   friendly_gqa,
                                   fused_qkv,
                                   attn_wqkv_bias_term,
                                   attn_wo_bias_term,
                                   proc_group=proc_group)
        self.feed_forward = FeedForward(args,
                                        layer_id,
                                        ffn_linear_bias_term,
                                        proc_group=proc_group)

        self.layer_id = layer_id
        self.attention_norm = SkipLayerNorm(args.hidden_dim, eps=args.norm_eps)
        self.ffn_norm = SkipLayerNorm(args.hidden_dim, eps=args.norm_eps)


    def forward(self, x: torch.Tensor, skip: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        norm, res1 = self.attention_norm(x, skip) # res1 = input_x when skip==None
        # TensorDumper.dump(norm, "layer{}_attention_norm_out".format(self.layer_id))
        # TensorDumper.dump(x, "layer{}_attention_norm_skip_out".format(self.layer_id))
        attn = self.attention.forward(norm, attn_mask)
        norm, res2 = self.ffn_norm(attn, res1) # res2 = attn + res1
        # TensorDumper.dump(norm, "layer{}_ffn_norm_out".format(self.layer_id))
        # TensorDumper.dump(h, "layer{}_ffn_norm_skip_out".format(self.layer_id))
        ffn = self.feed_forward.forward(norm)
        return ffn, res2


class ClipVisionTransformer(nn.Module):
    def __init__(self, params: ModelParams,
                 friendly_gqa: bool,
                 fused_qkv: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 proc_group: dist.ProcessGroup):
        super().__init__()
        self.params = params
        self.n_layers = params.num_layers
        self.proc_group = proc_group
        self.fused_qkv = fused_qkv

        world_size = 1 if proc_group is None else proc_group.size()
        num_kv_heads = params.num_heads if params.num_kv_heads is None else params.num_kv_heads
        num_local_heads = params.num_heads // world_size
        num_local_kv_heads = num_kv_heads // world_size
        head_dim = params.hidden_dim // params.num_heads
        self.local_q_dim = num_local_heads * head_dim
        self.local_kv_dim = num_local_kv_heads * head_dim

        self.vision_embeddings = VisionEmbeddings(params.hidden_dim, params.image_size, params.patch_size)
        self.pre_layernorm = LayerNorm(params.hidden_dim, eps=params.norm_eps)
        self.post_layernorm = LayerNorm(params.hidden_dim, eps=params.norm_eps)
        self.vision_projection = Linear(params.hidden_dim, params.projection_dim, bias_term=False)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_layers):
            self.layers.append(TransformerBlock(
                layer_id, params,
                friendly_gqa,
                fused_qkv,
                attn_wqkv_bias_term,
                attn_wo_bias_term,
                ffn_linear_bias_term,
                proc_group=proc_group))


    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        h = self.vision_embeddings(pixel_values)
        h = self.pre_layernorm(h)
        # TensorDumper.dump(h, "emb_out")

        if attn_mask is not None:
            TensorDumper.dump(attn_mask, "attn_mask")

        norm = None
        for layer in self.layers:
            h, norm = layer(h, norm, attn_mask)

        pooled_output = (norm+h)[:, 0, :] # get cls token
        output = self.post_layernorm(pooled_output)
        # output = self.vision_projection(output)
        # h, norm = self.norm(h, norm)
        # TensorDumper.dump(h, "last_rms_norm")
        # output = self.output(h[:, -1, :])  # only compute last logits
        # TensorDumper.dump(output, "logits_before_cast")
        # output = output.float()
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
            except AttributeError as e:
                raise Exception(f'Failed to inject model weight {key}, can not find corresponding layer.')

        for key in state_dict:
            if key not in loaded_params:
                print(f'{key} is not loaded.')
