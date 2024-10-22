import sys
import os

import torch
from torch import nn
import torch.distributed as dist

from typing import Mapping, Any, Optional

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

import torch_function as OPMX
from Params import DeepSeekV2Params
import ModelUtils
from ModelParallel import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from ModelParallel import MoeColumnParallelLinear, MoeRowParallelLinear
from ModelLayers import SkipRMSNorm, RMSNorm

TensorDumper = ModelUtils.__TensorDumper__()

class Attention(nn.Module):
    def __init__(
            self,
            args: DeepSeekV2Params,
            layer_id: int,
            proc_group: dist.ProcessGroup):
        super().__init__()

        self.args = args
        self.proc_group = proc_group

        tp_world_size = 1 if proc_group is None else proc_group.size()
        self.tp_num_heads = args.num_heads // tp_world_size

        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.layer_id = layer_id

        if args.q_lora_rank > 0:
            self.q_a_proj = ColumnParallelLinear( # 一定DP
                    None, args.hidden_dim, args.q_lora_rank,
                    bias_term=False, gather_output=False)
            self.q_a_layernorm = RMSNorm(args.q_lora_rank) # 一定DP
            self.q_b_proj = ColumnParallelLinear( # 要么TP要么DP
                    proc_group, args.q_lora_rank, self.tp_num_heads * self.q_head_dim,
                    bias_term=False, gather_output=False)
        else:
            self.q_a_proj = ColumnParallelLinear( # 要么TP要么DP
                    proc_group, args.hidden_dim, self.tp_num_heads * self.q_head_dim,
                    bias_term=False, gather_output=False)
        self.kv_a_proj = ColumnParallelLinear( # 一定DP
                None, args.hidden_dim, args.kv_lora_rank + args.qk_rope_head_dim,
                bias_term=False, gather_output=False)
        self.kv_a_layernorm = RMSNorm(args.kv_lora_rank) # 一定DP
        self.k_b_proj = ColumnParallelLinear( # 要么TP要么DP
                proc_group, args.kv_lora_rank, self.tp_num_heads * args.qk_nope_head_dim,
                bias_term=False, gather_output=False)
        self.v_b_proj = ColumnParallelLinear( # 要么TP要么DP
                proc_group, args.kv_lora_rank, self.tp_num_heads * args.v_head_dim,
                bias_term=False, gather_output=False)
        self.o_proj = RowParallelLinear( # 要么TP要么DP
            proc_group, self.tp_num_heads * args.v_head_dim, args.hidden_dim,
            bias_term=False, input_is_parallel=True)


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor],
                rotary_sin: Optional[torch.Tensor], rotary_cos: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                decoding_batches: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                kv_cache: torch.Tensor, kv_scale: torch.Tensor = None):
        _place_holder = torch.empty(0, device=x.device)
        output = OPMX.dynamic_batching.tensor_parallel_fused_multi_head_cache_attention(
            x,
            self.q_a_proj.weight,
            self.q_a_layernorm.weight if self.args.q_lora_rank > 0 else _place_holder,
            self.q_b_proj.weight if self.args.q_lora_rank > 0 else _place_holder,
            self.kv_a_proj.weight, self.kv_a_layernorm.weight,
            self.k_b_proj.weight, self.v_b_proj.weight, self.o_proj.weight,
            rotary_sin, rotary_cos,
            seqstarts, kvstarts, cachestarts,
            start_pos, decoding_batches,
            max_seqlen, max_kvlen,
            kv_cache, kv_scale, attn_mask,
            proc_group=self.proc_group,
            num_heads=self.args.num_heads,
            hidden_dim=self.args.hidden_dim,
            q_lora_rank=self.args.q_lora_rank,
            kv_lora_rank=self.args.kv_lora_rank,
            head_dim=self.q_head_dim,
            rotray_dim=self.args.qk_rope_head_dim,
            is_causal=True,
            is_interleaved_rotary=True,
            num_kv_heads=0,
            vo_head_dim=self.args.v_head_dim,
            num_layer=self.args.num_layers,
            layer_idx=self.layer_id,
            quant_bit=self.args.cache_quant_bit,
            quant_group=self.args.cache_quant_group,
            cache_mode=self.args.cache_mode,
            cache_layout=self.args.cache_layout,
            page_size=self.args.page_size)

        return output

class GateMLPWeight(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        proc_group: dist.ProcessGroup
    ):
        super().__init__()

        self.up_proj = ColumnParallelLinear(
            proc_group, hidden_dim, 2 * intermediate_dim,
            bias_term=False, gather_output=False)
        self.down_proj = RowParallelLinear(
            proc_group, intermediate_dim, hidden_dim,
            bias_term=False, input_is_parallel=True)


    def forward(self):
        assert False, "should use this module to forward"


class GateMoEWeight(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        proc_group: dist.ProcessGroup
    ):
        super().__init__()

        self.up_proj = MoeColumnParallelLinear(
            proc_group, num_experts, hidden_dim, 2 * intermediate_dim,
            bias_term=False, gather_output=False)
        self.down_proj = MoeRowParallelLinear(
            proc_group, num_experts, intermediate_dim, hidden_dim,
            bias_term=False, input_is_parallel=True)


    def forward(self):
        assert False, "should use this module to forward"


class FeedForward(nn.Module):
    def __init__(
        self,
        args: DeepSeekV2Params,
        layer_id: int,
        expert_parallel_mode: str,
        proc_group: dist.ProcessGroup
    ):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        
        self.ep_proc_group = proc_group
        if 'etp' in expert_parallel_mode:
            self.companion_parallel_mode = 'tensor_parallel'
            self.tp_proc_group = proc_group
        else:
            self.companion_parallel_mode = 'data_parallel'
            self.tp_proc_group = None

        ep_world_size = 1 if self.ep_proc_group is None else self.ep_proc_group.size()
        assert args.num_experts % ep_world_size == 0, "{} is not divisible by {}".format(args.num_experts, ep_world_size)
        self.ep_num_experts = args.num_experts // ep_world_size

        tp_world_size = 1 if self.tp_proc_group is None else self.tp_proc_group.size()
        assert args.intermediate_dim % tp_world_size == 0, "{} is not divisible by {}".format(args.intermediate_dim, tp_world_size)
        assert (args.num_shared_experts * args.moe_intermediate_dim) % tp_world_size == 0, "{} is not divisible by {}".format(
            args.num_shared_experts * args.moe_intermediate_dim, tp_world_size)

        if self.layer_id < self.args.num_first_dense_layers:
            self.up_proj = ColumnParallelLinear( # 要么TP要么DP
                self.tp_proc_group, args.hidden_dim, 2 * args.intermediate_dim,
                bias_term=False, gather_output=False)
            self.down_proj = RowParallelLinear( # 要么TP要么DP
                self.tp_proc_group, args.intermediate_dim, args.hidden_dim,
                bias_term=False, input_is_parallel=True)
        else:
            self.gate = ColumnParallelLinear( # 一定是DP
                None, args.hidden_dim, args.num_experts,
                bias_term=False, gather_output=False)
            if self.args.num_shared_experts > 0:
                self.shared_experts = GateMLPWeight( # 要么TP要么DP
                    args.hidden_dim, args.num_shared_experts * args.moe_intermediate_dim,
                    proc_group=self.tp_proc_group)
            self.experts = GateMoEWeight( # EP在前面分了
                self.ep_num_experts, args.hidden_dim, args.moe_intermediate_dim,
                proc_group=None)


    def forward(self, x):
        if self.layer_id < self.args.num_first_dense_layers:
            x = self.up_proj(x)
            x = OPMX.swiglu(x)
            output = self.down_proj(x)
        else:
            bias = torch.empty(0, device=x.device)
            output = OPMX.moe_expert_parallel_feed_forward(
                x,
                self.gate.weight,
                self.experts.up_proj.weight, bias,
                self.experts.down_proj.weight, bias,
                self.shared_experts.up_proj.weight if self.args.num_shared_experts > 0 else None,
                bias if self.num_shared_experts > 0 else None,
                self.shared_experts.down_proj.weight if self.args.num_shared_experts > 0 else None,
                bias if self.num_shared_experts > 0 else None,
                proc_group=self.ep_proc_group,
                hidden_dim=self.args.hidden_dim,
                expert_intermediate_dim=self.args.moe_intermediate_dim,
                num_experts=self.args.num_experts,
                num_experts_per_token=self.args.num_experts_per_token,
                has_feed_forward_gate=True,
                up_porj_bias_term=False,
                down_proj_bias_term=False,
                has_shared_expert=self.args.num_shared_experts > 0,
                shared_intermediate_dim=self.args.num_shared_experts * self.args.moe_intermediate_dim,
                num_expert_groups=self.args.num_expert_groups,
                num_groups_per_token=self.args.moe_topk_group,
                gating_scaling_factor=self.args.moe_scaling_factor,
                gating_normalize_prob=self.args.moe_normalize_prob,
                gating_method=self.args.moe_topk_method,
                activation_method='silu',
                companion_parallel_mode=self.companion_parallel_mode)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int,
                 args: DeepSeekV2Params,
                 expert_parallel_mode: str,
                 proc_group: dist.ProcessGroup):
        super().__init__()

        if 'etp' in expert_parallel_mode:
            tp_proc_group = proc_group
        else:
            tp_proc_group = None

        self.self_attn = Attention(args, # 要么TP要么DP
                                   layer_id,
                                   proc_group=tp_proc_group)
        self.mlp = FeedForward(args,
                               layer_id,
                               expert_parallel_mode,
                               proc_group=proc_group)

        self.layer_id = layer_id
        self.input_layernorm = SkipRMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.post_attention_layernorm = SkipRMSNorm(args.hidden_dim, eps=args.norm_eps)


    def forward(self, x: torch.Tensor, skip: torch.Tensor, attn_mask: Optional[torch.Tensor],
                rotary_sin: Optional[torch.Tensor], rotary_cos: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                decoding_batches: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                kv_cache: torch.Tensor, kv_sacle: torch.Tensor = None):
        norm, x = self.input_layernorm(x, skip)
        # TensorDumper.dump(norm, "layer{}_attention_norm_out".format(self.layer_id))
        # TensorDumper.dump(x, "layer{}_attention_norm_skip_out".format(self.layer_id))
        attn = self.self_attn.forward(norm, attn_mask,
                                      rotary_sin, rotary_cos,
                                      seqstarts, kvstarts,
                                      cachestarts, decoding_batches,
                                      start_pos, max_seqlen, max_kvlen,
                                      kv_cache, kv_sacle)
        norm, h = self.post_attention_layernorm(x, attn)
        # TensorDumper.dump(norm, "layer{}_ffn_norm_out".format(self.layer_id))
        # TensorDumper.dump(h, "layer{}_ffn_norm_skip_out".format(self.layer_id))
        ffn = self.mlp.forward(norm)
        return h, ffn


class Transformer(nn.Module):
    def __init__(self, params: DeepSeekV2Params,
                 expert_parallel_mode: str,
                 proc_group: dist.ProcessGroup):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.num_layers
        self.proc_group = proc_group

        ep_world_size = 1 if proc_group is None else proc_group.size()
        if 'etp' in expert_parallel_mode:
            tp_world_size = ep_world_size
            tp_proc_group = proc_group
        else:
            tp_world_size = 1
            tp_proc_group = None
        self.tp_imm_dim = params.intermediate_dim // tp_world_size
        self.tp_moe_imm_dim = params.moe_intermediate_size // tp_world_size
        self.moe_imm_dim = params.moe_intermediate_size

        self.embed_tokens = ParallelEmbedding(tp_proc_group, params.vocab_size, params.hidden_dim) # 要么TP要么DP

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_layers):
            self.layers.append(TransformerBlock(
                layer_id, params,
                expert_parallel_mode,
                proc_group=proc_group))

        self.norm = SkipRMSNorm(params.hidden_dim, eps=params.norm_eps)
        self.lm_head = ColumnParallelLinear(tp_proc_group, params.hidden_dim, params.vocab_size, bias_term=False)


    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor,
                cachestarts: torch.Tensor, decoding_batches: torch.Tensor,
                start_pos: torch.Tensor, max_seqlen: torch.Tensor,  max_kvlen: torch.Tensor,
                kv_cache: torch.Tensor, kv_scale: torch.Tensor = None):
        h = self.embed_tokens(tokens)
        # TensorDumper.dump(h, "emb_out")

        _kv_scale = kv_scale
        TensorDumper.dump(tokens, "token_ids")
        if attn_mask is not None:
            TensorDumper.dump(attn_mask, "attn_mask")
        if attn_mask is not None and kv_scale is None: # mount an empty scale for friendly exporting
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

        rope_sin, rope_cos = OPMX.rotary_position_coefficient(
            max_kvlen, h.device,
            data_type=h.data_type,
            rotary_dim=self.params.qk_rope_head_dim,
            theta=self.params.rope_theta,
            max_position_embeddings=self.params.max_position_embeddings,
            original_max_position_embeddings=self.params.origin_max_position_embeddings,
            scaling_type=self.params.rope_scaling_type,
            scaling_factor=self.params.rope_scaling_factor,
            scaling_beta_fast=self.params.rope_scaling_beta_fast,
            scaling_beta_slow=self.params.rope_scaling_beta_slow,
            scaling_mscale=self.params.rope_scaling_mscale,
            scaling_mscale_all_dim=self.params.rope_scaling_mscale_all_dim
        )

        norm = None
        for layer in self.layers:
            h, norm = layer(h, norm, attn_mask, rope_sin, rope_cos,
                            seqstarts, kvstarts, cachestarts,
                            decoding_batches, start_pos, max_seqlen, max_kvlen,
                            kv_cache, _kv_scale)

        h, norm = self.norm(h, norm)
        # TensorDumper.dump(h, "last_rms_norm")
        gathered_h = torch.index_select(h, 0, seqstarts[1:] - 1)
        # TensorDumper.dump(gathered_h, "gathered_h")
        output = self.lm_head(gathered_h)  # only compute last logits
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
                if 'kv_b_proj' in key:
                    loaded_params.add(key)
                    k_b_weight, v_b_weight = torch.split(
                        value.view(self.params.num_heads, self.params.qk_nope_head_dim + self.params.v_head_dim, -1),
                        [self.params.qk_nope_head_dim, self.params.v_head_dim], dim=1)
                    k_module_name = module_name.replace('kv_b_proj', 'k_b_proj')
                    self.get_submodule(k_module_name)._parameters[param_name][:] = k_b_weight
                    replaced_key = k_module_name + '.' + param_name
                    print(f'Loaded: {key} -> {replaced_key}[{k_b_weight.shape}]')
                    v_module_name = module_name.replace('kv_b_proj', 'v_b_proj')
                    self.get_submodule(v_module_name)._parameters[param_name][:] = v_b_weight
                    replaced_key = v_module_name + '.' + param_name
                    print(f'Loaded: {key} -> {replaced_key}[{v_b_weight.shape}]')

                if True: # fused ffn
                    if 'mlp.up_proj' in key:
                        loaded_params.add(key)
                        self.get_submodule(module_name)._parameters[param_name][:self.tp_imm_dim] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    if 'mlp.gate_proj' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('gate_proj', 'up_proj')
                        self.get_submodule(module_name)._parameters[param_name][self.tp_imm_dim:] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    if 'mlp.shared_experts.up_proj' in key:
                        loaded_params.add(key)
                        self.get_submodule(module_name)._parameters[param_name][:self.tp_moe_imm_dim] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    if 'mlp.shared_experts.gate_proj' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('gate_proj', 'up_proj')
                        self.get_submodule(module_name)._parameters[param_name][self.tp_moe_imm_dim:] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    if 'mlp.experts.up_proj' in key:
                        loaded_params.add(key)
                        self.get_submodule(module_name)._parameters[param_name][:, :self.moe_imm_dim] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    if 'mlp.experts.gate_proj' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('gate_proj', 'up_proj')
                        self.get_submodule(module_name)._parameters[param_name][:, self.moe_imm_dim:] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
            except AttributeError as e:
                raise Exception(f'Failed to inject model weight {key}, can not find corresponding layer.')

        for key in state_dict:
            if key not in loaded_params:
                print(f'{key} is not loaded.')

    @torch.no_grad()
    def random_weight(self):
        model_params = {key: value for key, value in self.named_parameters()}

        for key, value in model_params.items():
            module_name, param_name = key.rsplit(".", 1)

            self.get_submodule(module_name)._parameters[param_name] = torch.randn_like(value)
            print(f'Random: {key} -> {key}[{value.shape}]')

