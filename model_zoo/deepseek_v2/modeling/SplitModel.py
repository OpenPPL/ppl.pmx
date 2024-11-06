import argparse
import gc
import json
import os
import shutil
import warnings

import torch

from pathlib import Path


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def split_pmx_model_etp(model_path, input_base_path, num_shards):
    os.makedirs(model_path, exist_ok=True)
    params = read_json((os.path.join(input_base_path, "opmx_params.json")))

    # weight sharding
    hidden_dim = params['hidden_dim']
    v_head_dim = params['v_head_dim']
    q_lora_rank = params['q_lora_rank']
    kv_lora_rank = params['kv_lora_rank']
    intermediate_dim = params['intermediate_dim']
    shared_expert_intermediate_size = params['moe_intermediate_dim'] * params['num_shared_experts']
    moe_intermediate_size = params['moe_intermediate_dim']
    n_heads_per_shard = params['num_heads'] // num_shards
    n_expert = params['num_experts']

    num_kv_heads = params['num_kv_heads'] if 'num_kv_heads' in params else params['num_heads']
    #num_kv_heads_per_shard = num_kv_heads // num_shards

    #dims_per_head = hidden_dim // params['num_heads']
    q_dims_per_head = params['qk_nope_head_dim'] + params['qk_rope_head_dim']
    kv_dims_per_head = params['qk_nope_head_dim'] + params['v_head_dim']


    #key_value_dim = dims_per_head * num_kv_heads

    write_json(params, os.path.join(model_path, "opmx_params.json"))

    state_dict = {}
    for ckpt_path in sorted(Path(input_base_path).glob("*.pth")):
        state_dict.update(torch.load(ckpt_path, map_location='cpu'))

    for layer_i in range(params['num_first_dense_layers'], params['num_layers']):

        if not params['q_lora_rank']:
            q_proj = state_dict[f"layers.{layer_i}.self_attn.q_proj.weight"].reshape(n_heads_per_shard*num_shards, q_dims_per_head, hidden_dim).split([n_heads_per_shard]*num_shards, dim=0)
            q_proj = [w.reshape(-1, hidden_dim) for w in q_proj]
            state_dict.update({f"layers.{layer_i}.self_attn.q_proj.weight": q_proj})
        else:
            q_b_proj = state_dict[f"layers.{layer_i}.self_attn.q_b_proj.weight"].reshape(n_heads_per_shard*num_shards, q_dims_per_head, q_lora_rank).split([n_heads_per_shard]*num_shards, dim=0)
            q_b_proj = [w.reshape(-1, q_lora_rank) for w in q_b_proj]
            state_dict.update({f"layers.{layer_i}.self_attn.q_b_proj.weight": q_b_proj})

        kv_b_proj = state_dict[f"layers.{layer_i}.self_attn.kv_b_proj.weight"].reshape(n_heads_per_shard*num_shards, kv_dims_per_head, kv_lora_rank).split([n_heads_per_shard]*num_shards, dim=0)
        kv_b_proj = [w.reshape(-1, kv_lora_rank) for w in kv_b_proj]

        o_proj = state_dict[f"layers.{layer_i}.self_attn.o_proj.weight"].split([v_head_dim * num_kv_heads // num_shards]*num_shards, dim=1)

        # shared expert
        share_gate_proj = state_dict[f"layers.{layer_i}.mlp.shared_experts.gate_proj.weight"].split([shared_expert_intermediate_size // num_shards]*num_shards, dim=-2)
        share_down_proj = state_dict[f"layers.{layer_i}.mlp.shared_experts.down_proj.weight"].split([shared_expert_intermediate_size // num_shards]*num_shards, dim=-1)
        share_up_proj = state_dict[f"layers.{layer_i}.mlp.shared_experts.up_proj.weight"].split([shared_expert_intermediate_size // num_shards]*num_shards, dim=-2)

        # expert ep
        gate_proj = state_dict[f"layers.{layer_i}.mlp.experts.gate_proj.weight"].split([n_expert//num_shards]*num_shards, dim=0)
        down_proj = state_dict[f"layers.{layer_i}.mlp.experts.down_proj.weight"].split([n_expert//num_shards]*num_shards, dim=0)
        up_proj = state_dict[f"layers.{layer_i}.mlp.experts.up_proj.weight"].split([n_expert//num_shards]*num_shards, dim=0)

        state_dict.update({
            #f"layers.{layer_i}.self_attn.q_proj.weight": q_proj,
            f"layers.{layer_i}.self_attn.kv_b_proj.weight": kv_b_proj,
            f"layers.{layer_i}.self_attn.o_proj.weight": o_proj,

            f"layers.{layer_i}.mlp.shared_experts.gate_proj.weight": share_gate_proj,
            f"layers.{layer_i}.mlp.shared_experts.down_proj.weight": share_down_proj,
            f"layers.{layer_i}.mlp.shared_experts.up_proj.weight": share_up_proj,

            f"layers.{layer_i}.mlp.experts.gate_proj.weight": gate_proj,
            f"layers.{layer_i}.mlp.experts.down_proj.weight": down_proj,
            f"layers.{layer_i}.mlp.experts.up_proj.weight": up_proj,
        })


    for layer_i in range(params['num_first_dense_layers']):
        if not params['q_lora_rank']:
            q_proj = state_dict[f"layers.{layer_i}.self_attn.q_proj.weight"].reshape(n_heads_per_shard*num_shards, q_dims_per_head, hidden_dim).split([n_heads_per_shard]*num_shards, dim=0)
            q_proj = [w.reshape(-1, hidden_dim) for w in q_proj]
            state_dict.update({f"layers.{layer_i}.self_attn.q_proj.weight": q_proj})
        else:
            q_b_proj = state_dict[f"layers.{layer_i}.self_attn.q_b_proj.weight"].reshape(n_heads_per_shard*num_shards, q_dims_per_head, q_lora_rank).split([n_heads_per_shard]*num_shards, dim=0)
            q_b_proj = [w.reshape(-1, q_lora_rank) for w in q_b_proj]
            state_dict.update({f"layers.{layer_i}.self_attn.q_b_proj.weight": q_b_proj})

        kv_b_proj = state_dict[f"layers.{layer_i}.self_attn.kv_b_proj.weight"].reshape(n_heads_per_shard*num_shards, kv_dims_per_head, kv_lora_rank).split([n_heads_per_shard]*num_shards, dim=0)
        kv_b_proj = [w.reshape(-1, kv_lora_rank) for w in kv_b_proj]

        o_proj = state_dict[f"layers.{layer_i}.self_attn.o_proj.weight"].split([v_head_dim * num_kv_heads // num_shards]*num_shards, dim=1)

        gate_proj = state_dict[f"layers.{layer_i}.mlp.gate_proj.weight"].split([intermediate_dim // num_shards]*num_shards, dim=-2)
        down_proj = state_dict[f"layers.{layer_i}.mlp.down_proj.weight"].split([intermediate_dim // num_shards]*num_shards, dim=-1)
        up_proj = state_dict[f"layers.{layer_i}.mlp.up_proj.weight"].split([intermediate_dim // num_shards]*num_shards, dim=-2)

        state_dict.update({
            #f"layers.{layer_i}.self_attn.q_proj.weight": q_proj,
            f"layers.{layer_i}.self_attn.kv_b_proj.weight": kv_b_proj,
            f"layers.{layer_i}.self_attn.o_proj.weight": o_proj,

            f"layers.{layer_i}.mlp.gate_proj.weight": gate_proj,
            f"layers.{layer_i}.mlp.down_proj.weight": down_proj,
            f"layers.{layer_i}.mlp.up_proj.weight": up_proj,})


    token_emb_weight = state_dict["tok_embeddings.weight"].split([hidden_dim // num_shards]*num_shards, dim=1)
    output_weight = state_dict["lm_head.weight"].split([params['vocab_size'] // num_shards]*num_shards, dim=0)
    state_dict.update({
        "tok_embeddings.weight": token_emb_weight,
        "lm_head.weight": output_weight
    })

    # only split ColParallelLinear bias
    # for key in state_dict.keys():
    #     if 'wo.bias' in key or 'w2.bias' in key: continue
    #     if 'bias' in key:
    #         bias_dim = state_dict[key].shape[0]
    #         split_bias = state_dict[key].split([bias_dim // num_shards]*num_shards)
    #         state_dict.update({key: split_bias})

    # dump weight
    tmp_weight_list = [{} for _ in range(num_shards)]
    for key, value in state_dict.items():
        for idx, w_dict in enumerate(tmp_weight_list):
            if torch.is_tensor(value):
                tmp_weight_list[idx].update({key:value.clone()})
            else:
                tmp_weight_list[idx].update({key:value[idx].clone()})
    for idx, weight_dict in enumerate(tmp_weight_list):
        torch.save(weight_dict, os.path.join(model_path, f"model.{idx:02d}.pth"))


def split_pmx_model_edp(model_path, input_base_path, num_shards):
    os.makedirs(model_path, exist_ok=True)
    params = read_json((os.path.join(input_base_path, "opmx_params.json")))

    # weight sharding
    # hidden_dim = params['hidden_dim']
    # v_head_dim = params['v_head_dim']
    # q_lora_rank = params['q_lora_rank']
    # kv_lora_rank = params['kv_lora_rank']
    # intermediate_dim = params['intermediate_dim']
    # shared_expert_intermediate_size = params['moe_intermediate_dim'] * params['num_shared_experts']
    # moe_intermediate_size = params['moe_intermediate_dim']
    n_expert = params['num_experts']

    # num_kv_heads = params['num_kv_heads'] if 'num_kv_heads' in params else params['num_heads']

    #dims_per_head = hidden_dim // params['num_heads']
    # q_dims_per_head = params['qk_nope_head_dim'] + params['qk_rope_head_dim']
    # kv_dims_per_head = params['qk_nope_head_dim'] + params['v_head_dim']


    #key_value_dim = dims_per_head * num_kv_heads

    write_json(params, os.path.join(model_path, "opmx_params.json"))

    state_dict = {}
    for ckpt_path in sorted(Path(input_base_path).glob("*.pth")):
        state_dict.update(torch.load(ckpt_path, map_location='cpu'))

    for layer_i in range(params['num_first_dense_layers'], params['num_layers']):
        # expert ep
        gate_proj = state_dict[f"layers.{layer_i}.mlp.experts.gate_proj.weight"].split([n_expert//num_shards]*num_shards, dim=0)
        down_proj = state_dict[f"layers.{layer_i}.mlp.experts.down_proj.weight"].split([n_expert//num_shards]*num_shards, dim=0)
        up_proj = state_dict[f"layers.{layer_i}.mlp.experts.up_proj.weight"].split([n_expert//num_shards]*num_shards, dim=0)

        state_dict.update({
            f"layers.{layer_i}.mlp.experts.gate_proj.weight": gate_proj,
            f"layers.{layer_i}.mlp.experts.down_proj.weight": down_proj,
            f"layers.{layer_i}.mlp.experts.up_proj.weight": up_proj,
        })

    # only split ColParallelLinear bias
    # for key in state_dict.keys():
    #     if 'wo.bias' in key or 'w2.bias' in key: continue
    #     if 'bias' in key:
    #         bias_dim = state_dict[key].shape[0]
    #         split_bias = state_dict[key].split([bias_dim // num_shards]*num_shards)
    #         state_dict.update({key: split_bias})

    # dump weight
    tmp_weight_list = [{} for _ in range(num_shards)]
    for key, value in state_dict.items():
        for idx, w_dict in enumerate(tmp_weight_list):
            if torch.is_tensor(value):
                tmp_weight_list[idx].update({key:value.clone()})
            else:
                tmp_weight_list[idx].update({key:value[idx].clone()})
    for idx, weight_dict in enumerate(tmp_weight_list):
        torch.save(weight_dict, os.path.join(model_path, f"model.{idx:02d}.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of opmx weights, which contains model folders",
    )
    parser.add_argument(
        "--num_shards",
        help="num of shards to split",
        type=int
    )
    parser.add_argument(
        "--parallel_mode",
        help="'etp' for expert-tensor parallel, 'edp' for expert-data parallel",
        type=str
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write OPMX model",
    )
    args = parser.parse_args()
    if args.parallel_mode == 'etp':
        split_pmx_model_etp(
            model_path=args.output_dir,
            input_base_path=args.input_dir,
            num_shards=args.num_shards
            )
    elif args.parallel_mode == 'edp':
        split_pmx_model_edp(
            model_path=args.output_dir,
            input_base_path=args.input_dir,
            num_shards=args.num_shards
            )
    else:
        assert False, f"unknown mode: {args.parallel_mode}"

if __name__ == "__main__":
    main()
