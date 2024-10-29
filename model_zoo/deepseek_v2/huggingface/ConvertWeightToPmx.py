# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gc
import json
import os
import shutil
import warnings

import torch

from pathlib import Path

"""
Sample usage:

```
python convert_hf_weights_to_pmx.py \
    --input_dir /path/to/downloaded/hf/weights/7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```
Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_pmx_model(model_path, input_base_path, model_type):
    os.makedirs(model_path, exist_ok=True)
    print ("Loading the checkpoint in a HF model")

    # convert opmx params
    pmx_params_dict = {}
    params = read_json((os.path.join(input_base_path, "config.json")))
    pmx_params_dict['hidden_dim'] = params['hidden_size']
    pmx_params_dict['intermediate_dim'] = params['intermediate_size']

    pmx_params_dict['num_layers'] = params['num_hidden_layers']
    pmx_params_dict['num_heads'] = params['num_attention_heads']
    pmx_params_dict['num_kv_heads'] = params['num_key_value_heads']
    pmx_params_dict['q_lora_rank'] = params['q_lora_rank'] if params['q_lora_rank'] is not None else 0
    pmx_params_dict['kv_lora_rank'] = params['kv_lora_rank']
    pmx_params_dict['qk_nope_head_dim'] = params['qk_nope_head_dim']
    pmx_params_dict['qk_rope_head_dim'] = params['qk_rope_head_dim']
    pmx_params_dict['v_head_dim'] = params['v_head_dim']

    pmx_params_dict['vocab_size'] = params['vocab_size']
    pmx_params_dict['norm_eps'] = params['rms_norm_eps']

    pmx_params_dict['rope_theta'] = params['rope_theta']
    pmx_params_dict['rope_scaling_type'] = params['rope_scaling']['type']
    pmx_params_dict['rope_scaling_factor'] = params['rope_scaling']['factor']
    pmx_params_dict['rope_scaling_beta_fast'] = params['rope_scaling']['beta_fast']
    pmx_params_dict['rope_scaling_beta_slow'] = params['rope_scaling']['beta_slow']
    pmx_params_dict['rope_scaling_mscale'] = params['rope_scaling']['mscale']
    pmx_params_dict['rope_scaling_mscale_all_dim'] = params['rope_scaling']['mscale_all_dim']
    pmx_params_dict['origin_max_position_embeddings'] = params['rope_scaling']['original_max_position_embeddings']
    pmx_params_dict['max_position_embeddings'] = params['max_position_embeddings']


    pmx_params_dict['num_first_dense_layers'] = params['first_k_dense_replace']
    pmx_params_dict['num_shared_experts'] = params['n_shared_experts']
    pmx_params_dict['num_experts'] = params['n_routed_experts']
    pmx_params_dict['num_experts_per_token'] = params['num_experts_per_tok']
    pmx_params_dict['num_expert_groups'] = params['n_group']
    pmx_params_dict['moe_scaling_factor'] = params['routed_scaling_factor']
    pmx_params_dict['moe_normalize_prob'] = params['norm_topk_prob']
    pmx_params_dict['moe_intermediate_dim'] = params['moe_intermediate_size']
    pmx_params_dict['moe_topk_group'] = params['topk_group']
    pmx_params_dict['moe_topk_method'] = params['topk_method']


    print(pmx_params_dict)
    write_json(pmx_params_dict, os.path.join(model_path, "opmx_params.json"))

    # # TO DO: GQA / MQA, only test on llama
    # num_heads = pmx_params_dict['num_heads']
    # num_kv_heads = pmx_params_dict['num_kv_heads']
    # dims_per_head = hidden_dim // num_heads
    # key_value_dim = dims_per_head * num_kv_heads

    # # load weights
    # def unpermute(w, n_heads=num_heads, dim1=hidden_dim, dim2=hidden_dim):
    #     return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    hf_model_state_dict, state_dict = {}, {}

    if model_type is None:
        if any(Path(input_base_path).glob("*.safetensors")):
            model_type = "safetensors"
        else:
            model_type = "bin"

    if model_type == "bin":
        for ckpt_path in sorted(Path(input_base_path).glob("*.bin")):
            hf_model_state_dict.update(torch.load(ckpt_path, map_location="cpu"))
    elif model_type == "safetensors":
        from safetensors import safe_open
        for ckpt_path in sorted(Path(input_base_path).glob("*.safetensors")):
            weights = safe_open(ckpt_path, 'pt', 'cpu')
            weights = {k: weights.get_tensor(k) for k in weights.keys()}
            hf_model_state_dict.update(weights)
    else:
        raise ValueError(f"Not support the model_type: {model_type}.")

    for layer_i in range(pmx_params_dict['num_first_dense_layers']):
        # attn weight
        if not pmx_params_dict['q_lora_rank']:
            wq = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"]
        else:
            pass # todo
        kv_a_proj = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.kv_a_proj_with_mqa.weight"]
        kv_a_layernorm_w = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.kv_a_layernorm.weight"]
        kv_b_proj = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.kv_b_proj.weight"]
        o_proj = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"]

        input_layernorm_w = hf_model_state_dict[f"model.layers.{layer_i}.input_layernorm.weight"]
        post_attention_layernorm_w = hf_model_state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"]

        gate_proj = hf_model_state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"]
        up_proj = hf_model_state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"]
        down_proj = hf_model_state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"]

        state_dict.update({
            f"layers.{layer_i}.self_attn.q_proj.weight": wq, 
            f"layers.{layer_i}.self_attn.kv_a_proj.weight": kv_a_proj,
            f"layers.{layer_i}.self_attn.kv_a_layernorm.weight": kv_a_layernorm_w,
            f"layers.{layer_i}.self_attn.kv_b_proj.weight": kv_b_proj,
            f"layers.{layer_i}.self_attn.o_proj.weight": o_proj,

            f"layers.{layer_i}.input_layernorm.weight": input_layernorm_w,
            f"layers.{layer_i}.post_attention_layernorm.weight": post_attention_layernorm_w,

            f"layers.{layer_i}.mlp.gate_proj.weight": gate_proj,
            f"layers.{layer_i}.mlp.up_proj.weight": up_proj,
            f"layers.{layer_i}.mlp.down_proj.weight": down_proj,
        })


    for layer_i in range(pmx_params_dict['num_first_dense_layers'], pmx_params_dict['num_layers']):

        # attn weight
        if not pmx_params_dict['q_lora_rank']:
            wq = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"]
        else:
            pass # todo
        kv_a_proj = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.kv_a_proj_with_mqa.weight"]
        kv_a_layernorm_w = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.kv_a_layernorm.weight"]
        kv_b_proj = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.kv_b_proj.weight"]
        o_proj = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"]



        # mlp shared expert
        mlp_gate = hf_model_state_dict[f"model.layers.{layer_i}.mlp.gate.weight"]
        shared_expert_gate_proj = hf_model_state_dict[f"model.layers.{layer_i}.mlp.shared_experts.gate_proj.weight"]
        shared_expert_up_proj = hf_model_state_dict[f"model.layers.{layer_i}.mlp.shared_experts.up_proj.weight"]
        shared_expert_down_proj =  hf_model_state_dict[f"model.layers.{layer_i}.mlp.shared_experts.down_proj.weight"]

        #mlp expert
        expert_gate_proj, expert_up_proj,  expert_down_proj = [], [], []
        for expert_i in range(pmx_params_dict['num_experts']):
            expert_gate_proj.append(hf_model_state_dict[f"model.layers.{layer_i}.mlp.experts.{expert_i}.gate_proj.weight"].unsqueeze(0))
            expert_up_proj.append(hf_model_state_dict[f"model.layers.{layer_i}.mlp.experts.{expert_i}.up_proj.weight"].unsqueeze(0))
            expert_down_proj.append(hf_model_state_dict[f"model.layers.{layer_i}.mlp.experts.{expert_i}.down_proj.weight"].unsqueeze(0))

        input_layernorm_w = hf_model_state_dict[f"model.layers.{layer_i}.input_layernorm.weight"]
        post_attention_layernorm_w = hf_model_state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"]
        
        state_dict.update({
            f"layers.{layer_i}.self_attn.q_proj.weight": wq, 
            f"layers.{layer_i}.self_attn.kv_a_proj.weight": kv_a_proj,
            f"layers.{layer_i}.self_attn.kv_a_layernorm.weight": kv_a_layernorm_w,
            f"layers.{layer_i}.self_attn.kv_b_proj.weight": kv_b_proj,
            f"layers.{layer_i}.self_attn.o_proj.weight": o_proj,

            f"layers.{layer_i}.mlp.gate.weight": mlp_gate,
            f"layers.{layer_i}.mlp.shared_experts.gate_proj.weight": shared_expert_gate_proj,
            f"layers.{layer_i}.mlp.shared_experts.up_proj.weight": shared_expert_up_proj,
            f"layers.{layer_i}.mlp.shared_experts.down_proj.weight": shared_expert_down_proj,

            f"layers.{layer_i}.input_layernorm.weight": input_layernorm_w,
            f"layers.{layer_i}.post_attention_layernorm.weight": post_attention_layernorm_w,

            f"layers.{layer_i}.mlp.experts.gate_proj.weight": torch.cat(expert_gate_proj, dim=0),
            f"layers.{layer_i}.mlp.experts.up_proj.weight": torch.cat(expert_up_proj, dim=0),
            f"layers.{layer_i}.mlp.experts.down_proj.weight": torch.cat(expert_down_proj, dim=0),

        })

    state_dict.update({
        "tok_embeddings.weight": hf_model_state_dict["model.embed_tokens.weight"],
        "norm.weight": hf_model_state_dict["model.norm.weight"],
        "lm_head.weight": hf_model_state_dict["lm_head.weight"]
    })
    torch.save(state_dict, os.path.join(model_path, "model.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of HF weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write OPMX model",
    )
    parser.add_argument(
        "--model_type",
        choices=["bin", "safetensors"],
        default=None,
        help="Input model type",
    )
    args = parser.parse_args()
    write_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_type=args.model_type
    )

if __name__ == "__main__":
    main()
