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
import json
import os

import torch

from pathlib import Path
from safetensors.torch import load_file


"""
Sample usage:

```
python convert_hf_weights_to_pmx.py \
    --input_dir /input/path --output_dir /output/path
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


def write_pmx_model(model_path, input_base_path):
    os.makedirs(model_path, exist_ok=True)
    print ("Loading the checkpoint in a HF model")

    # convert opmx params
    pmx_params_dict = {}
    params = read_json((os.path.join(input_base_path, "config.json")))
    pmx_params_dict['hidden_dim'] = params['hidden_size']
    pmx_params_dict['num_heads'] = params['num_attention_heads']
    pmx_params_dict['num_layers'] = params['num_hidden_layers']
    pmx_params_dict['norm_eps'] = params['rms_norm_eps']
    pmx_params_dict['vocab_size'] = params['vocab_size']
    pmx_params_dict['num_kv_heads'] = params.get('num_key_value_heads', params['num_attention_heads'])
    pmx_params_dict['rope_theta'] = params['rope_theta']
    pmx_params_dict["max_position_embeddings"] = params["max_position_embeddings"]


    # compute intermediate_size
    hidden_dim = pmx_params_dict['hidden_dim']
    multiple_of = params.get("multiple_of", 256)
    ffn_dim_multiplier = params.get("ffn_dim_multiplier", 1)
    if "intermediate_size" in params.keys():
        pmx_params_dict['intermediate_dim'] = params.get("intermediate_size")
    else:
        pmx_params_dict['intermediate_dim'] = compute_intermediate_size(hidden_dim, ffn_dim_multiplier, multiple_of)
    write_json(pmx_params_dict, os.path.join(model_path, "opmx_params.json"))

    # TO DO: GQA / MQA, only test on llama
    num_heads = pmx_params_dict['num_heads']
    num_kv_heads = pmx_params_dict['num_kv_heads']
    dims_per_head = hidden_dim // num_heads
    # key_value_dim = pmx_params_dict['hidden_dim']
    key_value_dim = dims_per_head * num_kv_heads

    # load weights
    def unpermute(w, n_heads=num_heads, dim_in=hidden_dim, dim_out=hidden_dim):
        return w.view(n_heads, 2, dim_out // n_heads // 2, dim_in).transpose(1, 2).reshape(dim_out, dim_in)


    def unpermute_bias(bias, n_heads=num_heads, dim=hidden_dim):
        return bias.view(n_heads, 2, dim // n_heads // 2).transpose(1, 2).reshape(dim)

    hf_model_state_dict, state_dict = {}, {}
    for ckpt_path in sorted(Path(input_base_path).glob("*.safetensors")):
        hf_model_state_dict.update(load_file(ckpt_path, device="cpu"))

    for layer_i in range(pmx_params_dict['num_layers']):
        wq = unpermute(hf_model_state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"], num_heads, hidden_dim, hidden_dim)
        wk = unpermute(hf_model_state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"], num_kv_heads, hidden_dim, key_value_dim)

        wq_bias = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.q_proj.bias"]
        wk_bias = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.k_proj.bias"]

        wq_bias = unpermute_bias(wq_bias, num_heads, hidden_dim)
        wk_bias = unpermute_bias(wk_bias, num_kv_heads, key_value_dim)

        wv = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"]

        state_dict.update({
            f"layers.{layer_i}.attention.wq.weight": wq,
            f"layers.{layer_i}.attention.wk.weight": wk,
            f"layers.{layer_i}.attention.wv.weight": wv,

            f"layers.{layer_i}.attention.wq.bias": wq_bias,
            f"layers.{layer_i}.attention.wk.bias": wk_bias,
            f"layers.{layer_i}.attention.wv.bias": hf_model_state_dict[f"model.layers.{layer_i}.self_attn.v_proj.bias"],

            f"layers.{layer_i}.attention.wo.weight": hf_model_state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"],
            # f"layers.{layer_i}.attention.wo.bias": hf_model_state_dict[f"model.layers.{layer_i}.self_attn.o_proj.bias"],
            f"layers.{layer_i}.feed_forward.w1.weight": hf_model_state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"],
            f"layers.{layer_i}.feed_forward.w2.weight": hf_model_state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"],
            f"layers.{layer_i}.feed_forward.w3.weight": hf_model_state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"],
            f"layers.{layer_i}.attention_norm.weight": hf_model_state_dict[f"model.layers.{layer_i}.input_layernorm.weight"],
            f"layers.{layer_i}.ffn_norm.weight": hf_model_state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"],
        })

    state_dict.update({
        "tok_embeddings.weight": hf_model_state_dict["model.embed_tokens.weight"],
        "norm.weight": hf_model_state_dict["model.norm.weight"],
        "output.weight": hf_model_state_dict["model.embed_tokens.weight"],
    })


    torch.save(state_dict, os.path.join(model_path, "model.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
         "--input_dir",
         help="Location of HF weights, which contains tokenizer.model and model folders",)
    parser.add_argument(
         "--output_dir",
         help="Location to write OPMX model",)
    args = parser.parse_args()
    write_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
    )

if __name__ == "__main__":
    main()
