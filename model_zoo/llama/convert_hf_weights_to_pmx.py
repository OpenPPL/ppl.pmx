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


def write_pmx_model(model_path, input_base_path):
    os.makedirs(model_path, exist_ok=True)
    print ("Loading the checkpoint in a HF model")

    # convert pmx params
    pmx_params_dict = {}
    params = read_json((os.path.join(input_base_path, "config.json")))
    pmx_params_dict['hidden_dim'] = params['hidden_size']
    pmx_params_dict['num_heads'] = params['num_attention_heads']
    pmx_params_dict['num_layers'] = params['num_hidden_layers']
    pmx_params_dict['norm_eps'] = params['rms_norm_eps']
    pmx_params_dict['vocab_size'] = params['vocab_size']
    pmx_params_dict['num_kv_heads'] = params.get('num_key_value_heads', params['num_attention_heads'])

    # compute intermediate_size
    hidden_dim = pmx_params_dict['hidden_dim']
    multiple_of = params.get("multiple_of", 256)
    ffn_dim_multiplier = params.get("ffn_dim_multiplier", 1)
    pmx_params_dict['intermediate_dim'] = compute_intermediate_size(hidden_dim, ffn_dim_multiplier, multiple_of)
    write_json(pmx_params_dict, os.path.join(model_path, "pmx_params.json"))

    # TO DO: GQA / MQA, only test on llama
    num_heads = pmx_params_dict['num_heads']
    num_kv_heads = pmx_params_dict['num_kv_heads']
    key_value_dim = pmx_params_dict['hidden_dim']
    dims_per_head = hidden_dim // num_heads

    # load weights
    def unpermute(w, n_heads=num_heads, dim1=hidden_dim, dim2=hidden_dim):
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    hf_model_state_dict, state_dict = {}, {}
    for ckpt_path in sorted(Path(input_base_path).glob("*.bin")):
        hf_model_state_dict.update(torch.load(ckpt_path, map_location="cpu"))

    for layer_i in range(pmx_params_dict['num_layers']):

        wq = unpermute(hf_model_state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"])
        wk = unpermute(hf_model_state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"], num_kv_heads, key_value_dim, hidden_dim)
        wv = hf_model_state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"]

        state_dict.update({
            f"layers.{layer_i}.attention.wq.weight": wq,
            f"layers.{layer_i}.attention.wk.weight": wk,
            f"layers.{layer_i}.attention.wv.weight": wv,
            f"layers.{layer_i}.attention.wo.weight": hf_model_state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"],
            f"layers.{layer_i}.feed_forward.w1.weight": hf_model_state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"],
            f"layers.{layer_i}.feed_forward.w2.weight": hf_model_state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"],
            f"layers.{layer_i}.feed_forward.w3.weight": hf_model_state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"],
            f"layers.{layer_i}.attention_norm.weight": hf_model_state_dict[f"model.layers.{layer_i}.input_layernorm.weight"],
            f"layers.{layer_i}.ffn_norm.weight": hf_model_state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"],
        })

    state_dict.update({
        "tok_embeddings.weight": hf_model_state_dict["model.embed_tokens.weight"],
        "norm.weight": hf_model_state_dict["model.norm.weight"],
        "output.weight": hf_model_state_dict["lm_head.weight"]
    })
    # check 7B
    '''
    cor_dict = torch.load('/mnt/hpc/share/llama_model/LLaMA/7B/consolidated.00.pth', map_location="cpu")
    for k,v in cor_dict.items():
        print (k)
        if 'rope.freqs' in k: continue
        diff = state_dict[k] - cor_dict[k]
        if diff.max() != 0 or diff.min() != 0:
            import ipdb;ipdb.set_trace()
    '''
    torch.save(state_dict, os.path.join(model_path, "model.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of HF weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write PMX model",
    )
    args = parser.parse_args()
    write_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir
    )

if __name__ == "__main__":
    main()
