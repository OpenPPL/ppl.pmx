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
    pmx_params_dict['num_layers'] = params['num_layers']
    pmx_params_dict['norm_eps'] = params['layernorm_epsilon']
    pmx_params_dict['vocab_size'] = params['vocab_size']
    pmx_params_dict['num_kv_heads'] = params.get('num_key_value_heads', params['num_attention_heads'])
    pmx_params_dict['intermediate_dim'] = params.get('inner_hidden_size', 4 * params['hidden_size'])
    
    write_json(pmx_params_dict, os.path.join(model_path, "pmx_params.json"))
    print(pmx_params_dict)
    
    # TO DO: GQA / MQA, only test on llama
    hidden_dim = pmx_params_dict['hidden_dim']
    num_heads = pmx_params_dict['num_heads']
    num_kv_heads = pmx_params_dict['num_kv_heads']  # same to num_head
    key_value_dim = pmx_params_dict['hidden_dim']   # same to key_value_dim
    dims_per_head = hidden_dim // num_heads


    # load weights
    def unpermute_weight(wqkv, n_heads, num_kv_heads, hidden_dim, key_value_dim):   
        assert n_heads == num_kv_heads, "as for chatglm1, n_heads == num_kv_heads"
        assert hidden_dim == key_value_dim, "as for chatglm1, hidden_dim == key_value_dim"
             
        wqkv = wqkv.reshape(n_heads, 3, hidden_dim // n_heads, hidden_dim).transpose(0, 1).reshape(3 * hidden_dim, hidden_dim)
        wq, wk, wv = wqkv.chunk(3, dim=0)   # [(num_heads,3,head_dim), hidden_dim] --> [hidden_dim, hidden_dim]

        # 先对半切，再分别转置
        wq0, wq1 = wq.chunk(2, dim=0)
        wq0 = wq0.reshape(n_heads, 2, hidden_dim // n_heads // 4, hidden_dim).transpose(1,2).reshape(hidden_dim // 2, hidden_dim)
        wq1 = wq1.reshape(n_heads, 2, hidden_dim // n_heads // 4, hidden_dim).transpose(1,2).reshape(hidden_dim // 2, hidden_dim)
        wq = torch.cat((wq0, wq1), dim=0)

        wk0, wk1 = wk.chunk(2, dim=0)
        wk0 = wk0.reshape(num_kv_heads, 2, key_value_dim // num_kv_heads // 4, hidden_dim).transpose(1,2).reshape(key_value_dim // 2, hidden_dim)
        wk1 = wk1.reshape(num_kv_heads, 2, key_value_dim // num_kv_heads // 4, hidden_dim).transpose(1,2).reshape(key_value_dim // 2, hidden_dim)
        wk = torch.cat((wk0, wk1), dim=0)

        return wq, wk, wv

    def unpermute_bias(wqkv_bias, n_heads, num_kv_heads, hidden_dim, key_value_dim):
        wqkv_bias = wqkv_bias.reshape(n_heads, 3, hidden_dim // n_heads).transpose(0, 1).reshape(3 * hidden_dim)
        wq_bias, wk_bias, wv_bias = wqkv_bias.chunk(3)

        wq_bias_0, wq_bias_1 = wq_bias.chunk(2)
        wq_bias_0 = wq_bias_0.view(n_heads, 2, hidden_dim // n_heads // 4).transpose(1, 2).reshape(hidden_dim // 2)
        wq_bias_1 = wq_bias_1.view(n_heads, 2, hidden_dim // n_heads // 4).transpose(1, 2).reshape(hidden_dim // 2)
        wq_bias = torch.cat((wq_bias_0, wq_bias_1))

        wk_bias_0, wk_bias_1 = wk_bias.chunk(2)
        wk_bias_0 = wk_bias_0.view(num_kv_heads, 2, key_value_dim // num_kv_heads // 4).transpose(1, 2).reshape(key_value_dim // 2)
        wk_bias_1 = wk_bias_1.view(num_kv_heads, 2, key_value_dim // num_kv_heads // 4).transpose(1, 2).reshape(key_value_dim // 2)
        wk_bias = torch.cat((wk_bias_0, wk_bias_1))
        
        return wq_bias, wk_bias, wv_bias

    hf_model_state_dict, state_dict = {}, {}
    for ckpt_path in sorted(Path(input_base_path).glob("*.bin")):
        hf_model_state_dict.update(torch.load(ckpt_path, map_location="cpu"))

    for layer_i in range(pmx_params_dict['num_layers']):
        wq, wk, wv = unpermute_weight(hf_model_state_dict[f"transformer.layers.{layer_i}.attention.query_key_value.weight"], num_heads, num_kv_heads, hidden_dim, key_value_dim)
        wq_bias, wk_bias, wv_bias = unpermute_bias(hf_model_state_dict[f"transformer.layers.{layer_i}.attention.query_key_value.bias"], num_heads, num_kv_heads, hidden_dim, key_value_dim)
    
        state_dict.update({
            f"layers.{layer_i}.attention.wq.weight": wq,
            f"layers.{layer_i}.attention.wk.weight": wk,
            f"layers.{layer_i}.attention.wv.weight": wv,
            f"layers.{layer_i}.attention.wq.bias": wq_bias,
            f"layers.{layer_i}.attention.wk.bias": wk_bias,
            f"layers.{layer_i}.attention.wv.bias": wv_bias,
            f"layers.{layer_i}.attention.wo.weight": hf_model_state_dict[f"transformer.layers.{layer_i}.attention.dense.weight"],
            f"layers.{layer_i}.attention.wo.bias": hf_model_state_dict[f"transformer.layers.{layer_i}.attention.dense.bias"],

            f"layers.{layer_i}.feed_forward.w1.weight": hf_model_state_dict[f"transformer.layers.{layer_i}.mlp.dense_h_to_4h.weight"],
            f"layers.{layer_i}.feed_forward.w1.bias": hf_model_state_dict[f"transformer.layers.{layer_i}.mlp.dense_h_to_4h.bias"],
            f"layers.{layer_i}.feed_forward.w2.weight": hf_model_state_dict[f"transformer.layers.{layer_i}.mlp.dense_4h_to_h.weight"],
            f"layers.{layer_i}.feed_forward.w2.bias": hf_model_state_dict[f"transformer.layers.{layer_i}.mlp.dense_4h_to_h.bias"],
            
            f"layers.{layer_i}.attention_norm.weight": hf_model_state_dict[f"transformer.layers.{layer_i}.input_layernorm.weight"],
            f"layers.{layer_i}.attention_norm.bias": hf_model_state_dict[f"transformer.layers.{layer_i}.input_layernorm.bias"],
            
            f"layers.{layer_i}.ffn_norm.weight": hf_model_state_dict[f"transformer.layers.{layer_i}.post_attention_layernorm.weight"],
            f"layers.{layer_i}.ffn_norm.bias": hf_model_state_dict[f"transformer.layers.{layer_i}.post_attention_layernorm.bias"],
        })

    state_dict.update({
        "tok_embeddings.weight": hf_model_state_dict["transformer.word_embeddings.weight"],
        "norm.weight": hf_model_state_dict["transformer.final_layernorm.weight"],
        "norm.bias": hf_model_state_dict["transformer.final_layernorm.bias"],
        "output.weight": hf_model_state_dict["lm_head.weight"]
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
        help="Location to write PMX model",
    )
    args = parser.parse_args()
    write_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir
    )

if __name__ == "__main__":
    main()