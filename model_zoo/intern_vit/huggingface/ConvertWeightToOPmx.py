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
from safetensors import safe_open

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


def write_pmx_model(model_path, input_base_path, pad_to_head):
    os.makedirs(model_path, exist_ok=True)
    print ("Loading the checkpoint in a HF model")

    # convert pmx params
    pmx_params_dict = {}
    params = read_json((os.path.join(input_base_path, "config.json")))

    # vision_config
    pmx_params_dict['hidden_dim'] = params['hidden_size']
    pmx_params_dict['num_heads'] = params['num_attention_heads']
    pmx_params_dict['num_layers'] = params['num_hidden_layers']
    pmx_params_dict['norm_eps'] = params['layer_norm_eps']
    pmx_params_dict['image_size'] = params['image_size']
    pmx_params_dict['patch_size'] = params['patch_size']
    #pmx_params_dict['projection_dim'] = params['vision_config']['projection_dim']
    pmx_params_dict['num_kv_heads'] = params.get('num_key_value_heads', params['num_attention_heads'])

    # TO DO: GQA / MQA, only test on llama
    num_heads = pmx_params_dict['num_heads']
    num_kv_heads = pmx_params_dict['num_kv_heads']
    dims_per_head = pmx_params_dict['hidden_dim'] // num_heads
    key_value_dim = dims_per_head * num_kv_heads

    # process pad params
    pmx_params_dict['padded_num_heads'] = (num_heads + pad_to_head - 1) // pad_to_head * pad_to_head
    pmx_params_dict['qk_norm_scale'] = num_heads / pmx_params_dict['padded_num_heads']
    pmx_params_dict['padded_num_kv_heads'] = (num_kv_heads + pad_to_head - 1) // pad_to_head * pad_to_head
    pmx_params_dict['head_dim'] = dims_per_head

    # compute intermediate_size
    pmx_params_dict['intermediate_dim'] = params.get('intermediate_size')
    write_json(pmx_params_dict, os.path.join(model_path, "opmx_params.json"))

    hf_model_state_dict, state_dict = {}, {}
    for ckpt_path in sorted(Path(input_base_path).glob("*.safetensors")):
        weights = safe_open(ckpt_path, 'pt', 'cpu')
        weights = {k: weights.get_tensor(k) for k in weights.keys()}
        hf_model_state_dict.update(weights)

    for layer_i in range(pmx_params_dict['num_layers']):

        split_dim = [head * dims_per_head for head in [num_heads, num_kv_heads, num_kv_heads]]
        wq, wk, wv = hf_model_state_dict[f"encoder.layers.{layer_i}.attn.qkv.weight"].split(split_dim, dim=0)

        tmp_wq = torch.zeros(size=[ pmx_params_dict['padded_num_heads'] * dims_per_head, wq.shape[1]], dtype=wq.dtype)
        tmp_wk = torch.zeros(size=[ pmx_params_dict['padded_num_kv_heads'] * dims_per_head, wk.shape[1]], dtype=wk.dtype)
        tmp_wv = torch.zeros(size=[ pmx_params_dict['padded_num_kv_heads'] * dims_per_head, wv.shape[1]], dtype=wv.dtype)
        tmp_wq[:wq.shape[0], :] = wq
        tmp_wk[:wk.shape[0], :] = wk
        tmp_wv[:wv.shape[0], :] = wv

        tmp_wo = torch.zeros(size=[wq.shape[0], pmx_params_dict['padded_num_heads'] * dims_per_head], dtype=wq.dtype)
        #tmp_wo_bias = torch.zeros(size=[(num_heads+1) * dims_per_head], dtype=wq.dtype)
        tmp_wo[:, :wq.shape[1]] = hf_model_state_dict[f"encoder.layers.{layer_i}.attn.proj.weight"]


        q_norm_w = hf_model_state_dict[f"encoder.layers.{layer_i}.attn.q_norm.weight"]
        k_norm_w = hf_model_state_dict[f"encoder.layers.{layer_i}.attn.k_norm.weight"]
        tmp_q_norm_w = torch.zeros(size=[ pmx_params_dict['padded_num_heads'] * dims_per_head], dtype=wq.dtype)
        tmp_k_norm_w = torch.zeros(size=[ pmx_params_dict['padded_num_kv_heads'] * dims_per_head], dtype=wq.dtype)

        tmp_q_norm_w[:q_norm_w.shape[0]] =  q_norm_w
        tmp_k_norm_w[:k_norm_w.shape[0]] =  k_norm_w

        state_dict.update({
            f"layers.{layer_i}.attention.wq.weight": tmp_wq,
            f"layers.{layer_i}.attention.wk.weight": tmp_wk,
            f"layers.{layer_i}.attention.wv.weight": tmp_wv,

            f"layers.{layer_i}.attention.wo.weight": tmp_wo,
            f"layers.{layer_i}.attention.wo.bias": hf_model_state_dict[f"encoder.layers.{layer_i}.attn.proj.bias"],

            # ls1 ls2 qk_norm
            f"layers.{layer_i}.attention.q_norm.weight": tmp_q_norm_w,
            f"layers.{layer_i}.attention.k_norm.weight": tmp_k_norm_w,
            f"layers.{layer_i}.ls1": hf_model_state_dict[f"encoder.layers.{layer_i}.ls1"],
            f"layers.{layer_i}.ls2": hf_model_state_dict[f"encoder.layers.{layer_i}.ls2"],

            f"layers.{layer_i}.feed_forward.w1.weight": hf_model_state_dict[f"encoder.layers.{layer_i}.mlp.fc1.weight"],
            f"layers.{layer_i}.feed_forward.w1.bias": hf_model_state_dict[f"encoder.layers.{layer_i}.mlp.fc1.bias"],
            f"layers.{layer_i}.feed_forward.w2.weight": hf_model_state_dict[f"encoder.layers.{layer_i}.mlp.fc2.weight"],
            f"layers.{layer_i}.feed_forward.w2.bias": hf_model_state_dict[f"encoder.layers.{layer_i}.mlp.fc2.bias"],

            f"layers.{layer_i}.attention_norm.weight": hf_model_state_dict[f"encoder.layers.{layer_i}.norm1.weight"],
            #f"layers.{layer_i}.attention_norm.bias": hf_model_state_dict[f"vision_model.encoder.layers.{layer_i}.layer_norm1.bias"],

            f"layers.{layer_i}.ffn_norm.weight": hf_model_state_dict[f"encoder.layers.{layer_i}.norm2.weight"],
            #f"layers.{layer_i}.ffn_norm.bias": hf_model_state_dict[f"vision_model.encoder.layers.{layer_i}.layer_norm2.bias"],
        })

    state_dict.update({
        "vision_embeddings.cls_emb_weight": hf_model_state_dict["embeddings.class_embedding"],
        "vision_embeddings.patch_emb_weight": hf_model_state_dict["embeddings.patch_embedding.weight"],
        "vision_embeddings.patch_emb_bias": hf_model_state_dict["embeddings.patch_embedding.bias"],
        "vision_embeddings.pos_emb_weight": hf_model_state_dict["embeddings.position_embedding"],
    })

    if "visual_projection.weight" in hf_model_state_dict.keys():
        state_dict.update({"vision_projection.weight": hf_model_state_dict["visual_projection.weight"]})

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
    parser.add_argument(
        "--pad_to_head",
        help="num of heads to be padded",
        type=int,
    )
    args = parser.parse_args()
    write_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir
	pad_to_head=args.pad_to_head
    )

if __name__ == "__main__":
    main()
