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


def split_pmx_model(model_path, input_base_path, num_shards):
    os.makedirs(model_path, exist_ok=True)
    params = read_json((os.path.join(input_base_path, "pmx_params.json")))
    # weight sharding
    hidden_dim = params['hidden_dim']
    intermediate_dim = params['intermediate_dim']
    n_heads_per_shard = params['num_heads'] // num_shards

    # TO DO: GQA / MQA, only test on llama
    num_local_key_value_heads = n_heads_per_shard
    key_value_dim = hidden_dim
    num_kv_heads = params['num_kv_heads'] if 'num_kv_heads' in params else params['num_heads']
    dims_per_head = hidden_dim // params['num_heads']
    write_json(params, os.path.join(model_path, "pmx_params.json"))

    state_dict = {}
    for ckpt_path in sorted(Path(input_base_path).glob("*.pth")):
        state_dict.update(torch.load(ckpt_path, map_location='cpu'))

    for layer_i in range(params['num_layers']):
        wq = state_dict[f"layers.{layer_i}.attention.wq.weight"].reshape(n_heads_per_shard*num_shards, dims_per_head, hidden_dim).split([n_heads_per_shard]*num_shards, dim=0)  # [40, 128, 5120]
        wq = [w.reshape(-1, hidden_dim) for w in wq]    # each for [2560, 1280]
        wq_bias = state_dict[f"layers.{layer_i}.attention.wq.bias"].split([hidden_dim // num_shards]*num_shards)
        
        wk = state_dict[f"layers.{layer_i}.attention.wk.weight"].reshape(num_local_key_value_heads*num_shards, dims_per_head, hidden_dim).split([num_local_key_value_heads]*num_shards, dim=0)
        wk = [w.reshape(-1, hidden_dim) for w in wk]
        wk_bias = state_dict[f"layers.{layer_i}.attention.wk.bias"].split([hidden_dim // num_shards]*num_shards)

        wv = state_dict[f"layers.{layer_i}.attention.wv.weight"].reshape(num_local_key_value_heads*num_shards, dims_per_head, hidden_dim).split([num_local_key_value_heads]*num_shards, dim=0)
        wv = [w.reshape(-1, hidden_dim) for w in wv]
        wv_bias = state_dict[f"layers.{layer_i}.attention.wv.bias"].split([hidden_dim // num_shards]*num_shards)

        wo = state_dict[f"layers.{layer_i}.attention.wo.weight"].split([hidden_dim // num_shards]*num_shards, dim=1)
        wo_bias = state_dict[f"layers.{layer_i}.attention.wo.bias"]

        ff_w1 = state_dict[f"layers.{layer_i}.feed_forward.w1.weight"].split([intermediate_dim // num_shards]*num_shards, dim=0)
        ff_w1_bias = state_dict[f"layers.{layer_i}.feed_forward.w1.bias"].split([intermediate_dim // num_shards]*num_shards)
        
        ff_w2 = state_dict[f"layers.{layer_i}.feed_forward.w2.weight"].split([intermediate_dim // num_shards]*num_shards, dim=1)
        ff_w2_bias = state_dict[f"layers.{layer_i}.feed_forward.w2.bias"]

        state_dict.update({
            f"layers.{layer_i}.attention.wq.weight": wq,
            f"layers.{layer_i}.attention.wk.weight": wk,
            f"layers.{layer_i}.attention.wv.weight": wv,
            f"layers.{layer_i}.attention.wo.weight": wo,
            f"layers.{layer_i}.attention.wq.bias": wq_bias,
            f"layers.{layer_i}.attention.wk.bias": wk_bias,
            f"layers.{layer_i}.attention.wv.bias": wv_bias,
            f"layers.{layer_i}.attention.wo.bias": wo_bias,
            f"layers.{layer_i}.feed_forward.w1.weight": ff_w1,
            f"layers.{layer_i}.feed_forward.w2.weight": ff_w2,
            f"layers.{layer_i}.feed_forward.w1.bias": ff_w1_bias,
            f"layers.{layer_i}.feed_forward.w2.bias": ff_w2_bias,
        })

    token_emb_weight = state_dict["tok_embeddings.weight"].split([hidden_dim // num_shards]*num_shards, dim=1)
    output_weight = state_dict["output.weight"].split([params['vocab_size'] // num_shards]*num_shards, dim=0)
    state_dict.update({
        "tok_embeddings.weight": token_emb_weight,
        "output.weight": output_weight
    })

    # dump weight
    tmp_weight_list = [{} for _ in range(num_shards)]
    for key, value in state_dict.items():
        for idx, w_dict in enumerate(tmp_weight_list):
            if torch.is_tensor(value):
                tmp_weight_list[idx].update({key:value.clone()})
            else:
                tmp_weight_list[idx].update({key:value[idx].clone()})
    for idx, weight_dict in enumerate(tmp_weight_list):
        torch.save(weight_dict, os.path.join(model_path, f"model.{idx}.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of pmx weights, which contains model folders",
    )
    parser.add_argument(
        "--num_shards",
        help="num of shards to split",
        type=int
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write PMX model",
    )
    args = parser.parse_args()
    split_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        num_shards=args.num_shards
        )

if __name__ == "__main__":
    main()
