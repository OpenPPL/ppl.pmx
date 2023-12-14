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

def merge_pmx_model(model_path, input_base_path, num_shards):
    os.makedirs(model_path, exist_ok=True)
    params = read_json(os.path.join(input_base_path, "pmx_params.json"))
    # weight sharding
    hidden_dim = params['hidden_dim']
    intermediate_dim = params['intermediate_dim']
    n_heads_per_shard = params['num_heads'] // num_shards

    # TO DO: GQA / MQA, only test when num_heads = num_kv_heads
    num_local_key_value_heads = n_heads_per_shard
    key_value_dim = hidden_dim
    num_key_value_heads = params['num_kv_heads'] if 'num_kv_heads' in params else params['num_heads']
    dims_per_head = hidden_dim // params['num_heads']
    write_json(params, os.path.join(model_path, "pmx_params.json"))

    loaded = [
        torch.load(os.path.join(input_base_path, f"model.{i}.pth"), map_location="cpu")
        for i in range(num_shards)
    ]

    state_dict = {}
    for layer_i in range(params['num_layers']):
        merge_wq = torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(
                            n_heads_per_shard, dims_per_head, hidden_dim
                        )
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(hidden_dim, hidden_dim)

        merge_wk = torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(
                            num_local_key_value_heads, dims_per_head, hidden_dim
                        )
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(key_value_dim, hidden_dim)

        merge_wv = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(
                        num_local_key_value_heads, dims_per_head, hidden_dim
                    )
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(key_value_dim, hidden_dim)

        state_dict.update({
            f"layers.{layer_i}.attention.wq.weight": merge_wq,
            f"layers.{layer_i}.attention.wk.weight": merge_wk,
            f"layers.{layer_i}.attention.wv.weight": merge_wv,
            f"layers.{layer_i}.attention.wq.bias": torch.cat([loaded[i][f"layers.{layer_i}.attention.wq.bias"] for i in range(num_shards)]),
            f"layers.{layer_i}.attention.wk.bias": torch.cat([loaded[i][f"layers.{layer_i}.attention.wk.bias"] for i in range(num_shards)]),
            f"layers.{layer_i}.attention.wv.bias": torch.cat([loaded[i][f"layers.{layer_i}.attention.wv.bias"] for i in range(num_shards)]),
            f"layers.{layer_i}.attention_norm.weight": loaded[0][f"layers.{layer_i}.attention_norm.weight"].clone(),
            f"layers.{layer_i}.attention_norm.bias": loaded[0][f"layers.{layer_i}.attention_norm.bias"].clone(),
            f"layers.{layer_i}.ffn_norm.weight": loaded[0][f"layers.{layer_i}.ffn_norm.weight"].clone(),
            f"layers.{layer_i}.ffn_norm.bias": loaded[0][f"layers.{layer_i}.ffn_norm.bias"].clone(),
            f"layers.{layer_i}.attention.wo.weight": torch.cat([loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1),
            f"layers.{layer_i}.attention.wo.bias": loaded[0][f"layers.{layer_i}.attention.wo.bias"].clone(),
            f"layers.{layer_i}.feed_forward.w1.weight": torch.cat([loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0),
            f"layers.{layer_i}.feed_forward.w1.bias": torch.cat([loaded[i][f"layers.{layer_i}.feed_forward.w1.bias"] for i in range(num_shards)]),
            f"layers.{layer_i}.feed_forward.w2.weight": torch.cat([loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1),
            f"layers.{layer_i}.feed_forward.w2.bias": loaded[0][f"layers.{layer_i}.feed_forward.w2.bias"].clone(),
        })

    state_dict.update({
        "norm.weight": loaded[0]["norm.weight"],
        "tok_embeddings.weight": torch.cat([loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1),
        "output.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
    })
    torch.save(state_dict, os.path.join(model_path, "model.pth"))


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
    merge_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        num_shards=args.num_shards
        )

if __name__ == "__main__":
    main()