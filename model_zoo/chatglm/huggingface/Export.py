import fire
import sys
import os
import json

from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import chatglm.modeling.Loader as Loader
from ModelParams import ModelParams


def main(
    ckpt_dir: str,
    export_path: str,
    friendly_gqa: bool = False, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool = True, # fuse qkv linear
    fused_kvcache: bool = True, # fuse key_value_cache and multi_head_attention
    quantized_cache: bool = True, # 8bit kv cache quantization
    cache_layout: int = 0, # change kv cache layout for hardware performance friendly
    cache_mode: int = 0, # change kv cache indexing mode for memory management friendly, only affected when dynamic_batching == True
    dynamic_batching: bool = True, # use dynamic batching scheduling
):
    
    with open(Path(ckpt_dir) / "pmx_params.json", "r") as f:
        params = json.loads(f.read())
    params: ModelParams = ModelParams(**params)

    generator = Loader.load(
        ckpt_dir, params, friendly_gqa,
        fused_qkv, fused_kvcache,
        quantized_cache, cache_layout,
        cache_mode, dynamic_batching,
        attn_linear_bias_term=True, ffn_linear_bias_term=True, load_to_cpu=False,
    )
    
    generator.export(export_path)

if __name__ == "__main__":
    fire.Fire(main)
