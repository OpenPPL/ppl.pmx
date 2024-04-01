import fire
import sys
import os
import json

from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import llama.modeling.Loader as Loader
from ModelParams import ModelParams

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    export_path: str,
    friendly_gqa: bool = False, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool = True, # fuse qkv linear
    fused_kvcache: bool = True, # fuse key_value_cache and multi_head_attention
    fused_ffn_glu: bool = True, # fuse feed forward gate linear unit
    auto_causal: bool = True, # causal mask is auto done by attention op, no need to pass additional mask to the model
    quantized_cache: bool = True, # 8bit kv cache quantization
    cache_layout: int = 0, # change kv cache layout for hardware performance friendly
    cache_mode: int = 0, # change kv cache indexing mode for memory management friendly, only affected when dynamic_batching == True
    dynamic_batching: bool = True, # use dynamic batching scheduling
):
    with open(Path(ckpt_dir) / "pmx_params.json", "r") as f:
        params = json.loads(f.read())
    params: ModelParams = ModelParams(**params)

    generator = Loader.load(
        ckpt_dir, params,
        friendly_gqa=friendly_gqa,
        fused_qkv=fused_qkv,
        fused_kvcache=fused_kvcache,
        fused_ffn_glu=fused_ffn_glu,
        fused_alibi=False,
        auto_causal=auto_causal,
        with_rope=True,
        with_alibi=False,
        quantized_cache=quantized_cache,
        cache_layout=cache_layout,
        cache_mode=cache_mode,
        dynamic_batching=dynamic_batching,
        attn_wqkv_bias_term=False,
        attn_wo_bias_term=False,
        ffn_linear_bias_term=False,
        load_to_cpu=True,
        rotary_dim=0,
    )

    generator.export(export_path)


if __name__ == "__main__":
    fire.Fire(main)
