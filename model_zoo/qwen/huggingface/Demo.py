import fire
import sys
import os
import json

from pathlib import Path
from typing import List
import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import llama.modeling.Loader as Loader
from Tokenizer import Tokenizer, make_context, decode_context
from ModelParams import ModelParams


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    top_p: float = 0.95,
    batch: int = 1,
    seqlen_scale_up: int = 1,
    max_gen_len: int = 512,
    friendly_gqa: bool = False, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool = True, # fuse qkv linear
    fused_kvcache: bool = True, # fuse key_value_cache and multi_head_attention
    fused_ffn_glu: bool = True, # fuse feed forward gate linear unit
    auto_causal: bool = True, # causal mask is auto done by attention op, no need to pass additional mask to the model
    quantized_cache: bool = True, # 8bit kv cache quantization
    cache_layout: int = 0, # change kv cache layout for hardware performance friendly
    cache_mode: int = 0, # change kv cache indexing mode for memory management friendly, only affected when dynamic_batching == True
    dynamic_batching: bool = True, # use dynamic batching scheduling
    dump_tensor_path: str = '',
    dump_steps: List[int] = []
):
    tokenizer = Tokenizer(model_path=tokenizer_path)

    with open(Path(ckpt_dir) / "pmx_params.json", "r") as f:
        params = json.loads(f.read())
    params: ModelParams = ModelParams(**params)

    # attn_wqkv_bias_term
    generator = Loader.load(
        ckpt_dir, params, friendly_gqa,
        fused_qkv, fused_kvcache, fused_ffn_glu,
        auto_causal, quantized_cache, cache_layout,
        cache_mode, dynamic_batching,
        True, False, False, False,
        0, dump_tensor_path, dump_steps
    )

    test_prompt = "I believe the meaning of life is"
    raw_text, test_prompt = make_context(tokenizer, test_prompt)
    
    _scale_up_prompt = []
    for _ in range(seqlen_scale_up):
        _scale_up_prompt.extend(test_prompt)
    test_prompt = _scale_up_prompt

    prompt_tokens = [test_prompt for _ in range(batch)]

    print(f"prepared {len(prompt_tokens)} prompts")
    results = generator.generate(
        prompt_tokens[:batch], tokenizer.get_eos_id(), tokenizer.get_pad_id(),
        max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, top_k=0
    )

    for result in results:
        if torch.is_tensor(result):
            result = result.cpu().numpy().tolist()
        result = decode_context(result, tokenizer=tokenizer, raw_text_len=len(raw_text), context_length=len(test_prompt))
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)