import fire
import sys
import os
import json

from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import llama.modeling.Loader as Loader
from Tokenizer import Tokenizer
from ModelParams import ModelParams
import ConvertParamsToPmx

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    top_p: float = 0.95,
    batch: int = 4,
    seqlen_scale_up: int = 1,
    unaligned_batch: bool = False,
    max_gen_len: int = 256,
    friendly_gqa: bool = False, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool = True, # fuse qkv linear
    fused_kvcache: bool = True, # fuse key_value_cache and multi_head_attention
    auto_causal: bool = True, # causal mask is auto done by attention op, no need to pass additional mask to the model
    quantized_cache: bool = True, # 8bit kv cache quantization
    cache_layout: int = 0, # change kv cache layout for hardware performance friendly
    cache_mode: int = 0, # change kv cache indexing mode for memory management friendly, only affected when dynamic_batching == True
    dynamic_batching: bool = False, # use dynamic batching scheduling
    dump_tensor_path: str = None,
    dump_steps: List[int] = []
):
    tokenizer = Tokenizer(model_path=tokenizer_path)

    if not os.path.exists(Path(ckpt_dir) / "pmx_params.json"):
        print("Info: pmx_params.json not found, do auto param conversion")
        ConvertParamsToPmx.main(ckpt_dir, tokenizer_path)

    with open(Path(ckpt_dir) / "pmx_params.json", "r") as f:
        params = json.loads(f.read())
    params: ModelParams = ModelParams(**params)

    generator = Loader.load(
        ckpt_dir, params, friendly_gqa,
        fused_qkv, fused_kvcache, auto_causal,
        quantized_cache, cache_layout,
        cache_mode, dynamic_batching,
        False, False, False, False,
        0, dump_tensor_path, dump_steps
    )

    if unaligned_batch:
        test_prompt = [        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        ]
        test_prompt = [tokenizer.encode(t, bos=True, eos=False) for t in test_prompt]

        prompt_tokens = test_prompt.copy()
        for _ in range((batch - 1) // len(test_prompt)):
            prompt_tokens.extend(test_prompt)
    else:
        test_prompt = "I believe the meaning of life is"
        test_prompt = tokenizer.encode(test_prompt, bos=True, eos=False)

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
        print(tokenizer.decode(result))
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
