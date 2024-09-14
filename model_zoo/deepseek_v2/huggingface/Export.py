import fire
import sys
import os
import json

from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import deepseek_v2.modeling.Loader as Loader
from deepseek_v2.modeling.Params import DeepSeekV2Params as ModelParams

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    export_path: str,
    expert_parallel_mode: str = "etp", # "etp" for expert-tensor-parellel, "edp" for expert-data-parallel
    cache_layout: int = 0, # change kv cache layout for hardware performance friendly
    cache_mode: int = 0, # change kv cache indexing mode for memory management friendly, only affected when dynamic_batching == True
    dynamic_batching: bool = True, # use dynamic batching scheduling
    dump_tensor_path: str = None,
    dump_steps: List[int] = []
):
    with open(Path(ckpt_dir) / "opmx_params.json", "r") as f:
        params = json.loads(f.read())
    params: ModelParams = ModelParams(**params)

    generator = Loader.load(
        ckpt_dir, params,
        expert_parallel_mode=expert_parallel_mode,
        cache_layout=cache_layout,
        cache_mode=cache_mode,
        dynamic_batching=dynamic_batching,
        load_to_cpu=False,
        dump_tensor_path=dump_tensor_path,
        dump_steps=dump_steps
    )

    generator.export(export_path)

if __name__ == "__main__":
    fire.Fire(main)
