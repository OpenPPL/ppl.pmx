import os
import sys
import torch
import time

from pathlib import Path
from typing import List

import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from Params import DeepSeekV2Params
from ModelUtils import __TextGenerator__
import ModelParallel

def load(
    ckpt_dir: str,
    model_params: DeepSeekV2Params,
    expert_parallel_mode: str,
    cache_layout: int, # change kv cache layout for hardware performance friendly
    cache_mode: int, # change kv cache indexing mode for memory management friendly, only affected when dynamic_batching == True
    dynamic_batching: bool, # use dynamic batching scheduling
    load_to_cpu: bool,
    dump_tensor_path: str = None,
    dump_steps: List[int] = []
) -> __TextGenerator__:
    start_time = time.time()

    assert dynamic_batching == True, "only support dynamic batching"
    assert len(expert_parallel_mode) == 3 and ('etp' in expert_parallel_mode or 'edp' in expert_parallel_mode), "only support 'etp' and 'edp'"

    if dynamic_batching:
        from deepseek_v2.modeling.dynamic_batching.Model import TensorDumper, Transformer
        from deepseek_v2.modeling.dynamic_batching.Pipeline import Pipeline
        if cache_layout != 3:
            print("Info: we suggest using cache_layout 3 for cuda inference performance")

    local_rank, world_size = ModelParallel.setup(load_to_cpu)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"

    ckpt_path = checkpoints[local_rank]

    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    proc_group = dist.new_group(ranks=[_ for _ in range(world_size)], backend=
                                'gloo' if load_to_cpu else 'nccl')

    model_params.dynamic_batching = bool(dynamic_batching)
    model_params.auto_causal = True
    model_params.cache_layout = cache_layout
    model_params.cache_mode = cache_mode
    model_params.cache_quant_bit = 0
    model_params.cache_quant_group = 0

    if load_to_cpu:
        torch.set_default_tensor_type(torch.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_params, expert_parallel_mode, proc_group=proc_group)
    torch.set_default_tensor_type(torch.FloatTensor)

    model.load_state_dict(checkpoint)

    generator = Pipeline(model)

    if dump_tensor_path is not None:
        dump_path = os.path.join(dump_tensor_path, "rank_{}".format(local_rank))
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        TensorDumper.dir = dump_path
        TensorDumper.enable_dump = True
        TensorDumper.dump_steps = dump_steps

    del checkpoint

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def random(
    ckpt_dir: str,
    model_params: DeepSeekV2Params,
    expert_parallel_mode: str,
    cache_layout: int, # change kv cache layout for hardware performance friendly
    cache_mode: int, # change kv cache indexing mode for memory management friendly, only affected when dynamic_batching == True
    dynamic_batching: bool, # use dynamic batching scheduling
    load_to_cpu: bool,
    dump_tensor_path: str = None,
    dump_steps: List[int] = []
) -> __TextGenerator__:
    start_time = time.time()

    assert dynamic_batching == True, "only support dynamic batching"
    assert len(expert_parallel_mode) == 3 and ('etp' in expert_parallel_mode or 'edp' in expert_parallel_mode), "only support 'etp' and 'edp'"

    if dynamic_batching:
        from deepseek_v2.modeling.dynamic_batching.Model import TensorDumper, Transformer
        from deepseek_v2.modeling.dynamic_batching.Pipeline import Pipeline
        if cache_layout != 3:
            print("Info: we suggest using cache_layout 3 for cuda inference performance")

    local_rank, world_size = ModelParallel.setup(load_to_cpu)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"

    ckpt_path = checkpoints[local_rank]

    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    proc_group = dist.new_group(ranks=[_ for _ in range(world_size)], backend=
                                'gloo' if load_to_cpu else 'nccl')

    model_params.dynamic_batching = bool(dynamic_batching)
    model_params.auto_causal = True
    model_params.cache_layout = cache_layout
    model_params.cache_mode = cache_mode
    model_params.cache_quant_bit = 0
    model_params.cache_quant_group = 0

    if load_to_cpu:
        torch.set_default_tensor_type(torch.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_params, expert_parallel_mode, proc_group=proc_group)
    torch.set_default_tensor_type(torch.FloatTensor)

    print("Randomizing")
    model.random_weight()

    generator = Pipeline(model)

    if dump_tensor_path is not None:
        dump_path = os.path.join(dump_tensor_path, "rank_{}".format(local_rank))
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        TensorDumper.dir = dump_path
        TensorDumper.enable_dump = True
        TensorDumper.dump_steps = dump_steps

    del checkpoint

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator
