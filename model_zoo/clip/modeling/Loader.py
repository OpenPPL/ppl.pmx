import os
import sys
import torch
import time

from pathlib import Path
from typing import List

import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from ModelParams import ModelParams, VisionModelParams
from clip.modeling.static_batching.Model import TensorDumper, ClipVisionTransformer
import ModelParallel


def load(
    ckpt_dir: str,
    model_params: ModelParams,
    friendly_gqa: bool, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool, # fuse qkv linear
    auto_causal: bool, # causal mask is auto done by attention op, no need to pass additional mask to the model
    attn_wqkv_bias_term: bool,
    attn_wo_bias_term: bool,
    ffn_linear_bias_term: bool,
    load_to_cpu: bool,
    dump_tensor_path: str = None,
    dump_steps: List[int] = []
):
    start_time = time.time()
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

    model_params.auto_causal = bool(auto_causal)
    #if load_to_cpu:
    #    torch.set_default_tensor_type(torch.HalfTensor)
    #else:
    #    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = ClipVisionTransformer(model_params,
                        friendly_gqa,
                        fused_qkv,
                        attn_wqkv_bias_term,
                        attn_wo_bias_term,
                        ffn_linear_bias_term,
                        proc_group=proc_group)
    torch.set_default_tensor_type(torch.FloatTensor)

    model.load_state_dict(checkpoint)

    if dump_tensor_path is not None:
        dump_path = os.path.join(dump_tensor_path, "rank_{}".format(local_rank))
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        TensorDumper.dir = dump_path
        TensorDumper.enable_dump = True
        TensorDumper.dump_steps = dump_steps

    del checkpoint

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model
