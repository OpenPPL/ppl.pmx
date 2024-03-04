import fire
import sys
import os
import json
import torch

from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import clip.modeling.Loader as Loader
from ModelParams import ModelParams, VisionModelParams

def main(
    ckpt_dir: str,
    export_path: str,
    friendly_gqa: bool = False, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool = True, # fuse qkv linear
    auto_causal: bool = True, # causal mask is auto done by attention op, no need to pass additional mask to the model
):
    with open(Path(ckpt_dir) / "pmx_vision_params.json", "r") as f:
        params = json.loads(f.read())
    params: VisionModelParams = VisionModelParams(**params)

    model = Loader.load(
        ckpt_dir, params, friendly_gqa,
        fused_qkv, auto_causal,
        True, True, True, True
    )

    # export model
    pixel_values = torch.ones([1, 3, params.image_size, params.image_size], dtype=torch.float32)
    attn_mask = torch.empty(0, dtype=torch.float32)

    # to do: dynamic batch / dump json
    torch.onnx.export(
        model.cpu(),
        (pixel_values, attn_mask),
        os.path.join(export_path, "model.onnx"),
        input_names=["pixel_values", "attn_mask"],
        output_names=["vision_logits"],
        do_constant_folding=True,
        opset_version=11,
    )


if __name__ == "__main__":
    fire.Fire(main)
