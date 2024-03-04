import fire
import sys
import os
import json
import torch

from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import clip.modeling.Loader as Loader
from ModelParams import ModelParams, VisionModelParams

def main(
    ckpt_dir: str,
    batch: int = 1,
    friendly_gqa: bool = False, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool = True, # fuse qkv linear
    auto_causal: bool = True, # causal mask is auto done by attention op, no need to pass additional mask to the model
    dump_tensor_path: str = None,
    dump_steps: List[int] = []
):

    with open(Path(ckpt_dir) / "pmx_vision_params.json", "r") as f:
        params = json.loads(f.read())
    params: VisionModelParams = VisionModelParams(**params)

    model = Loader.load(
        ckpt_dir, params, friendly_gqa,
        fused_qkv, auto_causal,
        True, True, True, False,
        dump_tensor_path, dump_steps
    )


    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    attn_mask = torch.empty(0, dtype=torch.float32)
    #pixel_values = torch.ones([1,3,224,224], dtype=torch.float32)

    pixel_values = torch.load('test_input.pt')

    #from PIL import Image
    #import requests
    #from transformers import AutoProcessor, CLIPVisionModelWithProjection
    #model_hf = CLIPVisionModelWithProjection.from_pretrained("/home/jizhe1/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/e6a30b603a447e251fdaca1c3056b2a16cdfebeb")
    #processor = AutoProcessor.from_pretrained("/home/jizhe1/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/e6a30b603a447e251fdaca1c3056b2a16cdfebeb")
    #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)
    #inputs = processor(images=image, return_tensors="pt")

    outputs = model.forward(pixel_values, attn_mask)

if __name__ == "__main__":
    fire.Fire(main)
