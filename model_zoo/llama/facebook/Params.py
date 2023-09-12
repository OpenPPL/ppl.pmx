import sys
import os
from dataclasses import dataclass
from typing import Optional
import json

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

from ModelParams import ModelParams


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    ffn_dim_multiplier: Optional[float] = None


def cvt_model_args(args: ModelArgs) -> ModelParams:
    ret = ModelParams()
    ret.hidden_dim = args.dim
    ret.num_layers = args.n_layers
    ret.num_heads = args.n_heads
    ret.num_kv_heads = args.n_kv_heads

    intermediate_dim = int(2 * 4 * args.dim / 3)
    # custom dim factor multiplier
    if args.ffn_dim_multiplier is not None:
        intermediate_dim = int(args.ffn_dim_multiplier * intermediate_dim)
    intermediate_dim = args.multiple_of * ((intermediate_dim + args.multiple_of - 1) // args.multiple_of)
    ret.intermediate_dim = intermediate_dim

    ret.norm_eps = args.norm_eps
    ret.vocab_size = args.vocab_size
    
    return ret


def load(params_path: str) -> ModelArgs:
    with open(params_path, "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(**params)
    model_args.n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads

    return model_args
