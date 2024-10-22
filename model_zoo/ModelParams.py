from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelParams:
    hidden_dim: int = 512
    intermediate_dim: int = 2048
    head_dim: Optional[int] = None

    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: Optional[int] = None

    vocab_size: int = -1
    norm_eps: float = 1e-5

    cache_quant_bit: int = 8
    cache_quant_group: int = 8

    cache_layout: int = 0
    cache_mode: int = 0 # only affected when dynamic_batching == True
    page_size: int = 128  # only affected when cache_mode == 1

    dynamic_batching: bool = True
    auto_causal: bool = True

    rope_theta: float = 10000.0
    rope_scaling_type: str = '' # '', 'dynamic', 'linear'
    rope_scaling_factor: float = 1.0
    max_position_embeddings: int = 2048

    num_experts: int = 1
    num_experts_per_token: int = 1
    sliding_window: int = 0
