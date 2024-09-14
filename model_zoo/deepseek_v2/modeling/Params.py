from dataclasses import dataclass

@dataclass
class DeepSeekV2Params:
    hidden_dim: int = 512
    intermediate_dim: int = 2048

    num_layers: int = 60
    num_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_dim: int = 128
    qk_rope_dim: int = 64
    v_head_dim: int = 128

    vocab_size: int = -1
    norm_eps: float = 1e-6

    cache_quant_bit: int = 0
    cache_quant_group: int = 8

    cache_layout: int = 0
    cache_mode: int = 0 # only affected when dynamic_batching == True
    page_size: int = 128  # only affected when cache_mode == 1

    dynamic_batching: bool = True
    auto_causal: bool = True

    rope_theta: float = 10000.0
    rope_scaling_type: str = 'yarn' # '', 'dynamic', 'linear', 'yarn'
    rope_scaling_factor: float = 40.0
    rope_scaling_beta_fast: int = 32
    rope_scaling_beta_slow: int = 1
    rope_scaling_mscale: float = 0.707
    rope_scaling_mscale_all_dim: float = 0.707
    origin_max_position_embeddings: int = 4096
    max_position_embeddings: int = 163840

    num_shared_experts: int = 2
    num_experts: int = 160
    num_experts_per_token: int = 6
    num_expert_groups: int = 8
    moe_topk_group: int = 3
    moe_topk_method: str = 'group_limited_greedy'
