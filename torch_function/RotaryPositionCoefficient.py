import torch
import math

torch2onnx_dtype = {torch.float16: 10,
                    torch.float32: 1}

def __rotray_yarn_coeff(max_seqlen: torch.Tensor,
                    device: torch.device, data_type: torch.dtype,
                    rotary_dim: int,
                    theta: float = 10000.0,
                    original_max_position_embeddings: int = 4096, 
                    scaling_factor: float = 1.0,
                    scaling_beta_fast: int = 32,
                    scaling_beta_slow: int = 1,
                    scaling_mscale: float = 0.707,
                    scaling_mscale_all_dim: float = 0.707):
    # Inverse dim formula to find dim based on number of rotations
    def yarn_find_correction_dim(
        num_rotations, dim, base=10000, max_position_embeddings=2048
    ):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    # Find dim range bounds based on rotations
    def yarn_find_correction_range(
        low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
    ):
        low = math.floor(
            yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)  # Clamp values just in case

    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def yarn_linear_ramp_mask(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
    
    dim = rotary_dim

    freq_extra = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    freq_inter = 1.0 / (
        scaling_factor
        * theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )

    low, high = yarn_find_correction_range(
        scaling_beta_fast,
        scaling_beta_slow,
        dim,
        theta,
        original_max_position_embeddings,
    )
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
        device=device, dtype=torch.float32
    )
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    t = torch.arange(max_seqlen.item(), device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)

    _mscale = float(
        yarn_get_mscale(scaling_factor, scaling_mscale)
        / yarn_get_mscale(scaling_factor, scaling_mscale_all_dim)
    )

    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * _mscale).to(data_type), (emb.sin() * _mscale).to(data_type)


class RotaryPositionCoefficient(torch.autograd.Function):
    @staticmethod
    def symbolic(g, max_seqlen: torch.Value,
                device: torch.device, data_type: torch.dtype,
                rotary_dim: int,
                theta: float = 10000.0, max_position_embeddings: int = 2048,
                original_max_position_embeddings: int = 4096, 
                scaling_type: str = '',
                scaling_factor: float = 1.0,
                scaling_beta_fast: int = 32,
                scaling_beta_slow: int = 1,
                scaling_mscale: float = 0.707,
                scaling_mscale_all_dim: float = 0.707):
        # g: GraphContext, defined in onnx/_internal/jit_utils.py
        rotary_sin, rotary_cos = g.op('opmx::RotaryPositionCoefficient',
            max_seqlen,
            data_type_i=data_type,
            rotary_dim_i=rotary_dim,
            theta_f=theta,
            max_position_embeddings_i=max_position_embeddings,
            original_max_position_embeddings_i=original_max_position_embeddings,
            scaling_type_s=scaling_type,
            scaling_factor_f=scaling_factor,
            scaling_beta_fast_i=scaling_beta_fast,
            scaling_beta_slow_i=scaling_beta_slow,
            scaling_mscale_f=scaling_mscale,
            scaling_mscale_all_dim_f=scaling_mscale_all_dim,
            outputs=2)
        return rotary_sin, rotary_cos

    @staticmethod
    def forward(ctx,
                max_seqlen: torch.Tensor, 
                device: torch.device, data_type: torch.dtype,
                rotary_dim: int,
                theta: float = 10000.0, max_position_embeddings: int = 2048,
                original_max_position_embeddings: int = 4096, 
                scaling_type: str = '',
                scaling_factor: float = 1.0,
                scaling_beta_fast: int = 32,
                scaling_beta_slow: int = 1,
                scaling_mscale: float = 0.707,
                scaling_mscale_all_dim: float = 0.707):

        if torch.onnx.is_in_onnx_export():
            coeff = torch.zeros(max_seqlen, rotary_dim, dtype=data_type)
            return coeff, coeff

        assert scaling_type == 'yarn'

        if scaling_type == 'yarn':
            rotary_sin, rotary_cos = __rotray_yarn_coeff(
                max_seqlen, device, data_type,
                rotary_dim, theta,
                original_max_position_embeddings,
                scaling_factor, scaling_beta_fast, scaling_beta_slow,
                scaling_mscale, scaling_mscale_all_dim
            )

        return rotary_sin, rotary_cos


def rotary_position_coefficient(
                max_seqlen: torch.Tensor, 
                device: torch.device, data_type: torch.dtype,
                rotary_dim: int,
                theta: float = 10000.0, max_position_embeddings: int = 2048,
                original_max_position_embeddings: int = 4096, 
                scaling_type: str = '',
                scaling_factor: float = 1.0,
                scaling_beta_fast: int = 32,
                scaling_beta_slow: int = 1,
                scaling_mscale: float = 0.707,
                scaling_mscale_all_dim: float = 0.707) -> torch.Tensor:
    return RotaryPositionCoefficient.apply(
        max_seqlen, device, data_type, rotary_dim,
        theta, max_position_embeddings, original_max_position_embeddings,
        scaling_type, scaling_factor, scaling_beta_fast, scaling_beta_slow,
        scaling_mscale, scaling_mscale_all_dim)


# if __name__ == "__main__":
#     class TestRotaryModule(torch.nn.Module):
#         def __init__(self, rotary_dim: int = 0, theta: float = 10000.0, bypass_key: bool = False,
#                      max_position_embeddings: int = 2048, scaling_type: str = '',
#                      scaling_factor: float = 1.0) -> None:
#             super().__init__()
#             self.rotary_dim = rotary_dim
#             self.theta = theta
#             self.bypass_key = bypass_key
#             self.max_position_embeddings = max_position_embeddings
#             self.scaling_type = scaling_type
#             self.scaling_factor = scaling_factor


#         def forward(self, query: torch.Tensor, key: torch.Tensor,
#                 start_pos: torch.Tensor, pad_len: torch.Tensor = None):
#             return rotary_position_embedding(query, key, start_pos, pad_len,
#                                     self.rotary_dim, self.theta, self.bypass_key,
#                                     self.max_position_embeddings, self.scaling_type,
#                                     self.scaling_factor)


#     bs = 2
#     seqlen = 1038
#     num_head = 32
#     head_dim = 128

#     theta = 10000
#     max_seqlen = 1024
#     rotary_dim = 80

#     q = torch.randn(bs, seqlen, num_head, head_dim)
#     k = torch.randn(bs, seqlen, num_head, head_dim)

#     start_pos = torch.tensor(2)
#     # rotary = TestRotaryModule(rotary_dim, theta=theta)
#     rotary = TestRotaryModule(rotary_dim, theta=theta, scaling_type='dynamic', scaling_factor=2.0)

#     model_str1 = torch.onnx.export_to_pretty_string(
#        rotary, (q, k, start_pos), "RotaryPositionEmbedding1.onnx",
#        input_names=["query", "key", "start_pos"], output_names=["query_out", "key_out"], opset_version=11)

#     pad_len = torch.tensor([2, 0])
#     model_str2 = torch.onnx.export_to_pretty_string(
#        rotary, (q, k, start_pos, pad_len), "RotaryPositionEmbedding2.onnx",
#        input_names=["query", "key", "start_pos", "pad_len"], output_names=["query_out", "key_out"], opset_version=11)

#     print(model_str1)
#     print(model_str2)
