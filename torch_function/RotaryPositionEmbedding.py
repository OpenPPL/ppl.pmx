import torch


class RotaryPositionEmbedding(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query: torch.Value, key: torch.Value,
                start_pos: torch.Value, pad_len: torch.Value,
                rotary_dim: int = 0, theta: float = 10000.0,
                bypass_key: bool = False, max_position_embeddings: int = 2048,
                scaling_type: str = '', scaling_factor: float = 1.0):
        # g: GraphContext, defined in onnx/_internal/jit_utils.py
        if pad_len is not None:
            rotated_query, rotated_key = g.op('pmx::RotaryPositionEmbedding',
                query, key, start_pos, pad_len,
                rotary_dim_i=rotary_dim,
                theta_f=theta,
                bypass_key_i=bypass_key,
                max_position_embeddings_i=max_position_embeddings,
                scaling_type_s=scaling_type,
                scaling_factor_f=scaling_factor,
                outputs=2)
        else:
            rotated_query, rotated_key = g.op('pmx::RotaryPositionEmbedding',
                query, key, start_pos,
                rotary_dim_i=rotary_dim,
                theta_f=theta,
                bypass_key_i=bypass_key,
                max_position_embeddings_i=max_position_embeddings,
                scaling_type_s=scaling_type,
                scaling_factor_f=scaling_factor,
                outputs=2)
        return rotated_query.setTypeAs(query), rotated_key.setTypeAs(key)

    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor,
                start_pos: torch.Tensor, pad_len: torch.Tensor = None,
                rotary_dim: int = 0, theta: float = 10000.0,
                bypass_key: bool = False, max_position_embeddings: int = 2048,
                scaling_type: str = '', scaling_factor: float = 1.0):
        if torch.onnx.is_in_onnx_export():
            return query, key

        # shape of query, key: [batch, seqlen, num_heads, head_dim]
        bs, seqlen = query.shape[0], query.shape[1]
        dim = query.shape[3] if rotary_dim == 0 else rotary_dim

        if pad_len is None:
            pad_len = torch.zeros(bs, dtype=start_pos.dtype, device=start_pos.device)

        # generate cos cache, sin cache
        if scaling_type == 'dynamic' and seqlen > max_position_embeddings:
            theta = theta * (
                (scaling_factor * seqlen / max_position_embeddings) - (scaling_factor - 1)
            ) ** (dim / (dim - 2))

        freqs = (1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=query.device)[: (dim // 2)] / dim)))
        freqs_cis = torch.zeros(bs, seqlen, dim // 2, dtype=torch.float, device=query.device)

        for i in range(bs):
            t = torch.arange(start_pos.item() - pad_len[i], start_pos.item() - pad_len[i] + seqlen, dtype=torch.float, device=query.device)
            if scaling_type == 'linear':
                t = t / scaling_factor
            freqs_cis[i] = torch.outer(t, freqs)

        cos, sin = freqs_cis.cos().unsqueeze(2), freqs_cis.sin().unsqueeze(2)  # (bs, seqlen, 1, dim / 2)


        def do_rotate(x: torch.Tensor, cos: torch.tensor, sin: torch.tensor):
            x_rot = x[..., :dim]
            x_pass = x[..., dim:]
            x_rot = x_rot.view(*x.shape[:-1], -1, 2).transpose(-2, -1).contiguous().flatten(-2)

            x_a = x_rot[..., :x_rot.shape[-1] // 2]
            x_b = x_rot[..., x_rot.shape[-1] // 2:]
            x_a_embed = x_a * cos - x_b * sin
            x_b_embed = x_b * cos + x_a * sin
            x_embed = torch.cat((x_a_embed, x_b_embed), dim=-1)
            
            x_embed = x_embed.view(*x_embed.shape[:-1], 2, -1).transpose(-2, -1).contiguous().flatten(-2)
            x_embed = torch.cat((x_embed, x_pass), dim=-1)
            return x_embed

        rotated_query = do_rotate(query.float(), cos, sin).type_as(query)
        rotated_key = key if bypass_key else do_rotate(key.float(), cos, sin).type_as(key)

        return rotated_query, rotated_key


def rotary_position_embedding(query: torch.Tensor, key: torch.Tensor,
                start_pos: torch.Tensor, pad_len: torch.Tensor = None,
                rotary_dim: int = 0, theta: float = 10000.0, bypass_key: bool = False,
                max_position_embeddings: int = 2048, scaling_type: str = '',
                scaling_factor: float = 1.0) -> torch.Tensor:
    return RotaryPositionEmbedding.apply(query, key, start_pos, pad_len, rotary_dim, theta, bypass_key,
                                         max_position_embeddings, scaling_type, scaling_factor)


if __name__ == "__main__":
    class TestRotaryModule(torch.nn.Module):
        def __init__(self, rotary_dim: int = 0, theta: float = 10000.0, bypass_key: bool = False,
                     max_position_embeddings: int = 2048, scaling_type: str = '',
                     scaling_factor: float = 1.0) -> None:
            super().__init__()
            self.rotary_dim = rotary_dim
            self.theta = theta
            self.bypass_key = bypass_key
            self.max_position_embeddings = max_position_embeddings
            self.scaling_type = scaling_type
            self.scaling_factor = scaling_factor


        def forward(self, query: torch.Tensor, key: torch.Tensor,
                start_pos: torch.Tensor, pad_len: torch.Tensor = None):
            return rotary_position_embedding(query, key, start_pos, pad_len,
                                    self.rotary_dim, self.theta, self.bypass_key,
                                    self.max_position_embeddings, self.scaling_type,
                                    self.scaling_factor)


    bs = 2
    seqlen = 1038
    num_head = 32
    head_dim = 128

    theta = 10000
    max_seqlen = 1024
    rotary_dim = 80

    q = torch.randn(bs, seqlen, num_head, head_dim)
    k = torch.randn(bs, seqlen, num_head, head_dim)

    start_pos = torch.tensor(2)
    # rotary = TestRotaryModule(rotary_dim, theta=theta)
    rotary = TestRotaryModule(rotary_dim, theta=theta, scaling_type='dynamic', scaling_factor=2.0)

    model_str1 = torch.onnx.export_to_pretty_string(
       rotary, (q, k, start_pos), "RotaryPositionEmbedding1.onnx",
       input_names=["query", "key", "start_pos"], output_names=["query_out", "key_out"], opset_version=11)

    pad_len = torch.tensor([2, 0])
    model_str2 = torch.onnx.export_to_pretty_string(
       rotary, (q, k, start_pos, pad_len), "RotaryPositionEmbedding2.onnx",
       input_names=["query", "key", "start_pos", "pad_len"], output_names=["query_out", "key_out"], opset_version=11)

    print(model_str1)
    print(model_str2)
