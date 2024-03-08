import torch


class RotaryPositionEmbedding(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query: torch.Value, key: torch.Value,
                seqstarts: torch.Value, start_pos: torch.Value,
                max_seqlen: torch.Value, rotary_dim: int = 0,
                theta: float = 10000.0, bypass_key: bool = False,
                max_position_embeddings: int = 2048,
                scaling_type: str = '', scaling_factor: float = 1.0):
        # g: GraphContext, defined in onnx/_internal/jit_utils.py
        rotated_query, rotated_key = g.op('pmx.dynamic_batching::RotaryPositionEmbedding',
            query, key, seqstarts, start_pos, max_seqlen,
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
                seqstarts: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, rotary_dim: int = 0,
                theta: float = 10000.0, bypass_key: bool = False,
                max_position_embeddings: int = 2048,
                scaling_type: str = '', scaling_factor: float = 1.0):
        if torch.onnx.is_in_onnx_export():
            return query, key

        # shape of query, key: [seqstarts[batch], num_heads, head_dim]
        dim = query.shape[2] if rotary_dim == 0 else rotary_dim

        def do_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            x_rot = x[..., :dim]
            x_pass = x[..., dim:]
            x_rot = x_rot.view(*x_rot.shape[:-1], -1, 2).transpose(-2, -1).contiguous().flatten(-2)
            x_a = x_rot[..., :x_rot.shape[-1] // 2]
            x_b = x_rot[..., x_rot.shape[-1] // 2:]
            x_a_embed = x_a * cos - x_b * sin
            x_b_embed = x_b * cos + x_a * sin
            x_embed = torch.cat((x_a_embed, x_b_embed), dim=-1)

            x_embed = x_embed.view(*x_embed.shape[:-1], 2, -1).transpose(-2, -1).contiguous().flatten(-2)

            x_embed = torch.cat((x_embed, x_pass), dim=-1)
            return x_embed

        rotated_query = torch.zeros_like(query)
        rotated_key = torch.zeros_like(key)

        # freqs = (1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=query.device)[: (dim // 2)] / dim)))
        seqlens = seqstarts[1:] - seqstarts[:-1]
        for b, seqlen in enumerate(seqlens):
            position = start_pos[b]
            seqbeg = seqstarts[b]
            seqend = seqstarts[b+1]
            # generate cos cache, sin cache
            t = torch.arange(position, position + seqlen, dtype=torch.float, device=query.device)
            if scaling_type == 'linear':
                t = t / scaling_factor
            if scaling_type == 'dynamic' and seqlen > max_position_embeddings:
                theta = theta * (
                    (scaling_factor * seqlen / max_position_embeddings) - (scaling_factor - 1)
                ) ** (dim / (dim - 2))
            freqs = (1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=query.device)[: (dim // 2)] / dim)))
            freqs_cis = torch.outer(t, freqs)

            cos, sin = freqs_cis.cos().unsqueeze(1), freqs_cis.sin().unsqueeze(1)  # (seqlen, 1, dim / 2)

            rotated_query[seqbeg:seqend] = do_rotate(query[seqbeg:seqend].float(), cos, sin).type_as(query)
            rotated_key[seqbeg:seqend] = key[seqbeg:seqend] if bypass_key else do_rotate(key[seqbeg:seqend].float(), cos, sin).type_as(key)

        return rotated_query, rotated_key


def rotary_position_embedding(query: torch.Tensor, key: torch.Tensor,
                seqstarts: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, rotary_dim: int = 0,
                theta: float = 10000.0, bypass_key: bool = False,
                max_position_embeddings: int = 2048, scaling_type: str = '',
                scaling_factor: float = 1.0) -> torch.Tensor:
    return RotaryPositionEmbedding.apply(query, key, seqstarts, start_pos, max_seqlen, rotary_dim, theta, bypass_key,
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
                seqstarts: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor):
            return rotary_position_embedding(query, key, seqstarts, start_pos, max_seqlen,
                                    self.rotary_dim, self.theta, self.bypass_key,
                                    self.max_position_embeddings, self.scaling_type,
                                    self.scaling_factor)


    bs = 2
    seqlen = 16
    num_head = 32
    head_dim = 128

    theta = 10000
    rotary_dim = 80

    q = torch.randn(bs * seqlen, num_head, head_dim)
    k = torch.randn(bs * seqlen, num_head, head_dim)

    start_pos = torch.tensor([2, 2], dtype=torch.int64)
    seqstarts = torch.tensor([0, seqlen, seqlen], dtype=torch.int64).cumsum(dim=0)

    max_seqlen = torch.tensor([seqlen])

    rotary = TestRotaryModule(rotary_dim, theta=theta)

    model_str1 = torch.onnx.export_to_pretty_string(
       rotary, (q, k, seqstarts, start_pos, max_seqlen), "RotaryPositionEmbedding1.onnx",
       input_names=["query", "key", "seqstarts", "start_pos", "max_seqlen"], output_names=["query_out", "key_out"], opset_version=11)

    print(model_str1)
