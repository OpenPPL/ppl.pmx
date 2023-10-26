import torch


class Rotary2DPositionEmbedding(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query: torch.Value, key: torch.Value, start_pos: torch.Value,
                 first_seq_len: torch.Value, pad_len: torch.Value = None,
                 theta: float = 10000.0, bypass_key: bool = False) -> torch.Value:
        if pad_len is not None:
            rotated_query, rotated_key = g.op('pmx::Rotary2DPositionEmbedding',
                                    query, key, start_pos, first_seq_len,
                                    pad_len, theta_f=theta,
                                    bypass_key_i=bypass_key,
                                    outputs=2)
        else:
            rotated_query, rotated_key = g.op('pmx::Rotary2DPositionEmbedding',
                                    query, key, start_pos, first_seq_len,
                                    theta_f=theta,
                                    bypass_key_i=bypass_key,
                                    outputs=2)
        return rotated_query.setTypeAs(query), rotated_key.setTypeAs(key)

    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, start_pos: torch.Tensor,
                first_seq_len: torch.Tensor, pad_len: torch.Tensor = None,
                theta: float = 10000.0, bypass_key: bool = False):
        if torch.onnx.is_in_onnx_export():
            return query, key

        bs, seqlen, rotary_dim = query.shape[0], query.shape[1], query.shape[-1] // 2

        pos0, pos1 = torch.zeros(2, bs, seqlen, dtype=start_pos.dtype, device=query.device)
        if pad_len is None:
            pad_len = torch.zeros(bs, dtype=start_pos)
        if seqlen != 1: # first predict
            for i in range(bs):
                pos0[i, pad_len[i]:] = torch.arange(seqlen - pad_len[i])
                pos0[i, seqlen - 1] = seqlen - pad_len[i] - 2
            pos1[:, seqlen - 1] = 1
        else:
            iters = start_pos - first_seq_len + 2
            pos0[:, 0] = first_seq_len - pad_len - 2
            pos1[:, 0] = iters
        pos0, pos1 = pos0.float(), pos1.float()


        # transfer to real mode
        _query = query.view(*query.shape[:-1], -1, 2).transpose(-2, -1).contiguous().flatten(-2)
        _key = key.view(*key.shape[:-1], -1, 2).transpose(-2, -1).contiguous().flatten(-2)
        
        # feature chunk
        query1, query2 = _query.chunk(2, dim=(_query.ndim - 1))
        key1, key2 = _key.chunk(2, dim=(_key.ndim - 1))

        # generate cos cache, sin cache
        freqs = (1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float, device=query.device)[: (rotary_dim // 2)] / rotary_dim)))

        freqs_cis1 = torch.zeros(bs, seqlen, rotary_dim // 2, device=query.device)
        freqs_cis2 = torch.zeros(bs, seqlen, rotary_dim // 2, device=query.device)

        for i in range(bs):
            freqs_cis1[i] = torch.outer(pos0[i], freqs)
            freqs_cis2[i] = torch.outer(pos1[i], freqs)
        
        cos1, sin1 = freqs_cis1.cos().unsqueeze(2), freqs_cis1.sin().unsqueeze(2)
        cos2, sin2 = freqs_cis2.cos().unsqueeze(2), freqs_cis2.sin().unsqueeze(2)


        def do_rotate(x_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            x_a = x_rot[..., :x_rot.shape[-1]//2]
            x_b = x_rot[..., x_rot.shape[-1]//2:]
            x_a_embed = x_a * cos - x_b * sin
            x_b_embed = x_b * cos + x_a * sin
            x_embed = torch.cat((x_a_embed, x_b_embed), dim=-1)
            return x_embed

        query1_emb = do_rotate(query1.float(), cos1, sin1)
        key1_emb = key1 if bypass_key else do_rotate(key1, cos1, sin1)
        query2_emb = do_rotate(query2.float(), cos2, sin2)
        key2_emb = key2 if bypass_key else do_rotate(key2, cos2, sin2)

        rotated_query = torch.cat((query1_emb, query2_emb), dim=-1)
        rotated_key = torch.cat((key1_emb, key2_emb), dim=-1)

        rotated_query = rotated_query.view(*rotated_query.shape[:-1], 2, -1).transpose(-2, -1).contiguous().flatten(-2)
        rotated_key = rotated_key.view(*rotated_key.shape[:-1], 2, -1).transpose(-2, -1).contiguous().flatten(-2)

        return rotated_query.type_as(query), rotated_key.type_as(key)


def rotary_2d_position_embedding(
        query: torch.Tensor, key: torch.Tensor, start_pos: torch.Tensor,
        first_seq_len: torch.Tensor, pad_len: torch.Tensor = None, 
        theta: float = 10000.0, bypass_key: bool = False):
    return Rotary2DPositionEmbedding.apply(query, key, start_pos,
                                           first_seq_len, pad_len,
                                           theta, bypass_key)


if __name__ == "__main__":
    class TestRotary2dModule(torch.nn.Module):
        def __init__(self, theta: float = 10000.0, bypass_key: bool = False) -> None:
            super().__init__()
            self.theta = theta
            self.bypass_key = bypass_key

        def forward(self, 
                    query: torch.Tensor, key: torch.Tensor, start_pos: torch.Tensor,
                    first_seq_len: torch.Tensor, pad_len: torch.Tensor = None):

            return rotary_2d_position_embedding(query, key, start_pos,
                    first_seq_len, pad_len, self.theta, self.bypass_key)

    bs = 2
    seqlen = 38
    num_head = 32
    head_size = 128

    theta = 10000
    max_seq_len = 1024

    query = torch.randn(bs, seqlen, num_head, head_size)
    key = torch.randn(bs, seqlen, num_head, head_size)

    start_pos = torch.tensor(0)
    pad_len = torch.tensor([2, 0])

    rotary = TestRotary2dModule(theta=theta)

    first_seq_len = torch.tensor(38)

    model_str1 = torch.onnx.export_to_pretty_string(
        rotary, (query, key, start_pos, first_seq_len, pad_len), "rotary1.onnx", input_names=["query", "key", "start_pos", "pad_len"], output_names=["rotated_query", "rotated_key"], opset_version=11)
    print(model_str1)
