import torch


class Rotary2DPositionEmbedding(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query: torch.Value, key: torch.Value,
                seqstarts: torch.Value, start_pos: torch.Value,
                max_seqlen: torch.Value, first_seq_len: torch.Value,
                theta: float = 10000.0, bypass_key: bool = False):
        # g: GraphContext, defined in onnx/_internal/jit_utils.py
        rotated_query, rotated_key = g.op('opmx.dynamic_batching::Rotary2DPositionEmbedding',
            query, key, seqstarts, start_pos, max_seqlen, first_seq_len,
            theta_f=theta,
            bypass_key_i=bypass_key,
            outputs=2)
        return rotated_query.setTypeAs(query), rotated_key.setTypeAs(key)
    
    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, 
                seqstarts: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, first_seq_len: torch.Tensor, 
                theta: float = 10000.0, bypass_key: bool = False):

        if torch.onnx.is_in_onnx_export():
            return query, key

        # shape of query, key: [seqstarts[batch], num_heads, head_dim]
        
        dim = query.shape[-1] // 2

        def do_rotate(x_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            x_a = x_rot[..., :x_rot.shape[-1]//2]
            x_b = x_rot[..., x_rot.shape[-1]//2:]
            x_a_embed = x_a * cos - x_b * sin
            x_b_embed = x_b * cos + x_a * sin
            x_embed = torch.cat((x_a_embed, x_b_embed), dim=-1)
            return x_embed

        # feature chunk
        query1, query2 = query.chunk(2, dim=(query.ndim - 1))
        key1, key2 = key.chunk(2, dim=(key.ndim - 1))

        # transfer to real mode
        query1 = query1.view(*query1.shape[:-1], -1, 2).transpose(-2, -1).contiguous().flatten(-2)
        query2 = query2.view(*query2.shape[:-1], -1, 2).transpose(-2, -1).contiguous().flatten(-2)

        key1 = key1.view(*key1.shape[:-1], -1, 2).transpose(-2, -1).contiguous().flatten(-2)
        key2 = key2.view(*key2.shape[:-1], -1, 2).transpose(-2, -1).contiguous().flatten(-2)

        rotated_query1, rotated_query2 = torch.zeros_like(query1), torch.zeros_like(query2)
        rotated_key1, rotated_key2 = torch.zeros_like(key1), torch.zeros_like(key2)
        freqs = (1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=query.device)[: (dim // 2)] / dim)))
        seqlens = seqstarts[1:] - seqstarts[:-1]
        for b, seqlen in enumerate(seqlens):
            # for position 0 and position 1
            if seqlen > 1: # first fill
                pos1 = torch.arange(0, seqlen, dtype=torch.float, device=query.device)
                pos1[-1] = pos1[seqlen - 2]
                pos2 = torch.zeros(seqlen, dtype=torch.float, device=query.device)
                pos2[-1] = 1
            else:   # decoding
                pos1 = torch.tensor([first_seq_len[b] - 2], dtype=torch.float, device=query.device)
                pos2 = torch.tensor([start_pos[b] - first_seq_len[b] + 2], dtype=torch.float, device=query.device)

            seqbeg = seqstarts[b]
            seqend = seqstarts[b+1]
            
            freqs_cis1 = torch.outer(pos1, freqs)
            freqs_cis2 = torch.outer(pos2, freqs)
            cos1, sin1 = freqs_cis1.cos().unsqueeze(1), freqs_cis1.sin().unsqueeze(1)  # (seqlen, 1, dim / 2)
            cos2, sin2 = freqs_cis2.cos().unsqueeze(1), freqs_cis2.sin().unsqueeze(1)  
            
            rotated_query1[seqbeg:seqend] = do_rotate(query1[seqbeg:seqend].float(), cos1, sin1).type_as(query)
            rotated_key1[seqbeg:seqend] = key1[seqbeg:seqend] if bypass_key else do_rotate(key1[seqbeg:seqend].float(), cos1, sin1).type_as(key)

            rotated_query2[seqbeg:seqend] = do_rotate(query2[seqbeg:seqend].float(), cos2, sin2).type_as(query)
            rotated_key2[seqbeg:seqend] = key2[seqbeg:seqend] if bypass_key else do_rotate(key2[seqbeg:seqend].float(), cos2, sin2).type_as(key)

        rotated_query1 = rotated_query1.view(*rotated_query1.shape[:-1], 2, -1).transpose(-2, -1).contiguous().flatten(-2)
        rotated_query2 = rotated_query2.view(*rotated_query1.shape[:-1], 2, -1).transpose(-2, -1).contiguous().flatten(-2)
        rotated_key1 = rotated_key1.view(*rotated_key1.shape[:-1], 2, -1).transpose(-2, -1).contiguous().flatten(-2)
        rotated_key2 = rotated_key2.view(*rotated_key2.shape[:-1], 2, -1).transpose(-2, -1).contiguous().flatten(-2)

        rotated_query = torch.cat((rotated_query1, rotated_query2), dim=-1)
        rotated_key = torch.cat((rotated_key1, rotated_key2), dim=-1)

        return rotated_query.type_as(query), rotated_key.type_as(key)

def rotary_2d_position_embedding(query: torch.Tensor, key: torch.Tensor, 
                seqstarts: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, first_seq_len: torch.Tensor,
                theta: float = 10000.0, bypass_key: bool = False) -> torch.Tensor:
    return Rotary2DPositionEmbedding.apply(query, key, seqstarts, start_pos, max_seqlen, first_seq_len, theta, bypass_key)

if __name__ == "__main__":
    class TestRotary2DModule(torch.nn.Module):
        def __init__(self, theta: float = 10000.0, bypass_key: bool = False) -> None:
            super().__init__()
            self.theta = theta
            self.bypass_key = bypass_key

        def forward(self, query: torch.Tensor, key: torch.Tensor, 
                seqstarts: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, first_seq_len: torch.Tensor):
            return rotary_2d_position_embedding(query, key, seqstarts, start_pos, max_seqlen,
                                    first_seq_len, self.theta, self.bypass_key)
            
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
    first_seql_len = torch.tensor([16, 16])
    
    max_seqlen = torch.tensor([seqlen])
    rotary = TestRotary2DModule(theta=theta)
    
    rotary_query, rotary_key = rotary(q, k, seqstarts, start_pos, max_seqlen, first_seql_len)
    # print(rotary_query.shape, rotary_key.shape)
    model_str1 = torch.onnx.export_to_pretty_string(
       rotary, (q, k, seqstarts, start_pos, max_seqlen, first_seql_len), "Rotary2DPositionEmbedding1.onnx",
       input_names=["query", "key", "seqstarts", "start_pos", "max_seqlen", "first_seq_len"], output_names=["query_out", "key_out"], opset_version=11)

    print(model_str1)