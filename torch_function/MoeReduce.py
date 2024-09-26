import torch


class MoeReduce(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g: torch._C.Graph, Y: torch.Value, expert_weights: torch.Value, 
        invert_permutation: torch.Value, num_experts_per_token: int):

        Y_out = g.op("opmx::MoeReduce", Y, expert_weights, invert_permutation,
                     num_experts_per_token_i=num_experts_per_token)
        return Y_out.setTypeAs(Y)


    @staticmethod
    def forward(self, Y: torch.Tensor, expert_weights: torch.Tensor,
                invert_permutation: torch.Tensor, num_experts_per_token: int):
        # Y: [*, num_experts_per_token, hidden_dim]
        # expert_weights: [*, num_experts_per_token]
        # invert_permutation: [*, num_experts_per_token]
        # Y_out: [*, hidden_dim]

        Y = Y.view(-1, Y.shape[-1])
        Y_out = Y[invert_permutation].view(*expert_weights.shape, -1)
        Y_out = (Y_out * expert_weights.unsqueeze(-1)).sum(dim=-2) # [*, hidden_dim]
        return Y_out


def moe_reduce(Y: torch.Tensor, expert_weights: torch.Tensor,
               invert_permutation: torch.Tensor, num_experts_per_token: int) -> torch.Tensor:

    return MoeReduce.apply(Y, expert_weights, invert_permutation, num_experts_per_token)


if __name__ == "__main__":
    class TestModule(torch.nn.Module):
        def __init__(self, num_experts_per_token: int):
            super().__init__()
            self.num_experts_per_token = num_experts_per_token


        def forward(self, Y: torch.Tensor, expert_weights: torch.Tensor, invert_permutation: torch.Tensor):
            return moe_reduce(Y, expert_weights, invert_permutation, num_experts_per_token)


    num_experts_per_token = 8
    test_op = TestModule(num_experts_per_token)
    Y = torch.randn(5, num_experts_per_token, 4096)
    expert_weights = torch.randn(5, num_experts_per_token)
    invert_permutation = torch.arange(5 * num_experts_per_token)    
    model_str = torch.onnx.export_to_pretty_string(
        test_op, (Y, expert_weights, invert_permutation), "MoeReduce.onnx", opset_version=11)

    print(model_str)

