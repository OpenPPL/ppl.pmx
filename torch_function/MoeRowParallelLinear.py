import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist


class MoeRowParallelLinear(torch.autograd.Function):
    @staticmethod
    def symbolic(g: torch._C.Graph, X: torch.Value, expert_offset: torch.Value, 
        W: torch.Value, B: torch.Value, proc_group: dist.ProcessGroup, num_experts: int, 
        in_features: int, out_features: int, input_is_parallel: bool = False):
        if B is not None:
            Y = g.op("pmx::MoeRowParallelLinear", X, expert_offset, W, B,
                     num_experts_i = num_experts,
                     in_features_i = in_features,
                     out_features_i = out_features,
                     bias_term_i = True,
                     input_is_parallel_i = input_is_parallel)
        else:
            Y = g.op("pmx::MoeRowParallelLinear", X, expert_offset, W,
                     num_experts_i = num_experts,
                     in_features_i = in_features,
                     out_features_i = out_features,
                     bias_term_i = False,
                     input_is_parallel_i = input_is_parallel)
        return Y


    @staticmethod
    def forward(self, X: torch.Tensor, expert_offset: torch.Tensor,
                W: torch.Tensor, B: torch.Tensor, proc_group: dist.ProcessGroup,
                num_experts: int, in_features: int, out_features: int,
                input_is_parallel: bool = False):
        # X: [*, hidden_dim]
        # expert_offset: [num_experts+1]
        # W: [num_experts, hidden_dim, hidden_dim]
        # B: [num_experts, hidden_dim]
        # output_parallel: [*, hidden_dim]
        
        out_dim = W.shape[1]
        if torch.onnx.is_in_onnx_export():
            output_parallel = torch.zeros(*X.shape[:-1], out_dim, dtype=W.dtype).to(X.device)
        else:
            X_flat = X.view(-1, X.shape[-1]) # (seqlen * num_experts_per_token, hidden_dim)
            output_parallel = torch.zeros(*X.shape[:-1], out_dim).view(-1, out_dim)        
            
            for i in range(num_experts):
                if expert_offset[i+1] - expert_offset[i] <= 0:
                    continue
                
                output_parallel[expert_offset[i]: expert_offset[i+1]] = (
                    F.linear(X_flat[expert_offset[i]: expert_offset[i+1]], W[i])
                )
                
            output_parallel = output_parallel.view(*X.shape[:-1], out_dim)

            if proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
                torch.distributed.all_reduce(output_parallel, group=proc_group)
            if B is not None:
                output_parallel = output_parallel + B

        return output_parallel


def moe_row_parallel_linear(X: torch.Tensor, expert_offset: torch.Tensor,
                            W: torch.Tensor, B: torch.Tensor,
                            proc_group: dist.ProcessGroup, num_experts: int,
                            in_features: int, out_features: int,
                            input_is_parallel: bool = False):

    return MoeRowParallelLinear.apply(X, expert_offset, W, B, proc_group,
                                      num_experts, in_features, out_features,
                                      input_is_parallel)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self, 
            proc_group: dist.ProcessGroup,
            num_experts: int,
            in_features: int,
            out_features: int,
            bias_term: bool = True,
            input_is_parallel: bool = False) -> None:
            super().__init__()
            
            self.proc_group = proc_group
            self.num_experts = num_experts
            self.in_features = in_features
            self.out_features = out_features
            self.input_is_parallel = input_is_parallel

            world_size = 1 if proc_group is None else proc_group.size()
            assert in_features % world_size == 0, "{} is not divisible by {}".format(in_features, world_size)

            self.in_features_per_partition = in_features // world_size
            
            self.weight = nn.Parameter(torch.ones(self.num_experts, self.out_features, self.in_features_per_partition))
            if bias_term:
                self.bias = nn.Parameter(torch.zeros(self.num_experts, self.out_features))
            else:
                self.register_parameter("bias", None)


        def forward(self, X: torch.Tensor, expert_offset: torch.Tensor):
            return moe_row_parallel_linear(X, expert_offset, self.weight, self.bias, self.proc_group, self.num_experts, self.in_features, self.out_features, self.input_is_parallel)


    num_experts = 8
    test_op1 = TestModule1(None, num_experts, 1024, 4096, True, True)

    x = torch.randn(10, num_experts, 1024)
    expert_offset = torch.arange(num_experts + 1)
    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (x, expert_offset), "MoeRowParallelLinear.onnx", opset_version=11
    )
    print(model_str1)
