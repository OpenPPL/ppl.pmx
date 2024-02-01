import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist


class MoeColumnParallelLinear(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g: torch._C.Graph, X: torch.Value, expert_offset: torch.Value,
        W: torch.Value, B: torch.Value, proc_group: dist.ProcessGroup,
        num_experts: int, in_features: int, out_features: int, gather_output: bool = True):
        if B is not None:
            Y = g.op("pmx::MoeColumnParallelLinear", X, expert_offset, W, B, 
                                num_experts_i = num_experts,
                                in_features_i = in_features,
                                out_features_i = out_features,
                                bias_term_i = True,
                                gather_output_i = gather_output)
        else:
            Y = g.op("pmx::MoeColumnParallelLinear", X, expert_offset, W, 
                                num_experts_i = num_experts,
                                in_features_i = in_features,
                                out_features_i = out_features,
                                bias_term_i = False,
                                gather_output_i = gather_output)
        return Y


    @staticmethod
    def forward(self, X: torch.Tensor, expert_offset: torch.Tensor,
                W: torch.Tensor, B: torch.Tensor, proc_group: dist.ProcessGroup,
                num_experts: int, in_features: int, out_features: int, gather_output: bool = True):
        # X: [*, hidden_dim]
        # expert_offset: [num_experts+1]
        # W: [num_experts, hidden_dim, hidden_dim]
        # B: [num_experts, hidden_dim]
        # Y: [*, hidden_dim]
        
        assert X.shape[-1] == in_features, "X.shape is {}, in_features is {}".format(X.shape, in_features) 
        
        out_dim = W.shape[1]
        
        if torch.onnx.is_in_onnx_export():
            output_parallel = torch.zeros(*X.shape[:-1], out_dim, dtype=W.dtype).to(X.device)

            if gather_output and proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
                last_dim = output_parallel.dim() - 1
                rank = torch.distributed.get_rank(group=proc_group)
                world_size = torch.distributed.get_world_size(group=proc_group)
                tensor_list = [torch.zeros_like(output_parallel) for _ in range(world_size)]
                tensor_list[rank] = output_parallel
                Y = torch.cat(tensor_list, dim=last_dim).contiguous()
            else:
                Y = output_parallel
            return Y
        else:
            X_flat = X.view(-1, X.shape[-1]) # (seqlen * num_experts_per_token, hidden_dim)
            output_parallel = torch.zeros(*X.shape[:-1], out_dim).view(-1, out_dim)        
            
            for i in range(num_experts):
                if expert_offset[i+1] - expert_offset[i] <= 0:
                    continue
                
                output_parallel[expert_offset[i]: expert_offset[i+1]] = (
                    F.linear(X_flat[expert_offset[i]: expert_offset[i+1]], W[i], B[i] if B is not None else None)
                )
                
            output_parallel = output_parallel.view(*X.shape[:-1], out_dim)

            if gather_output and proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
                last_dim = output_parallel.dim() - 1
                rank = torch.distributed.get_rank(group=proc_group)
                world_size = torch.distributed.get_world_size(group=proc_group)
                tensor_list = [torch.empty_like(output_parallel) for _ in range(world_size)]
                tensor_list[rank] = output_parallel
                torch.distributed.all_gather(tensor_list, output_parallel, group=proc_group)
                Y = torch.cat(tensor_list, dim=last_dim).contiguous()
            else:
                Y = output_parallel
        return Y


def moe_column_parallel_linear(
    X: torch.Tensor, expert_offset: torch.Tensor,
    W: torch.Tensor, B: torch.Tensor, proc_group: dist.ProcessGroup,
    num_experts: int, in_features: int,
    out_features: int, gather_output: bool = True) -> torch.Tensor:

    return MoeColumnParallelLinear.apply(
        X, expert_offset, W, B, proc_group,
        num_experts, in_features,
        out_features, gather_output)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self,
            proc_group: dist.ProcessGroup,
            num_experts: int, 
            in_features: int,
            out_features: int,
            bias_term: bool = True,
            gather_output: bool = True) -> None:
            super().__init__()
            
            self.proc_group = proc_group
            self.num_experts = num_experts
            self.in_features = in_features
            self.out_features = out_features
            self.gather_output = gather_output
            
            world_size = 1 if proc_group is None else proc_group.size()

            assert out_features % world_size == 0, "{} is not divisible by {}".format(out_features, world_size)

            self.out_features_per_partition = out_features // world_size

            self.weight = nn.Parameter(torch.ones(self.num_experts, self.out_features_per_partition, self.in_features))
            if bias_term:
                self.bias = nn.Parameter(torch.zeros(self.num_experts, self.out_features_per_partition))
            else:
                self.register_parameter("bias", None)
                
        def forward(self, X: torch.Tensor, expert_offset: torch.Tensor):
            return moe_column_parallel_linear(X, expert_offset, self.weight,
                                              self.bias, self.proc_group,
                                              self.num_experts, self.in_features,
                                              self.out_features, self.gather_output)
    
    num_experts = 8
    test_op1 = TestModule1(None, num_experts, 1024, 4096, True, False)
    
    x = torch.randn(10, num_experts, 1024)
    expert_offset = torch.arange(num_experts + 1)
    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (x, expert_offset), "MoeColumnParallelLinear.onnx", opset_version=11
    )
    print(model_str1)