import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

if __name__ == "__main__":
    from SwiGLU import swiglu
    from SiLU import silu
    from MoeSelect import moe_select
    from MoeReduce import moe_reduce
else:
    from .SwiGLU import swiglu
    from .SiLU import silu
    from .MoeSelect import moe_select
    from .MoeReduce import moe_reduce


class MoeExpertParallelFeedForward(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, hidden_states: torch.Value, gating_weight: torch.Value,
        expert_up_proj_weight: torch.Value, expert_up_proj_bias: torch.Value,
        expert_down_proj_weight: torch.Value, expert_down_proj_bias: torch.Value,
        shared_up_proj_weight: Optional[torch.Value],
        shared_up_proj_bias: Optional[torch.Value],
        shared_down_proj_weight: Optional[torch.Value],
        shared_down_proj_bias: Optional[torch.Value],
        proc_group: dist.ProcessGroup,
        hidden_dim: int,
        expert_intermediate_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        has_feed_forward_gate: bool = True,
        up_porj_bias_term: bool = False,
        down_proj_bias_term: bool = False,
        has_shared_expert: bool = False,
        shared_intermediate_dim: int = 0,
        num_expert_groups: int = 1,
        num_groups_per_token: int = 1,
        gating_scaling_factor: float = 1.0,
        gating_normalize_prob: bool = False,
        gating_method: str='greedy',
        activation_method: str='silu',
        companion_parallel_mode='tensor_parallel'):
        if has_shared_expert:
            Y = g.op("opmx::MoeExpertParallelFeedForward",
                    hidden_states, gating_weight,
                    expert_up_proj_weight, expert_up_proj_bias,
                    expert_down_proj_weight, expert_down_proj_bias,
                    shared_up_proj_weight, shared_up_proj_bias,
                    shared_down_proj_weight, shared_down_proj_bias,
                    hidden_dim_i=hidden_dim,
                    expert_intermediate_dim_i=expert_intermediate_dim,
                    num_experts_i=num_experts,
                    num_experts_per_token_i=num_experts_per_token,
                    has_feed_forward_gate_i=has_feed_forward_gate,
                    up_porj_bias_term_i=up_porj_bias_term,
                    down_proj_bias_term_i=down_proj_bias_term,
                    has_shared_expert_i=has_shared_expert,
                    shared_intermediate_dim_i=shared_intermediate_dim,
                    num_expert_groups_i=num_expert_groups,
                    num_groups_per_token_i=num_groups_per_token,
                    gating_scaling_factor_f=gating_scaling_factor,
                    gating_normalize_prob_i=gating_normalize_prob,
                    gating_method_s=gating_method,
                    activation_method_s=activation_method,
                    companion_parallel_mode_s=companion_parallel_mode)
        else:
            Y = g.op("opmx::MoeExpertParallelFeedForward",
                    hidden_states, gating_weight,
                    expert_up_proj_weight, expert_up_proj_bias,
                    expert_down_proj_weight, expert_down_proj_bias,
                    hidden_dim_i=hidden_dim,
                    expert_intermediate_dim_i=expert_intermediate_dim,
                    num_experts_i=num_experts,
                    num_experts_per_token_i=num_experts_per_token,
                    has_feed_forward_gate_i=has_feed_forward_gate,
                    up_porj_bias_term_i=up_porj_bias_term,
                    down_proj_bias_term_i=down_proj_bias_term,
                    has_shared_expert_i=has_shared_expert,
                    shared_intermediate_dim_i=shared_intermediate_dim,
                    num_expert_groups_i=num_expert_groups,
                    num_groups_per_token_i=num_groups_per_token,
                    gating_scaling_factor_f=gating_scaling_factor,
                    gating_normalize_prob_i=gating_normalize_prob,
                    gating_method_s=gating_method,
                    activation_method_s=activation_method,
                    companion_parallel_mode_s=companion_parallel_mode)
        return Y.typeAs(hidden_states)


    @staticmethod
    def forward(
        self, hidden_states: torch.Tensor, route_weight: torch.Tensor,
        expert_up_proj_weight: torch.Tensor, expert_up_proj_bias: torch.Tensor,
        expert_down_proj_weight: torch.Tensor, expert_down_proj_bias: torch.Tensor,
        shared_up_proj_weight: Optional[torch.Tensor],
        shared_up_proj_bias: Optional[torch.Tensor],
        shared_down_proj_weight: Optional[torch.Tensor],
        shared_down_proj_bias: Optional[torch.Tensor],
        proc_group: dist.ProcessGroup,
        hidden_dim: int,
        expert_intermediate_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        has_feed_forward_gate: bool = True,
        up_porj_bias_term: bool = False,
        down_proj_bias_term: bool = False,
        has_shared_expert: bool = False,
        shared_intermediate_dim: int = 0,
        num_expert_groups: int = 1,
        num_groups_per_token: int = 1,
        gating_scaling_factor: float = 1.0,
        gating_normalize_prob: bool = False,
        gating_method: str='greedy',
        activation_method: str='silu',
        companion_parallel_mode='tensor_parallel'):
        if torch.onnx.is_in_onnx_export():
            hidden_states

        assert 'tensor_parallel' in companion_parallel_mode and len('tensor_parallel') == len(companion_parallel_mode)
        assert 'silu' in activation_method and len('silu') == len(activation_method)

        act_fns = {
            'silu': silu
        }
        gated_act_fns = {
            'silu': swiglu
        }
        act_fn = gated_act_fns[activation_method] if has_feed_forward_gate else act_fns[activation_method]

        world_size = 1
        local_rank = 0
        if proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
            world_size = torch.distributed.get_world_size(proc_group)
            local_rank = torch.distributed.get_rank(proc_group)

        assert num_experts % world_size == 0, "{} is not divisible by {}".format(num_experts, world_size)
        num_local_experts = num_experts // world_size
        local_expert_offset = local_rank * num_local_experts

        logits = F.linear(hidden_states.view(-1, hidden_dim), route_weight).view(*hidden_states.shape[:-1], num_experts)

        if 'tensor_parallel' in companion_parallel_mode and len('tensor_parallel') == len(companion_parallel_mode):
            # TP+EP方案，shared expert走TP，moe expert走EP
            if has_shared_expert: # do tp for shared expert
                assert shared_intermediate_dim % world_size == 0, "{} is not divisible by {}".format(shared_intermediate_dim, world_size)
                output_parallel = F.linear(
                    hidden_states.view(-1, hidden_dim),
                    shared_up_proj_weight,
                    shared_up_proj_bias if up_porj_bias_term else None)
                output_parallel = act_fn(output_parallel)
                output_parallel = F.linear(
                    output_parallel,
                    shared_down_proj_weight,
                    shared_down_proj_bias if down_proj_bias_term and local_rank == 0 else None)
                output_parallel.view_as(hidden_states)
            else:
                output_parallel = torch.zeros_like(hidden_states) # (b, s, hid)

            # 为了开BUFFER方便，中间tensor基本都是为所有token分配
            # 重排所有的
            # do ep for normal experts
            # (b, s, es, hid), (b, s, es), (b, s, es), (e + 1)
            states_expand_permute, expert_weights, invert_permutation, expert_offset = moe_select(
                hidden_states, logits, num_experts, num_experts_per_token,
                num_expert_groups, num_groups_per_token,
                gating_scaling_factor, gating_normalize_prob,
                gating_method
            )

            # 分配所有token的空间，但只计算本地expert的部分
            flat_stats = states_expand_permute.view(-1, hidden_dim) # (b * s * es, hid)
            up_proj_output = torch.zeros(flat_stats.shape[0], expert_intermediate_dim) # (b * s * es, e_imm)
            # here will be grouped gemm
            for local_expert_idx in range(num_local_experts):
                expert_idx = local_expert_idx + local_expert_offset
                expert_beg = expert_offset[expert_idx]
                expert_end = expert_offset[expert_idx + 1]
                if expert_end - expert_beg <= 0:
                    continue
                up_proj_output[expert_beg:expert_end] = (
                    F.linear(flat_stats[expert_beg:expert_end],
                             expert_up_proj_weight[local_expert_idx],
                             expert_up_proj_bias[local_expert_idx] if up_porj_bias_term else None)
                )
            up_proj_output = swiglu(up_proj_output) if has_feed_forward_gate else silu(up_proj_output)
            # 非本地expert的token直接置零，实现时可以将置零融合到MOE REDUCE里面
            down_proj_output = states_expand_permute.zero_().view(-1, hidden_dim)
            # here will be grouped gemm
            for local_expert_idx in range(num_local_experts):
                expert_idx = local_expert_idx + local_expert_offset
                expert_beg = expert_offset[expert_idx]
                expert_end = expert_offset[expert_idx + 1]
                if expert_end - expert_beg <= 0:
                    continue
                down_proj_output[expert_beg:expert_end] = (
                    F.linear(up_proj_output[expert_beg:expert_end],
                             expert_down_proj_weight[local_expert_idx],
                             expert_down_proj_bias[local_expert_idx] if down_proj_bias_term else None)
                )
            down_proj_output = down_proj_output.view(*hidden_states.shape[:-1], num_experts_per_token, hidden_dim) # (b, s, es, hid)
            # we can only reduce local expert's weight in cuda
            output_parallel += moe_reduce(down_proj_output, expert_weights, invert_permutation, num_experts_per_token) # (b, s, hid)

            if world_size > 1:
                torch.distributed.all_reduce(output_parallel, group=proc_group)

        return output_parallel


def moe_expert_parallel_feed_forward(
        hidden_states: torch.Tensor, gating_weight: torch.Tensor,
        expert_up_proj_weight: torch.Tensor, expert_up_proj_bias: torch.Tensor,
        expert_down_proj_weight: torch.Tensor, expert_down_proj_bias: torch.Tensor,
        shared_up_proj_weight: Optional[torch.Tensor],
        shared_up_proj_bias: Optional[torch.Tensor],
        shared_down_proj_weight: Optional[torch.Tensor],
        shared_down_proj_bias: Optional[torch.Tensor],
        proc_group: dist.ProcessGroup,
        hidden_dim: int,
        expert_intermediate_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        has_feed_forward_gate: bool = True,
        up_porj_bias_term: bool = False,
        down_proj_bias_term: bool = False,
        has_shared_expert: bool = False,
        shared_intermediate_dim: int = 0,
        num_expert_groups: int = 1,
        num_groups_per_token: int = 1,
        gating_scaling_factor: float = 1.0,
        gating_normalize_prob: bool = False,
        gating_method: str='greedy',
        activation_method: str='silu',
        companion_parallel_mode='tensor_parallel') -> torch.Tensor:
    return MoeExpertParallelFeedForward.apply(
        hidden_states, gating_weight,
        expert_up_proj_weight, expert_up_proj_bias,
        expert_down_proj_weight, expert_down_proj_bias,
        shared_up_proj_weight, shared_up_proj_bias,
        shared_down_proj_weight, shared_down_proj_bias,
        proc_group,
        hidden_dim,
        expert_intermediate_dim,
        num_experts,
        num_experts_per_token,
        has_feed_forward_gate,
        up_porj_bias_term,
        down_proj_bias_term,
        has_shared_expert,
        shared_intermediate_dim,
        num_expert_groups,
        num_groups_per_token,
        gating_scaling_factor,
        gating_normalize_prob,
        gating_method,
        activation_method,
        companion_parallel_mode
    )


# if __name__ == "__main__":
#     class TestModule1(torch.nn.Module):
#         def __init__(
#             self,
#             proc_group: dist.ProcessGroup,
#             in_features: int,
#             out_features: int,
#             bias_term: bool = True,
#             gather_output: bool = True) -> None:
#             super().__init__()

#             self.in_features = in_features
#             self.out_features = out_features
#             self.gather_output = gather_output
#             self.proc_group = proc_group

#             world_size = 1 if proc_group is None else proc_group.size()
#             assert out_features % world_size == 0, "{} is not divisible by {}".format(out_features, world_size)

#             self.out_features_per_partition = out_features // world_size

#             self.weight = nn.Parameter(torch.ones(self.out_features_per_partition, self.in_features))
#             if bias_term:
#                 self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition))
#             else:
#                 self.register_parameter("bias", None)


#         def forward(self, X: torch.Tensor):
#             return column_parallel_linear(
#                 X, self.weight, self.bias, self.proc_group,
#                 self.in_features, self.out_features)


#     test_op1 = TestModule1(None, 1024, 4096, True, False)

#     input = torch.ones([8, 1024])

#     model_str1 = torch.onnx.export_to_pretty_string(
#         test_op1, (input), "ColumnParallelLinear1.onnx", opset_version=11)

#     print(model_str1)
