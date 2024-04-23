import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class RowParallelLinear(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, W: torch.Value, B: torch.Value, proc_group: torch.Value,
        in_features: int, out_features: int, input_is_parallel: bool = False):
        if B is not None:
            Y = g.op("opmx::RowParallelLinear", X, W, B,
                    in_features_i = in_features,
                    out_features_i = out_features,
                    bias_term_i = True,
                    input_is_parallel_i = input_is_parallel)
        else:
            Y = g.op("opmx::RowParallelLinear", X, W,
                    in_features_i = in_features,
                    out_features_i = out_features,
                    bias_term_i = False,
                    input_is_parallel_i = input_is_parallel)
        return Y


    @staticmethod
    def forward(
        self, X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, proc_group: dist.ProcessGroup,
        in_features: int, out_features: int, input_is_parallel: bool = False):
        if input_is_parallel:
            input_parallel = X
        else:
            raise Exception("scatter input has not implement yet")
        if torch.onnx.is_in_onnx_export():
            output_parallel = torch.zeros(*X.shape[:-1], W.shape[0], dtype=W.dtype).to(X.device)
        else:
            # Matrix multiply.
            output_parallel = F.linear(input_parallel, W)
            # All-reduce across all the partitions.
            if proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
                torch.distributed.all_reduce(output_parallel, group=proc_group)
            if B is not None:
                output_parallel = output_parallel + B

        return output_parallel


def row_parallel_linear(
        X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, proc_group: dist.ProcessGroup,
        in_features: int, out_features: int, input_is_parallel: bool = False) -> torch.Tensor:
    return RowParallelLinear.apply(X, W, B, proc_group, in_features, out_features, input_is_parallel)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self,
            proc_group: dist.ProcessGroup,
            in_features: int,
            out_features: int,
            bias_term: bool = True,
            input_is_parallel: bool = False) -> None:
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.input_is_parallel = input_is_parallel
            self.proc_group = proc_group

            world_size = 1 if proc_group is None else proc_group.size()
            assert in_features % world_size == 0, "{} is not divisible by {}".format(in_features, world_size)

            self.in_features_per_partition = in_features // world_size

            self.weight = nn.Parameter(torch.ones(self.out_features, self.in_features_per_partition))
            if bias_term:
                self.bias = nn.Parameter(torch.zeros(self.out_features))
            else:
                self.register_parameter("bias", None)


        def forward(self, X: torch.Tensor):
            return row_parallel_linear(
                X, self.weight, self.bias, self.proc_group,
                self.in_features, self.out_features, self.input_is_parallel)


    test_op1 = TestModule1(None, 1024, 4096, True, True)

    input = torch.ones([8, 1024])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "RowParallelLinear1.onnx", opset_version=11)

    print(model_str1)
