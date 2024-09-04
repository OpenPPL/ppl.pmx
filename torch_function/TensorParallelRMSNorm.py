import torch
import torch.distributed as dist

class TensorParallelRMSNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, weight: torch.Value, proc_group: torch.Value,
        axis: int = -1, eps: float = 1e-5,
        scale: float = 1.0, input_is_parallel: bool = False):
        Y = g.op("opmx::TensorParallelRMSNorm", X, weight,
                 axis_i = axis,
                 eps_f = eps,
                 scale_f = scale,
                 input_is_parallel_i = input_is_parallel,
                 )
        return Y.setTypeAs(X)


    @staticmethod
    def forward(
        self, X: torch.Tensor, weight: torch.Tensor, proc_group: dist.ProcessGroup,
        axis: int = -1, eps: float = 1e-5, scale: float = 1.0, input_is_parallel: bool = False):

        if torch.onnx.is_in_onnx_export():
            return X

        if input_is_parallel:
            x_parallel = X.float()
        else:
            raise Exception("scatter input has not implement yet")

        tp_variance = x_parallel.pow(2).mean(axis, keepdim=True)
        world_size = 1 if proc_group is None else proc_group.size()

        if proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
            dist.all_reduce(tp_variance, op=dist.ReduceOp.SUM, async_op=False)
        variance = tp_variance / (world_size * scale) # scale for pad head
        Y = x_parallel * torch.rsqrt(variance + eps)
        return Y.type_as(X) * weight



def tensor_parallel_rms_norm(X: torch.Tensor, weight: torch.Tensor, proc_group: dist.ProcessGroup,
                             axis: int = -1, eps: float = 1e-5, scale: float = 1.0,
                             input_is_parallel: bool = False) -> torch.Tensor:
    return TensorParallelRMSNorm.apply(X, weight, proc_group, axis, eps, scale, input_is_parallel)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, proc_group: dist.ProcessGroup, dim: int, eps: float = 1e-5,
                     scale: float = 1.0, input_is_parallel: bool = False) -> None:
            super().__init__()
            self.eps = eps
            self.embed_dim = dim
            self.scale = scale
            self.input_is_parallel = input_is_parallel
            self.proc_group = proc_group

            world_size = 1 if proc_group is None else proc_group.size()
            assert dim % world_size == 0, "{} is not divisible by {}".format(dim, world_size)
            self.dim_per_partition = dim // world_size
            self.weight = torch.nn.Parameter(torch.ones(self.dim_per_partition))

        def forward(self, X: torch.Tensor):
            return tensor_parallel_rms_norm(X, self.weight, self.proc_group, -1, self.eps, self.scale, self.input_is_parallel)



    test_op1 = TestModule1(None, 4096, 1e-6, 1., True)

    input = torch.ones([8, 4096])


    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "TensorParallelRMSNorm1.onnx", opset_version=11)

    print(model_str1)
