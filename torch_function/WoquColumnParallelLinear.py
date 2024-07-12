import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from WeightOnlyQuantUtils import Int4QuantUtils


class WoquColumnParallelLinear(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, W: torch.Value, Scale: torch.Value, ZeroPoint: Optional[torch.Value],
        B: Optional[torch.Value], proc_group: torch.Value, quant_data_type: str,
        in_features: int, out_features: int, gather_output: bool = True,
        quant_method: str='', quant_axis: int=1, group_size: int=128,
        has_zeropoint: bool=False, float_zeropoint: bool=False):
        if B is not None:
            Y = g.op("opmx::WoquColumnParallelLinear", X, W, Scale, ZeroPoint, B,
                    quant_data_type_s = quant_data_type,
                    in_features_i = in_features,
                    out_features_i = out_features,
                    bias_term_i = True,
                    gather_output_i = gather_output,
                    quant_method_s = quant_method,
                    quant_axis_i = quant_axis,
                    group_size_i = group_size,
                    has_zeropoint_i = has_zeropoint,
                    float_zeropoint_i = float_zeropoint)
        elif ZeroPoint is not None:
            Y = g.op("opmx::WoquColumnParallelLinear", X, W, Scale, ZeroPoint,
                    quant_data_type_s = quant_data_type,
                    in_features_i = in_features,
                    out_features_i = out_features,
                    bias_term_i = False,
                    gather_output_i = gather_output,
                    quant_method_s = quant_method,
                    quant_axis_i = quant_axis,
                    group_size_i = group_size,
                    has_zeropoint_i = has_zeropoint,
                    float_zeropoint_i = float_zeropoint)
        else:
            Y = g.op("opmx::WoquColumnParallelLinear", X, W, Scale,
                     quant_data_type_s = quant_data_type,
                     in_features_i = in_features,
                     out_features_i = out_features,
                     bias_term_i = False,
                     gather_output_i = gather_output,
                     quant_method_s = quant_method,
                     quant_axis_i = quant_axis,
                     group_size_i = group_size,
                     has_zeropoint_i = has_zeropoint,
                     float_zeropoint_i = False)
        return Y


    @staticmethod
    def forward(
        self, X: torch.Tensor, W: torch.Tensor, Scale: torch.Value, ZeroPoint: Optional[torch.Value],
        B: Optional[torch.Value], proc_group: dist.ProcessGroup, quant_data_type: str,
        in_features: int, out_features: int, gather_output: bool = True, quant_method: str='',
        quant_axis: int=1, group_size: int=128, has_zeropoint: bool=False, float_zeropoint: bool=False):

        if torch.onnx.is_in_onnx_export():
            output_parallel = torch.zeros(*X.shape[:-1], W.shape[0], dtype=W.dtype).to(X.device)
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
            # Matrix multiply.
            assert quant_data_type == 'int4', 'int8 dequantize is not implemented'
            unpacked_int4_w = Int4QuantUtils.unpack(W)
            dequant_fp16_w = Int4QuantUtils.dequantize_int4_to_fp16(unpacked_int4_w, Scale, ZeroPoint, group_size)
            output_parallel = F.linear(X, dequant_fp16_w, B)
            # All-gather across the partitions.
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


def woqu_column_parallel_linear(
    X: torch.Tensor, W: torch.Tensor, Scale: torch.Value, ZeroPoint: Optional[torch.Value],
    B: Optional[torch.Value], proc_group: dist.ProcessGroup, quant_data_type: str, in_features: int,
    out_features: int, gather_output: bool = True, quant_method: str='', quant_axis: int=1,
    group_size: int=128, float_zeropoint: bool=False) -> torch.Tensor:

    if B is not None and ZeroPoint is None:
        _ZeroPoint = torch.empty(0, device=X.device)
        has_zeropoint = False
    elif ZeroPoint is not None:
        _ZeroPoint = ZeroPoint
        has_zeropoint  = True
    else:
        _ZeroPoint = ZeroPoint
        has_zeropoint  = False

    return WoquColumnParallelLinear.apply(X, W, Scale, _ZeroPoint, B, proc_group, quant_data_type,
                                      in_features, out_features, gather_output, quant_method,
                                      quant_axis, group_size, has_zeropoint, float_zeropoint)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self,
            proc_group: dist.ProcessGroup,
            in_features: int,
            out_features: int,
            quant_data_type: str = 'int4',
            quant_method: str = 'weight only',
            quant_axis: int = 1,
            group_size: int = 128,
            storage_bits: int = 16,
            has_zeropoint: bool=False,
            float_zeropoint: bool=False,
            bias_term: bool = True,
            gather_output: bool = True) -> None:
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.gather_output = gather_output
            self.proc_group = proc_group

            self.quant_data_type = quant_data_type
            self.quant_method= quant_method
            self.quant_axis = quant_axis
            self.group_size = group_size
            self.has_zeropoint = has_zeropoint
            self.float_zeropoint = float_zeropoint

            world_size = 1 if proc_group is None else proc_group.size()
            assert out_features % world_size == 0, "{} is not divisible by {}".format(out_features, world_size)

            self.out_features_per_partition = out_features // world_size

            # pack int4 to int16
            if self.quant_data_type == 'int4':
                self.register_buffer('weight', torch.ones( self.out_features_per_partition // (storage_bits // 4), self.in_features, dtype=torch.int16 ))
            elif self.quant_data_type == 'int8' and self.has_zeropoint == False:
                self.weight = self.register_buffer('weight', torch.ones(self.out_features_per_partition, self.in_features, dtype=torch.int8))

            if bias_term:
                self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition, dtype=torch.float16))
            else:
                self.register_parameter("bias", None)

            if self.has_zeropoint:
                self.register_buffer('ZeroPoint', torch.ones(self.out_features_per_partition, self.in_features // self.group_size, dtype=torch.int8))
            else:
                self.register_buffer('ZeroPoint', None)
            self.Scale = nn.Parameter(torch.ones(self.out_features_per_partition, self.in_features // self.group_size, dtype=torch.float16))


        def forward(self, X: torch.Tensor):
            return woqu_column_parallel_linear(
                X, self.weight, self.Scale, self.ZeroPoint, self.bias, self.proc_group, self.quant_data_type,
                self.in_features, self.out_features, self.gather_output, self.quant_method, self.quant_axis,
                self.group_size, self.float_zeropoint)


    test_op1 = TestModule1(None, 512, 2048, has_zeropoint=False, bias_term=True)

    input = torch.ones([8, 512], dtype=torch.float16)
    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "WoquColumnParallelLinear1.onnx", opset_version=11)

    print(model_str1)
    torch.onnx.export(
        test_op1,
        (input),
        "WoquColumnParallelLinear1.onnx",
        input_names=['input'],
        output_names=['out'],
        opset_version=11,)
