import torch

from typing import Optional

class KeyValueCache(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, current_key: torch.Value, current_value: torch.Value,
        start_pos: torch.Value, cache: torch.Value, scale: Optional[torch.Value],
        num_layer: int = 1, layer_idx: int = 0, quant_bit: int = 0,
        quant_group: int = 8, num_repeat: int = 1, cache_layout: int = 0):
        if scale is not None:
            key, value = g.op("pmx::KeyValueCache",
                    current_key, current_value,
                    start_pos, cache, scale,
                    num_layer_i = num_layer,
                    layer_idx_i = layer_idx,
                    quant_bit_i = quant_bit,
                    quant_group_i = quant_group,
                    num_repeat_i = num_repeat,
                    cache_layout_i = cache_layout,
                    outputs = 2)
        else:
            key, value = g.op("pmx::KeyValueCache",
                    current_key, current_value,
                    start_pos, cache,
                    num_layer_i = num_layer,
                    layer_idx_i = layer_idx,
                    quant_bit_i = quant_bit,
                    quant_group_i = quant_group,
                    num_repeat_i = num_repeat,
                    cache_layout_i = cache_layout,
                    outputs = 2)
        return key, value


    @staticmethod
    def forward(
        self, current_key: torch.Tensor, current_value: torch.Tensor,
        start_pos: torch.Tensor, cache: torch.Tensor, scale: Optional[torch.Tensor],
        num_layer: int = 1, layer_idx: int = 0, quant_bit: int = 0,
        quant_group: int = 8, num_repeat: int = 1, cache_layout: int = 0):
        if torch.onnx.is_in_onnx_export():
            return current_key, current_value


        def quant(input: torch.Tensor, quant_bit: int, quant_group: int):
            if quant_bit != 8:
                raise Exception("only supporte 8bit quantize")
            X = input.reshape(*input.shape[:-1], -1, quant_group)
            scale, _ = torch.max(torch.abs(X), -1, True)
            scale = scale / torch.tensor([127.0], dtype=input.dtype, device=input.device)
            scale = torch.maximum(scale, torch.tensor([1e-5], dtype=input.dtype, device=input.device))
            output = torch.round(X / scale).type(torch.int8)

            output = output.reshape_as(input)
            scale = scale.reshape(*input.shape[:-1], -1)
            return output, scale


        def dequant(input: torch.Tensor, scale: torch.Tensor,
                    quant_bit: int, quant_group: int):
            if quant_bit != 8:
                raise Exception("only supporte 8bit dequantize")
            X = input.reshape(*input.shape[:-1], -1, quant_group).type_as(scale)
            S = scale.reshape(*input.shape[:-1], -1, 1)
            return (X * S).reshape_as(input)

        assert start_pos.numel() == 1, "start_pos.numel() = {}".format(start_pos.numel())
        # must use for loop to take the value in start_pos
        for position in start_pos:
            bs, seqlen, num_head, head_dim = current_key.shape # [bs, seqlen, num_head, head_dim]
            out_seqlen = position + seqlen
            key = torch.zeros(bs, out_seqlen, num_head, head_dim, dtype=current_key.dtype, device=current_key.device)
            value = torch.zeros(bs, out_seqlen, num_head, head_dim, dtype=current_value.dtype, device=current_value.device)


            def process_layout_0():
                if quant_bit > 0:
                    cache[:bs, layer_idx, 0, position:out_seqlen], \
                    scale[:bs, layer_idx, 0, position:out_seqlen] = \
                        quant(current_key, quant_bit, quant_group)
                    cache[:bs, layer_idx, 1, position:out_seqlen], \
                    scale[:bs, layer_idx, 1, position:out_seqlen] = \
                        quant(current_value, quant_bit, quant_group)
                    key[:] = dequant(
                        cache[:bs, layer_idx, 0, :out_seqlen],
                        scale[:bs, layer_idx, 0, :out_seqlen],
                        quant_bit, quant_group)
                    value[:] = dequant(
                        cache[:bs, layer_idx, 1, :out_seqlen],
                        scale[:bs, layer_idx, 1, :out_seqlen],
                        quant_bit, quant_group)
                else:
                    cache[:bs, layer_idx, 0, position:out_seqlen] = current_key
                    cache[:bs, layer_idx, 1, position:out_seqlen] = current_value
                    key[:] = cache[:bs, layer_idx, 0, :out_seqlen]
                    value[:] = cache[:bs, layer_idx, 1, :out_seqlen]


            def process_layout_1():
                if quant_bit > 0:
                    c, s = quant(current_key, quant_bit, quant_group)
                    cache[layer_idx, :bs, 0, :, position:out_seqlen], \
                    scale[layer_idx, :bs, 0, :, position:out_seqlen] = \
                        c.transpose(-3, -2), s.transpose(-3, -2)
                    c, s = quant(current_value, quant_bit, quant_group)
                    cache[layer_idx, :bs, 1, :, position:out_seqlen], \
                    scale[layer_idx, :bs, 1, :, position:out_seqlen] = \
                        c.transpose(-3, -2), s.transpose(-3, -2)
                    key[:] = dequant(
                        cache[layer_idx, :bs, 0, :, :out_seqlen],
                        scale[layer_idx, :bs, 0, :, :out_seqlen],
                        quant_bit, quant_group).transpose(-3, -2)
                    value[:] = dequant(
                        cache[layer_idx, :bs, 1, :, :out_seqlen],
                        scale[layer_idx, :bs, 1, :, :out_seqlen],
                        quant_bit, quant_group).transpose(-3, -2)
                else:
                    cache[layer_idx, :bs, 0, :, position:out_seqlen] = current_key.transpose(-3, -2)
                    cache[layer_idx, :bs, 1, :, position:out_seqlen] = current_value.transpose(-3, -2)
                    key[:] = cache[layer_idx, :bs, 0, :, :out_seqlen].transpose(-3, -2)
                    value[:] = cache[layer_idx, :bs, 1, :, :out_seqlen].transpose(-3, -2)


            if cache_layout == 0:
                process_layout_0()
            elif cache_layout == 1:
                process_layout_1()
            else:
                raise Exception("invalid cache_layout: {}".format(cache_layout))
        if num_repeat > 1:
            return key[:, :, :, None, :].expand(bs, out_seqlen, num_head, num_repeat, head_dim).reshape(bs, out_seqlen, num_head * num_repeat, head_dim), \
                value[:, :, :, None, :].expand(bs, out_seqlen, num_head, num_repeat, head_dim).reshape(bs, out_seqlen, num_head * num_repeat, head_dim)
        else:
            return key, value


def key_value_cache(
        current_key: torch.Tensor, current_value: torch.Tensor,
        start_pos: torch.Tensor, cache: torch.Tensor, scale: Optional[torch.Tensor],
        num_layer: int = 1, layer_idx: int = 0, quant_bit: int = 0,
        quant_group: int = 8, num_repeat: int = 1, cache_layout: int = 0) -> torch.Tensor:
    return KeyValueCache.apply(current_key, current_value, start_pos, cache, scale,
                        num_layer, layer_idx, quant_bit, quant_group, num_repeat, cache_layout)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self,
            num_layer: int,
            layer_idx: int = 0,
            quant_bit: int = 0,
            quant_group: int = 8) -> None:
            super().__init__()
            self.num_layer = num_layer
            self.layer_idx = layer_idx
            self.quant_bit = quant_bit
            self.quant_group = quant_group


        def forward(self, current_key: torch.Tensor, current_value: torch.Tensor,
            start_pos: torch.Tensor, cache: torch.Tensor, scale: torch.Tensor = None):
            return key_value_cache(
                current_key, current_value, start_pos, cache, scale,
                self.num_layer, self.layer_idx, self.quant_bit, self.quant_group, 1, 1)

    bs = 4
    seqlen = 16
    num_heads = 16
    head_dim = 64

    num_layer = 2
    layer_idx = 1
    quant_group = 8
    quant_bit = 8
    test_op1 = TestModule1(num_layer, layer_idx, quant_bit, quant_group)

    cur_key = torch.ones([bs, seqlen, num_heads, head_dim])
    cur_value = torch.ones([bs, seqlen, num_heads, head_dim])

    cache = torch.zeros([num_layer, bs, 2, num_heads, seqlen * 2, head_dim], dtype=torch.int8)
    scale = torch.zeros([num_layer, bs, 2, num_heads, seqlen * 2, head_dim // quant_group])
    start_pos = torch.tensor([seqlen], dtype=torch.int64)

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (cur_key, cur_value, start_pos, cache, scale), "KeyValueCache1.onnx", opset_version=11)

    print(model_str1)
