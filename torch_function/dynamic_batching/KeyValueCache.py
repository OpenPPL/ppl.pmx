import torch

from typing import Optional


class KeyValueCache(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, current_key: torch.Value, current_value: torch.Value,
        seqstarts: torch.Value, kvstarts: torch.Value,
        cachestarts: torch.Value, start_pos: torch.Value,
        max_seqlen: torch.Value, max_kvlen: torch.Value,
        cache: torch.Value, scale: Optional[torch.Value],
        num_layer: int = 1, layer_idx: int = 0, quant_bit: int = 0,
        quant_group: int = 8, num_repeat: int = 1,
        cache_mode: int = 0, cache_layout: int = 0):
        if scale is not None:
             key, value = g.op("pmx.dynamic_batching::KeyValueCache",
                    current_key, current_value, seqstarts, kvstarts,
                    cachestarts, start_pos, max_seqlen, max_kvlen, cache, scale,
                    num_layer_i = num_layer,
                    layer_idx_i = layer_idx,
                    quant_bit_i = quant_bit,
                    quant_group_i = quant_group,
                    num_repeat_i = num_repeat,
                    cache_mode_i = cache_mode,
                    cache_layout_i = cache_layout,
                    outputs = 2)
        else:
            key, value = g.op("pmx.dynamic_batching::KeyValueCache",
                    current_key, current_value, seqstarts, kvstarts,
                    cachestarts, start_pos, max_seqlen, max_kvlen, cache,
                    num_layer_i = num_layer,
                    layer_idx_i = layer_idx,
                    quant_bit_i = quant_bit,
                    quant_group_i = quant_group,
                    num_repeat_i = num_repeat,
                    cache_mode_i = cache_mode,
                    cache_layout_i = cache_layout,
                    outputs = 2)
        return key, value


    @staticmethod
    def forward(
        self, current_key: torch.Tensor, current_value: torch.Tensor,
        seqstarts: torch.Tensor, kvstarts: torch.Tensor,
        cachestarts: torch.Tensor, start_pos: torch.Tensor,
        max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
        cache: torch.Tensor, scale: Optional[torch.Tensor],
        num_layer: int = 1, layer_idx: int = 0, quant_bit: int = 0,
        quant_group: int = 8, num_repeat: int = 1,
        cache_mode: int = 0, cache_layout: int = 0):
        if torch.onnx.is_in_onnx_export():
            return current_key, current_value


        def quant(input: torch.Tensor, quant_bit: int, quant_group: int):
            if quant_bit != 8:
                raise Exception("only supporte 8bit quantize")
            X = input.reshape(*input.shape[:-1], -1, quant_group)
            scale, _ = torch.max(torch.abs(X), -1, True)
            scale = scale / torch.tensor([127.0]).type_as(input)
            scale = torch.maximum(scale, torch.tensor([1e-5]).type_as(input))
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


        _, num_head, head_dim = current_key.shape
        key = torch.zeros(kvstarts[-1], num_head, head_dim, dtype=current_key.dtype, device=current_key.device)
        value = torch.zeros(kvstarts[-1], num_head, head_dim, dtype=current_value.dtype, device=current_value.device)

        seqlens = seqstarts[1:] - seqstarts[:-1]
        for b, seqlen in enumerate(seqlens):
            position = start_pos[b]
            seqbeg = seqstarts[b]
            seqend = seqstarts[b+1]
            kvbeg = kvstarts[b]
            kvend = kvstarts[b+1]
            if cache_mode == 0:
                storebeg = cachestarts[b] + position
                storeend = cachestarts[b] + position + seqlen
                loadbeg = cachestarts[b]
                loadend = storeend
            elif cache_mode == 1:
                storeidx = cachestarts[kvbeg + position:kvend]
                loadidx = cachestarts[kvbeg:kvend]
            else:
                raise Exception("invalid cache_mode: {}".format(cache_mode))


            def process_mode_0_layout_0():
                if quant_bit > 0:
                    cache[storebeg:storeend, layer_idx, 0], \
                    scale[storebeg:storeend, layer_idx, 0] =  \
                        quant(current_key[seqbeg:seqend], quant_bit, quant_group)
                    cache[storebeg:storeend, layer_idx, 1], \
                    scale[storebeg:storeend, layer_idx, 1] =  \
                        quant(current_value[seqbeg:seqend], quant_bit, quant_group)
                    key[kvbeg:kvend] = dequant(
                        cache[loadbeg:loadend, layer_idx, 0],
                        scale[loadbeg:loadend, layer_idx, 0],
                        quant_bit, quant_group)
                    value[kvbeg:kvend] = dequant(
                        cache[loadbeg:loadend, layer_idx, 1],
                        scale[loadbeg:loadend, layer_idx, 1],
                        quant_bit, quant_group)
                else:
                    cache[storebeg:storeend, layer_idx, 0] = current_key[seqbeg:seqend]
                    cache[storebeg:storeend, layer_idx, 1] = current_value[seqbeg:seqend]
                    key[kvbeg:kvend] = cache[loadbeg:loadend, layer_idx, 0]
                    value[kvbeg:kvend] = cache[loadbeg:loadend, layer_idx, 1]


            def process_mode_0_layout_1():
                if quant_bit > 0:
                    cache[layer_idx, storebeg:storeend, 0], \
                    scale[layer_idx, storebeg:storeend, 0] =  \
                        quant(current_key[seqbeg:seqend], quant_bit, quant_group)
                    cache[layer_idx, storebeg:storeend, 1], \
                    scale[layer_idx, storebeg:storeend, 1] =  \
                        quant(current_value[seqbeg:seqend], quant_bit, quant_group)
                    key[kvbeg:kvend] = dequant(
                        cache[layer_idx, loadbeg:loadend, 0],
                        scale[layer_idx, loadbeg:loadend, 0],
                        quant_bit, quant_group)
                    value[kvbeg:kvend] = dequant(
                        cache[layer_idx, loadbeg:loadend, 1],
                        scale[layer_idx, loadbeg:loadend, 1],
                        quant_bit, quant_group)
                else:
                    cache[layer_idx, storebeg:storeend, 0] = current_key[seqbeg:seqend]
                    cache[layer_idx, storebeg:storeend, 1] = current_value[seqbeg:seqend]
                    key[kvbeg:kvend] = cache[layer_idx, loadbeg:loadend, 0]
                    value[kvbeg:kvend] = cache[layer_idx, loadbeg:loadend, 1]


            def process_mode_0_layout_2():
                if quant_bit > 0:
                    cache[layer_idx, 0, storebeg:storeend], \
                    scale[layer_idx, 0, storebeg:storeend] =  \
                        quant(current_key[seqbeg:seqend], quant_bit, quant_group)
                    cache[layer_idx, 1, storebeg:storeend], \
                    scale[layer_idx, 1, storebeg:storeend] =  \
                        quant(current_value[seqbeg:seqend], quant_bit, quant_group)
                    key[kvbeg:kvend] = dequant(
                        cache[layer_idx, 0, loadbeg:loadend],
                        scale[layer_idx, 0, loadbeg:loadend],
                        quant_bit, quant_group)
                    value[kvbeg:kvend] = dequant(
                        cache[layer_idx, 1, loadbeg:loadend],
                        scale[layer_idx, 1, loadbeg:loadend],
                        quant_bit, quant_group)
                else:
                    cache[layer_idx, 0, storebeg:storeend] = current_key[seqbeg:seqend]
                    cache[layer_idx, 1, storebeg:storeend] = current_value[seqbeg:seqend]
                    key[kvbeg:kvend] = cache[layer_idx, 0, loadbeg:loadend]
                    value[kvbeg:kvend] = cache[layer_idx, 1, loadbeg:loadend]


            def process_mode_0_layout_3():
                if quant_bit > 0:
                    c, s = quant(current_key[seqbeg:seqend], quant_bit, quant_group)
                    cache[layer_idx, 0, :, storebeg:storeend], \
                    scale[layer_idx, 0, :, storebeg:storeend] =  \
                        c.transpose(-3, -2), s.transpose(-3, -2)
                    c, s = quant(current_value[seqbeg:seqend], quant_bit, quant_group)
                    cache[layer_idx, 1, :, storebeg:storeend], \
                    scale[layer_idx, 1, :, storebeg:storeend] =  \
                        c.transpose(-3, -2), s.transpose(-3, -2)
                    key[kvbeg:kvend] = dequant(
                        cache[layer_idx, 0, :, loadbeg:loadend],
                        scale[layer_idx, 0, :, loadbeg:loadend],
                        quant_bit, quant_group).transpose(-3, -2)
                    value[kvbeg:kvend] = dequant(
                        cache[layer_idx, 1, :, loadbeg:loadend],
                        scale[layer_idx, 1, :, loadbeg:loadend],
                        quant_bit, quant_group).transpose(-3, -2)
                else:
                    cache[layer_idx, 0, :, storebeg:storeend] = current_key[seqbeg:seqend].transpose(-3, -2)
                    cache[layer_idx, 1, :, storebeg:storeend] = current_value[seqbeg:seqend].transpose(-3, -2)
                    key[kvbeg:kvend] = cache[layer_idx, 0, :, loadbeg:loadend].transpose(-3, -2)
                    value[kvbeg:kvend] = cache[layer_idx, 1, :, loadbeg:loadend].transpose(-3, -2)


            def process_mode_1_layout_0():
                if quant_bit > 0:
                    cache[storeidx, layer_idx, 0], \
                    scale[storeidx, layer_idx, 0] =  \
                        quant(current_key[seqbeg:seqend], quant_bit, quant_group)
                    cache[storeidx, layer_idx, 1], \
                    scale[storeidx, layer_idx, 1] =  \
                        quant(current_value[seqbeg:seqend], quant_bit, quant_group)
                    key[kvbeg:kvend] = dequant(
                        cache[loadidx, layer_idx, 0],
                        scale[loadidx, layer_idx, 0],
                        quant_bit, quant_group)
                    value[kvbeg:kvend] = dequant(
                        cache[loadidx, layer_idx, 1],
                        scale[loadidx, layer_idx, 1],
                        quant_bit, quant_group)
                else:
                    cache[storeidx, layer_idx, 0] = current_key[seqbeg:seqend]
                    cache[storeidx, layer_idx, 1] = current_value[seqbeg:seqend]
                    key[kvbeg:kvend] = cache[loadidx, layer_idx, 0]
                    value[kvbeg:kvend] = cache[loadidx, layer_idx, 1]


            def process_mode_1_layout_1():
                if quant_bit > 0:
                    cache[layer_idx, storeidx, 0], \
                    scale[layer_idx, storeidx, 0] =  \
                        quant(current_key[seqbeg:seqend], quant_bit, quant_group)
                    cache[layer_idx, storeidx, 1], \
                    scale[layer_idx, storeidx, 1] =  \
                        quant(current_value[seqbeg:seqend], quant_bit, quant_group)
                    key[kvbeg:kvend] = dequant(
                        cache[layer_idx, loadidx, 0],
                        scale[layer_idx, loadidx, 0],
                        quant_bit, quant_group)
                    value[kvbeg:kvend] = dequant(
                        cache[layer_idx, loadidx, 1],
                        scale[layer_idx, loadidx, 1],
                        quant_bit, quant_group)
                else:
                    cache[layer_idx, storeidx, 0] = current_key[seqbeg:seqend]
                    cache[layer_idx, storeidx, 1] = current_value[seqbeg:seqend]
                    key[kvbeg:kvend] = cache[layer_idx, loadidx, 0]
                    value[kvbeg:kvend] = cache[layer_idx, loadidx, 1]


            def process_mode_1_layout_2():
                if quant_bit > 0:
                    cache[layer_idx, 0, storeidx], \
                    scale[layer_idx, 0, storeidx] =  \
                        quant(current_key[seqbeg:seqend], quant_bit, quant_group)
                    cache[layer_idx, 1, storeidx], \
                    scale[layer_idx, 1, storeidx] =  \
                        quant(current_value[seqbeg:seqend], quant_bit, quant_group)
                    key[kvbeg:kvend] = dequant(
                        cache[layer_idx, 0, loadidx],
                        scale[layer_idx, 0, loadidx],
                        quant_bit, quant_group)
                    value[kvbeg:kvend] = dequant(
                        cache[layer_idx, 1, loadidx],
                        scale[layer_idx, 1, loadidx],
                        quant_bit, quant_group)
                else:
                    cache[layer_idx, 0, storeidx] = current_key[seqbeg:seqend]
                    cache[layer_idx, 1, storeidx] = current_value[seqbeg:seqend]
                    key[kvbeg:kvend] = cache[layer_idx, 0, loadidx]
                    value[kvbeg:kvend] = cache[layer_idx, 1, loadidx]


            def process_mode_1_layout_3():
                if quant_bit > 0:
                    c, s = quant(current_key[seqbeg:seqend], quant_bit, quant_group)
                    cache[layer_idx, 0, :, storeidx], \
                    scale[layer_idx, 0, :, storeidx] =  \
                        c.transpose(-3, -2), s.transpose(-3, -2)
                    c, s = quant(current_value[seqbeg:seqend], quant_bit, quant_group)
                    cache[layer_idx, 1, :, storeidx], \
                    scale[layer_idx, 1, :, storeidx] =  \
                        c.transpose(-3, -2), s.transpose(-3, -2)
                    key[kvbeg:kvend] = dequant(
                        cache[layer_idx, 0, :, loadidx],
                        scale[layer_idx, 0, :, loadidx],
                        quant_bit, quant_group).transpose(-3, -2)
                    value[kvbeg:kvend] = dequant(
                        cache[layer_idx, 1, :, loadidx],
                        scale[layer_idx, 1, :, loadidx],
                        quant_bit, quant_group).transpose(-3, -2)
                else:
                    cache[layer_idx, 0, :, storeidx] = current_key[seqbeg:seqend].transpose(-3, -2)
                    cache[layer_idx, 1, :, storeidx] = current_value[seqbeg:seqend].transpose(-3, -2)
                    key[kvbeg:kvend] = cache[layer_idx, 0, :, loadidx].transpose(-3, -2)
                    value[kvbeg:kvend] = cache[layer_idx, 1, :, loadidx].transpose(-3, -2)


            if cache_mode == 0:
                if cache_layout == 0:
                    process_mode_0_layout_0()
                elif cache_layout == 1:
                    process_mode_0_layout_1()
                elif cache_layout == 2:
                    process_mode_0_layout_2()
                elif cache_layout == 3:
                    process_mode_0_layout_3()
                else:
                    raise Exception("invalid cache_layout: {}".format(cache_layout))
            elif cache_mode == 1:
                if cache_layout == 0:
                    process_mode_1_layout_0()
                elif cache_layout == 1:
                    process_mode_1_layout_1()
                elif cache_layout == 2:
                    process_mode_1_layout_2()
                elif cache_layout == 3:
                    process_mode_1_layout_3()
                else:
                    raise Exception("invalid cache_layout: {}".format(cache_layout))
            else:
                raise Exception("invalid cache_mode: {}".format(cache_mode))
        if num_repeat > 1:
            return key[:, :, None, :].expand(kvstarts[-1], num_head, num_repeat, head_dim).reshape(kvstarts[-1], num_head * num_repeat, head_dim), \
                value[:, :, None, :].expand(kvstarts[-1], num_head, num_repeat, head_dim).reshape(kvstarts[-1], num_head * num_repeat, head_dim)
        else:
            return key, value


def key_value_cache(
        current_key: torch.Tensor, current_value: torch.Tensor,
        seqstarts: torch.Tensor, kvstarts: torch.Tensor,
        cachestarts: torch.Tensor, start_pos: torch.Tensor,
        max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
        cache: torch.Tensor, scale: Optional[torch.Tensor],
        num_layer: int = 1, layer_idx: int = 0, quant_bit: int = 0,
        quant_group: int = 8, num_repeat: int = 1,
        cache_mode: int = 0, cache_layout: int = 0) -> torch.Tensor:
    return KeyValueCache.apply(current_key, current_value, seqstarts, kvstarts, cachestarts, 
                               start_pos, max_seqlen, max_kvlen, cache, scale, num_layer, layer_idx,
                               quant_bit, quant_group, num_repeat, cache_mode, cache_layout)


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
                    seqstarts: torch.Tensor, kvstarts: torch.Tensor,
                    cachestarts: torch.Tensor, start_pos: torch.Tensor,
                    max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                    cache: torch.Tensor, scale: Optional[torch.Tensor] = None):
            return key_value_cache(
                current_key, current_value, seqstarts, kvstarts, cachestarts, 
                start_pos, max_seqlen, max_kvlen, cache, scale,
                self.num_layer, self.layer_idx, self.quant_bit, self.quant_group)

    bsz = 4
    seqlen = 16
    max_seq_len = 64
    num_layer = 2
    layer_idx = 1
    quant_group = 8
    quant_bit = 8
    test_op1 = TestModule1(num_layer, layer_idx, quant_bit, quant_group)
    test_op2 = TestModule1(num_layer, layer_idx, 0, quant_group)

    cur_key = torch.ones([bsz * seqlen, 16, 64])
    cur_value = torch.ones([bsz * seqlen, 16, 64])
    seqstarts = torch.arange(0, (bsz + 1) * seqlen, seqlen, dtype=torch.int64)

    cache = torch.zeros([bsz * max_seq_len, num_layer, 2, 16, 64], dtype=torch.int8)
    scale = torch.zeros([bsz * max_seq_len, num_layer, 2, 16, 64 // quant_group])
    start_pos = torch.full([bsz], 32, dtype=torch.int64)
    cachestarts = torch.arange(0, bsz * max_seq_len, max_seq_len, dtype=torch.int64)

    kvstarts = torch.zeros([bsz + 1], dtype=torch.int64)
    kvstarts[1:] = start_pos.cumsum(0)
    kvstarts = kvstarts + seqstarts

    max_seqlen = torch.tensor([seqlen])
    max_kvlen = torch.tensor([seqlen + 32])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (cur_key, cur_value, seqstarts, kvstarts, cachestarts, start_pos, max_seqlen, max_kvlen, cache, scale), "KeyValueCache1.onnx", opset_version=11)
    model_str2 = torch.onnx.export_to_pretty_string(
        test_op2, (cur_key, cur_value, seqstarts, kvstarts, cachestarts, start_pos, max_seqlen, max_kvlen, cache), "KeyValueCache2.onnx", opset_version=11)

    print(model_str1)
    print(model_str2)
