# KeyValueCache

An operator used to manage key and value cache access. Can be combined with linear quantized compression for memory usage optimization. The cache space contains caches of multiple Attention layers, and the `layer_idx` parameter is used to index the caches of each layer to provide more flexible memory management capabilities.

Quantize method refer to: [KeyValueCache Quantization](/docs/appendix/KeyValueCacheQuantization.md)

In the description below, `start_p` is `start_pos`, `s` is `sequence_length`, and `l` is `layer_idx`. If `quant_bit` is zero, the scale access and the cache quantization/dequantization process can be skipped.

Below is an example when `cache_layout` is `0`:

First store the input key and value in the position of `start_pos` indexing
```python
k_scale, k_quant = quant(current_key)
v_scale, v_quant = quant(current_value)
scale[:batch, l, 0, start_p:start_p + s, :, :] = k_scale
cache[:batch, l, 0, start_p:start_p + s, :, :] = k_quant
scale[:batch, l, 1, start_p:start_p + s, :, :] = v_scale
cache[:batch, l, 1, start_p:start_p + s, :, :] = v_quant
```

Then extract the key and value from the begining to `start_pos + sequence_length`
```python
k_quant = cache[:batch, l, 0, :start_p + s, :, :]
k_scale = scale[:batch, l, 0, :start_p + s, :, :]
v_quant = cache[:batch, l, 1, :start_p + s, :, :]
v_scale = scale[:batch, l, 1, :start_p + s, :, :]
key = dequant(k_quant, k_scale)
value = dequant(v_quant, v_scale)
```

In the description below, $L$ is number of attention layers(`num_layer`), $B$ is batch size, $S$ is sequence length, $MaxS$ is max sequence length, $H$ is `num_heads` of transformer, $Dh$ is `dims_per_head` or `head_dim` of transformer.

> NOTE：`cache` and `scale` are used as in-out tensor, so it is recommended to use them as model inputs, and let the user set the shape by themselves (mainly because `max_sequence_length` and `num_layer` need to be configured separately).

## Attributes/Parameters

### `num_layer`: int(default: 1)

Number of attention layers.

### `layer_idx`: int(default: 0)

Attention layer index for cache and scale.

### `quant_bit`: int(default: 0)

Quantize bit for cache compression. For example, 8 means int8 compression. `0` means disabled.

### `quant_group`: int(default: 8)

Quantize scale shared group size. $2^n$ and $n > 2$ is recommanded for hardware implementation.

### `num_repeat`: int(default: 1)

For [Grouped-Query Attention](https://arxiv.org/pdf/2305.13245.pdf). Repeat key and value `num_repeat` time on axis `num_heads` to construct an input compatiltable with non-grouped MultiHeadAttention.

### `cache_layout`: int(default: 0)

Define data layout of `cache` and `scale`. Default is zero.

Meaning of numbers:
- `0`: $cache(MaxB,L,2,MaxS,H,Dh)$ and $scale(MaxB,L,2,MaxS,H,Dh/quant\\_group)$
- `1`: $cache(L,MaxB,2,H,MaxS,Dh)$ and $scale(L,MaxB,2,H,MaxS,Dh/quant\\_group)$

## Inputs

### `current_key`: tensor(T1)

Shape: $(B,S,H,Dh)$

### `current_value`: tensor(T1)

Shape: $(B,S,H,Dh)$

### `start_pos`: scalar(int64)

Sequence position where `current_key` and `current_value` begining to store.

### `cache`: tensor(T2)

Shape: Determinated by `cache_layout`.

Contains key and value caches of attention layer. When `cache_layout` is `0`, subspace $(:B,:,0,:,:,:)$ contains key caches and subspace $(:B,:,1,:,:,:)$ contains value caches. **Data in this tensor will be modified.**

### `scale`(optional): tensor(T3)

Shape: Determinated by `cache_layout`.

Contains key and value cache quantize scales of attention layer. When `cache_layout` is `0`, subspace $(:B,:,0,:,:,:)$ contains key cache scales and subspace $(:B,:,1,:,:,:)$ contains value cache scales. Must appear if `quant_bit` is not zero. **Data in this tensor will be modified.**

## Outputs

### `key`: tensor(T1)

Shape: $(B,start\\_pos+S,H*num\\_repeat,Dh)$

Key contains current key and all pass key. If `quant_bit` is not `0`, it should be decompressed.

### `value`: tensor(T1)

Shape: $(B,start\\_pos+S,H*num\\_repeat,Dh)$

Value contains current value and all pass value. If `quant_bit` is not `0`, it should be decompressed.

## Type Constraints

### `T1`: float32, float16, int8

### `T2`: float32, float16, int8, int4

### `T3`: float32, float16
