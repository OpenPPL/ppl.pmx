# DequantizeLinear

The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full-precision tensor. `Scale` and `ZeroPoint` must have the same shape, determining the quantization’s granularity: a scalar for per-tensor/per-layer quantization, a a rank identical tensor for per-axis quantization and grouped quantization. See details of `Scale` for quantization granularity.

## Attributes/Parameters

### `quant_data_type`: string

Quantized weight data type.

Options: `int4`, `int8`

### `quant_method`: string(default: "")

Reserved. Auxiliary attribute for quantization method.

### `quant_axis`: int(default: 1)

Axis direction of quantization parameters for per group or per tensor quantization. Only accept `0` or `1`. `0` for `out_features` and `1` for `in_features`. 

### `group_size`: int(default: 128)

Group size for per group quantization.

Use per group quantization when `group_size != 0`, otherwise use per channel(per out features) or per tensor quantization(determinate by shape of `Scale`).

### `has_zeropoint`: bool(default: False)

Is there zeropoint for asymmetry quantization. Provide convenience for graph optimization.

Definition of asymmetry quantization: `quantized = round(X / scale + zeropoint)` and `dequantized = scale * (quantized - zeropoint)`.

### `float_zeropoint`: bool(default: False)

Use floating point zeropoint for untraditional asymmetry quantization.

In this term the usage of zeropoint is different from traditional quantization method. And zeropoint is more like a bias.

Zeropoint is performed as `zeropoint = (max(X) + min(X)) / 2`.

And `quantized = round((X - zeropoint) / scale)`, `dequantized = scale * quantized + zeropoint`.

## Inputs

### `X`: tensor(T1)

Input quantized tensor.

Shape: Different shape for each `quant_data_type`

- `int8` or higher bit qunatization: $(d_1, d_2, \cdots, d_n)$
- `int4`: $(d_0/4, d_1, \cdots, d_n)$. Data is packed as `int4x4`, so the datatype of `X` must be `int16`.

### `Scale`(constant): tensor(T2)

Quantization scale.

Shape: Different shape for each `quant_axis`, and `per_group`. Let's use some combinations as examples

- `per_channel` and `quant_axis == 1`: $(d_0, 1, \cdots, d_n)$.
- `per_tensor` : $(1)$ or scalar.
- `group_size != 0` and `quant_axis == 0`: $(d_0/group\\_ size, d_1, \cdots, d_n)$. $d_0$ must be aligned with `group_size`.
- `group_size != 0` and `quant_axis == 1`: $(d_0, d_1/group\\_ size, \cdots, d_n)$. $d_0$ must be aligned with `group_size`.

### `ZeroPoint`(constant, optional): tensor(T3)

Quantization zeropoint, must appear and not be empty when `has_zeropoint == True`. Data type MUST be floating point when `float_zeropoint == True`.

Shape: Same as `Scale`

## Outputs

### `Y`: tensor(T2)

Output dequantized tensor.

Shape: $(d_1, d_2, \cdots, d_n)$

## Type Constraints

### `T1`: int8, int16(for int4x4)

### `T2`: float32, float16

### `T3`: int8, float16, float32
