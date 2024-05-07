# WoquColumnParallelLinear

Woqu means weight only quantization

Apply [ColumnParallelLinear](./ColumnParallelLinear.md) with weight only quantizatiion.

## Attributes/Parameters

### `quant_data_type`: string

Quantized weight data type.

Options: `int4`, `int8`

### `quant_method`: string(default: "")

Reserved. Auxiliary attribute for quantization method.

### `quant_axis`: int(default: 1)

Axis direction of quantization parameters for per group or per tensor quantization. Only accept `0` or `1`. `0` for `out_features` and `1` for `in_features`. 

### `per_group`: bool(default: True)

Use per group quantization when `per_group == True`, otherwise use per channel(per out features) or per tensor quantization(determinate by shape of `Scale`).

### `group_size`: int(default: 128)

Group size for per group quantization.

### `has_zeropoint`: bool(default: False)

Is there zeropoint for asymmetry quantization. Provide convenience for graph optimization.

Definition of asymmetry quantization: `quantized = round(X / scale + zeropoint)` and `dequantized = scale * (quantized - zeropoint)`.

### `float_zeropoint`: bool(default: False)

Use floating point zeropoint for untraditional asymmetry quantization.

In this term the usage of zeropoint is different from traditional quantization method.

Zeropoint is performed as `zeropoint = (max(X) + min(X)) / 2`.

And `quantized = round((X - zeropoint) / scale)`, `dequantized = scale * quantized + zeropoint`.

Zeropoint is more like a bias.

### `in_features`: int

$K$ dim of weight.

### `out_features`: int

$N$ dim of weight.

### `bias_term`: bool(default: True)

Mark that whether there is bias term. Provide convenience for graph optimization.

### `gather_output`: bool(default: True)

Do all gather on output and make Y avaiable to all devices, otherwise, every device $d$ will hold its output which is $y_d = xW_d^T+b_d$.

## Inputs

### `X`: tensor(T1)

Input feature of linear transformation.

Shape: $(*,K)$, where $∗$ means any number of dimensions including none.

### `W`(constant): tensor(T2)

Transformation weight.

Shape: Different shape for each `quant_data_type`

- `int8`: for $(N,K)$ or $(N_d,K)$ for each device $d$ when $TPsize > 1$.
- `int4`: for $(N/4,K)$ or $(N_{d}/4,K)$ for each device $d$ when $TPsize > 1$. $N$ or $N\\_d$ must be aligned with 4. And weight data is packed as `int4x4`, so the datatype of `W` must be `int16`.

### `Scale`(constant): tensor(T1)

Quantization scale.

Shape: Different shape for each `quant_axis`, and `per_group`. Let's use some combinations as examples

- `per_channel` and `quant_axis==1`: $(N)$ or $(N_d)$ for each device $d$ when $TPsize > 1$.
- `per_tensor` : $(1)$ or scalar.
- `per_group` and `quant_axis==0`: $(N/group\\_size,K)$ or $(N_{d}/group\\_size,K)$ for each device $d$ when $TPsize > 1$. $N$ or $N\\_d$ must be aligned with `group_size`.
- `per_group` and `quant_axis==1`: $(N,K/group\\ _size)$ or $(N_{d},K/group\\_size)$ for each device $d$ when $TPsize > 1$. $K$ must be aligned with `group_size`.

### `ZeroPoint`(constant, optional): tensor(T3)

Quantization zeropoint, must appear and not be empty when `has_zeropoint == True`. Data type MUST be floating point when `float_zeropoint == True`.

Shape: Same as `Scale`

### `B`(constant, optional): tensor(T1)

Transformation bias.

Shape: $(N)$ or $(N_d)$ for each device $d$ when $TPsize > 1$. 

## Outputs

### `Y`: tensor(T1)

Output feature of linear transformation.

Shape: $(\*,N)$ or $(\*,N\\_d)$ for each device $d$ when `gather_output` is `False`, where $∗$ means any number of dimensions including none.

## Type Constraints

### `T1`: float32, float16

### `T2`: int8, int16(for int4x4)

### `T3`: int8, float16, float32
