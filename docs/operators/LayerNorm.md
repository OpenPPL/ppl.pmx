# LayerNorm

Applies Layer Normalization over a mini-batch of inputs as described in the paper [Layer Normalization](https://arxiv.org/abs/1607.06450)

$$y=\frac{x-E[x]}{\sqrt{Var[x]+\epsilon}}*\gamma+\beta$$

The mean and standard-deviation are calculated over the last $D$ dimensions, where $D$ is the dimension split by `axis`. For example, if `axis` = -2, the mean is computed over the last 2 dimensions of the input.
$\gamma$ and $\beta$ are learnable affine transform parameters with shape as last $D$ dimensions of input.

SkipLayerNorm is performed by the formula below:

$$z=skip\\_out=x+skip\\_in$$

$$y=\frac{z-E[z]}{\sqrt{Var[z]+\epsilon}}*\gamma+\beta$$

## Attributes/Parameters

### `axis`: int(default: -1)

axis to split the normalization dimension.

### `eps`: float(default: 1e-5)

$\epsilon$ for Layer normalization.

### `elementwise_affine `: bool(default: False)

Whether has `W` and `B`.

### `skip_term`: bool(default: False)

Whether apply SkipLayerNorm.

## Inputs

### `X`: tensor(T)

Input features.

Shape: $(D_0, D_1, \dots,D_N)$

### `W`(constant, optional): tensor(T)

Transformation weight.

Shape: $(D_{axis}, \dots, D_N)$

### `B`(constant, optional): tensor(T)

Transformation bias.

Shape: $(D_{axis}, \dots, D_N)$

### `SkipIn`(optional): tensor(T)

Skip input.

Shape: same as `X` 

## Outputs

### `Y`: tensor(T)

Output features.

Shape: same as `X`

### `SkipOut`(optional): tensor(T)

SkipOutput. If `SkipIn` is not appear, `SkipOut` will be a copy of `X`

Shape: same as `X`.

## Type Constraints

### `T`: float32, float16

If input is float16, data will convert to float32 before LayerNorm.
