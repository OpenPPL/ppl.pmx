# RMSNorm

Applies RMS(Root Mean Square) Normalization over a mini-batch of inputs as described in the paper [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

$$y=\frac{x}{\sqrt{E[x^2]+\epsilon}}*\gamma$$

The mean is calculated over the last $D$ dimensions, where $D$ is the dimension split by `axis`. For example, if `axis` = -2, the mean is computed over the last 2 dimensions of the input.
$\gamma$ is learnable affine transform parameters with shape as last $D$ dimensions of input.

SkipRMSNorm is performed by the formula below:

$$z=skip\\_out=x+skip\\_in$$

$$y=\frac{z}{\sqrt{E[z^2]+\epsilon}}*\gamma$$

## Attributes/Parameters

### `axis`: int(default: -1)

axis to split the normalization dimension.

### `eps`: float(default: 1e-5)

$\epsilon$ for RMS normalization.

### `skip_term`: bool(default: False)

Whether apply SkipRMSNorm.

## Inputs

### `X`: tensor(T)

Input features.

Shape: $(D_0, D_1, \dots,D_N)$

### `W`(constant): tensor(T)

Transformation weight.

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

Shape: same as `X`

## Type Constraints

### `T`: float32, float16

If input is float16, data will convert to float32 before RMSNorm.
