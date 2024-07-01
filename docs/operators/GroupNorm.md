# GroupNorm

Applies Group Normalization over a mini-batch of inputs as described in the paper [Group Normalization](https://arxiv.org/abs/1803.08494)

$$y=\frac{x-E[x]}{\sqrt{Var[x]+\epsilon}}*\gamma+\beta$$


The input shape is (N, C, *), where C = num_channels,
The input channels are separated into $num\_groups$ groups, each containing $num\_channels / num\_groups$ channels. $num\_channels$ must be divisible by $num\_groups$. The mean and standard-deviation are calculated separately over the each group. $\gamma$ and $\beta$ are learnable per-channel affine transform parameter vectors of size $num\_channels$. The standard-deviation is calculated via the biased estimator, equivalent to torch.var(input, unbiased=False).

## Attributes/Parameters

### `num_groups`: int(default: -1)

number of groups to separate the channels into

### `eps`: float(default: 1e-5)

$\epsilon$ for Group normalization.

### `elementwise_affine `: bool(default: False)

Whether has `Weight` and `Bias`.

## Inputs

### `X`: tensor(T)

Input features.

Shape: $(N, C, *)$

### `Weight`(constant, optional): tensor(T)

Transformation weight.

Shape: $(C,)$

### `Bias`(constant, optional): tensor(T)

Transformation bias.

Shape: $(C,)$


## Outputs

### `Y`: tensor(T)

Output features.

Shape: same as `X`


## Type Constraints

### `T`: float32, float16

If input is float16, data will convert to float32 before GroupNorm.