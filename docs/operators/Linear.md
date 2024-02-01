# Linear

Applies a linear transformation to the incoming data:

$$y=XW^T+b$$

## Attributes/Parameters

### `in_features`: int

$K$ dim of weight.

### `out_features`: int

$N$ dim of weight

### `bias_term`: bool(default: True)

Mark that whether there is bias term. Provide convenience for graph optimization.

## Inputs

### `X`: tensor(T1)

Input feature of linear transformation.

Shape: $(\*,K)$

### `W`(constant): tensor(T1)

Transformation weight.

Shape: $(N,K)$. 

### `B`(constant, optional): tensor(T2)

Transformation bias.

Shape: $(N)$. 

## Outputs

### `Y`: tensor(T2)

Output feature of linear transformation.

Shape: $(\*,N)$.

## Type Constraints

### `T1`: float32, float16, int8

### `T2`: float32, float16, int8, int32

enable accumulate with int32 when using int8 linear