# SiLU

Applies the Sigmoid Linear Unit (SiLU) function, element-wise. The SiLU function is also known as the swish function.

$$silu(x)=x∗sigmoid(x)$$

## Inputs

### `X`: tensor(T)

Shape: $(*)$

### `gate`(optional): tensor(T)

Shape: same as `X`

If `gate` is presentes, output should become $Y=gate*silu(X)$

## Outputs

### `Y`: tensor(T)

Shape: same as `X`

## Type Constraints

### `T`: float32, float16
