# Swish

Applies the Swish function, element-wise. The Swish function is also known as the  function.

$$swish(x)=x∗sigmoid(\beta x)$$

## Attributes/Parameters

### `beta`: float(default: 1.0f)

## Inputs

### `X`: tensor(T)

Shape: $(*)$

### `gate`(optional): tensor(T)

Shape: same as `X`

If `gate` is presentes, output should become $Y=gate*swish(X)$

## Outputs

### `Y`: tensor(T)

Shape: same as `X`

## Type Constraints

### `T`: float32, float16
