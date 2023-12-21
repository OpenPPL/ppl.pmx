# SwiGLU

Apply gated Swish to $X$, where last dimension of $X$ is $D$.

$$Swish(X)=x*sigmoid(\beta X)$$

$$SwiGLU(X)=X[\cdots, D/2:]*Swish(X[\cdots, :D/2])$$

## Attributes/Parameters

### `beta`: float(default: 1.0f)

## Inputs

### `X`: tensor(T)

Shape: $(*,D)$

## Outputs

### `Y`: tensor(T)

Shape: $(*,D/2)$

## Type Constraints

### `T`: float32, float16
