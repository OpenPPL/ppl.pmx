# GeGLU

Apply gated GELU to $X$, where last dimension of $X$ is $D$.

$$GeGLU(X)=X[\cdots, D/2:]*GELU(X[\cdots, :D/2])$$

## Attributes/Parameters

### `approximate`: bool(default: False)

Estimated GELU with tanh approximate

## Inputs

### `X`: tensor(T)

Shape: $(*,D)$

## Outputs

### `Y`: tensor(T)

Shape: $(*,D/2)$

## Type Constraints

### `T`: float32, float16
