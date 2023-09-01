# GELU

Applies the Gaussian Error Linear Units function:

$$GELU(x)=x*\Phi(x)=0.5*x*(1+Erf(x/\sqrt{2}))$$

$\Phi(x)$ is the Cumulative Distribution Function for Gaussian Distribution.

When the `approximate` is `True`, GELU is estimated with:

$$GELU(x)=0.5*x*(1+Tanh(\sqrt{2/\pi}*(x+0.44715*x^3)))$$

## Attributes/Parameters

### `approximate`: bool(default: False)

Estimated GELU with tanh approximate

## Inputs

### `X`: tensor(T)

Shape: $(*)$

## Outputs

### `Y`: tensor(T)

Shape: same as `X`

## Type Constraints

### `T`: float32, float16
