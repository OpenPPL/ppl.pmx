# Reshape

Reshape the input tensor similar to numpy.reshape. First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor. At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions. A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor). The input tensor's shape and the output tensor's shape are required to have the same number of elements.

## Inputs

### `data`: tensor(T)

An input tensor.

### `shape`: tensor(int64)

Specified shape for output.

## Outputs

### `reshaped`: tensor(T)

Reshaped data.

## Type Constraints

### `T`: any
