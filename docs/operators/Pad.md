# Pad

Pad the input tensor similar to torch.nn.functional.pad. 

The padding size by which to pad some dimensions of input are described starting from the last dimension and moving forward.

See torch.nn.CircularPad2d, torch.nn.ConstantPad2d, torch.nn.ReflectionPad2d, and torch.nn.ReplicationPad2d for concrete examples on how each of the padding modes works. 


## Attributes/Parameters

### `mode`: str

'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'

## Inputs

### `input`: tensor(T)

N-dimensional tensor

### `padding`: tensor(int64)

m-elements tensor


### `value`:  float

fill value for 'constant' padding. Default: 0


## Outputs

### `output`: tensor(T)

Paded data.

## Type Constraints

### `T`: any