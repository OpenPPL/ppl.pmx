# Pad

Pad the input tensor similar to torch.nn.functional.pad. 

## Attributes/Parameters

### `mode`: str (default: 'constant')

'constant', 'reflect', 'replicate' or 'circular'.

See torch.nn.CircularPad2d, torch.nn.ConstantPad2d, torch.nn.ReflectionPad2d, and torch.nn.ReplicationPad2d for concrete examples on how each of the padding modes works. 

## Inputs

### `input`: tensor(T)

N-dimensional tensor

### `padding`: tensor(int64)

m-elements tensor, where $\frac{m}{2} \leq$ input dimensions and $m$ is even.

The padding size by which to pad some dimensions of input are described starting from the last dimension and moving forward.
$\left\lfloor\frac{\operatorname{len}(\mathrm{padding})}{2}\right\rfloor$ dimensions of input will be padded.

For example, 

to pad only the last dimension of the input tensor, padding use 
(padding_left, padding_right); 

to pad the last 2 dimensions of the input tensor, padding use (padding_left, padding_right, padding_top, padding_bottom)

to pad the last 3 dimensions of the input tensor, padding use (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)


### `value`:  (optional):tensor(float)

fill value for 'constant' padding. 


## Outputs

### `output`: tensor(T)

Paded data.

## Type Constraints

### `T`: any