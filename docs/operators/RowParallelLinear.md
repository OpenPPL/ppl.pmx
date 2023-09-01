# RowParallelLinear

Applies a linear transformation to the incoming data:

$$y=xW^T+b$$

Tensor parallel is performed along the $K$ dimension, and weight $W$ will be divided into $TPsize$ parts along the $K$ dimension: $W(N,K) \rightarrow W(N,[K^0,K^1,\cdots,K^t])$. Each device will have full bias $B$.

After each device in the same communicate world performs linear transformation, it is necessary to perform all reduce on the $TPsize$ parts of result whose shape are $(*,N)$ to get the final result.

$TPsize$ means communicate world size of tensor parallel.

## Attributes/Parameters

### `in_features`: int

$K$ dim of weight.

### `out_features`: int

$N$ dim of weight

### `bias_term`: bool(default: True)

Mark that whether there is bias. Provide convenience for graph optimization.

### `input_is_parallel`: bool(default: False)

If true, we assume that the input is already split across the devices and we do not need to split it again.

If false, input should split into $TPsize$ pieces: $(*,K) \rightarrow (*,[K^0,K^1,\cdots,K^t])$, and each device pick its own piece.

## Inputs

### `X`: tensor(T1)

Input feature of linear transformation.

Shape: $(*,K)$ or $(*, K_d)$ for each device $d$ when `input_is_parallel` is `True`, where $∗$ means any number of dimensions including none.

### `W`(constant): tensor(T1)

Transformation weight.

Shape: $(N,K)$ or or $(N,K_d)$ for each device $d$ when $TPsize > 1$. 

### `B`(constant, optional): tensor(T2)

Transformation bias.

Shape: $(N)$. 

## Outputs

### `Y`: tensor(T2)

Output feature of linear transformation.

Shape: $(*,N)$, where $∗$ means any number of dimensions including none.

## Type Constraints

### `T1`: float32, float16, int8

### `T2`: float32, float16, int8, int32

enable accumulate with int32 when using int8 linear
