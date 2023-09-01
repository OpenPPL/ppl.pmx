# ColumnParallelLinear

Applies a linear transformation to the incoming data:

$$y=xW^T+b$$

Tensor parallel is performed along the $N$ dimension, and weight $W$ will be divided into $TPsize$ parts along the $N$ dimension: $W(N,K) \rightarrow W([N_0,N_1,\cdots,N_t], K)$. Same splitting operation for the bias $B$.

$TPsize$ means communicate world size of tensor parallel.

## Attributes/Parameters

### `in_features`: int

$K$ dim of weight.

### `out_features`: int

$N$ dim of weight.

### `bias_term`: bool(default: True)

Mark that whether there is bias. Provide convenience for graph optimization.

### `gather_output`: bool(default: True)

Do all gether on output and make Y avaiable to all device, otherwise, every device $d$ will hold its output which is $y_d = xW_d^T+b_d$.

## Inputs

### `X`: tensor(T1)

Input feature of linear transformation.

Shape: $(*,K)$, where $∗$ means any number of dimensions including none.

### `W`(constant): tensor(T1)

Transformation weight.

Shape: $(N,K)$ or $(N_d,K)$ for each device $d$ when `gather_output` is `False`. 

### `B`(constant, optional): tensor(T2)

Transformation bias.

Shape: $(N)$ or $(N_d)$ for each device $d$ when `gather_output` is `False`. 

## Outputs

### `Y`: tensor(T2)

Output feature of linear transformation.

Shape: $(*,N)$ or $(*, N_d)$ for each device $d$ when `gather_output` is `False`, where $∗$ means any number of dimensions including none.

## Type Constraints

### `T1`: float32, float16, int8

### `T2`: float32, float16, int8, int32

enable accumulate with int32 when using int8 linear
