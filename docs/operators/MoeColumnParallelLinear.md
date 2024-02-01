# MoeColumnParallelLinear

Apply [ColumnParallelLinear](./ColumnParallelLinear.md) function for each expert. Variable `num_experts` is the number of experts, and `num_experts_per_token` is the number of selected experts for each token. 

Tensor parallel is performed along the $N$ dimension, and weight $W$ will be divided into $TPsize$ parts along the $N$ dimension: $W(N,K) \rightarrow W([N_0,N_1,\cdots,N_t], K)$. Same splitting operation for the bias $B$.

$TPsize$ means communicate world size of tensor parallel.

## Attributes/Parameters

### `num_experts`: int

Number of experts.

### `in_features`: int

$K$ dim of weight.

### `out_features`: int

$N$ dim of weight.

### `bias_term`: bool(default: True)

Mark that whether there is bias term. Provide convenience for graph optimization.

### `gather_output`: bool(default: True)

Do all gather on output and make Y avaiable to all devices, otherwise, every device $d$ will hold its output which is $y_d = xW_d^T+b_d$.

## Inputs

### `X`: tensor(T1)

Input feature of linear transformation.

Shape: $(\*,K)$, where $∗$ means any number of dimensions including none.

### `expert_offset`: tensor(int64)

Contains the offset of the first token for each expert for `X` after flattening in dimension `*`. Region $[expert\\_offset[i], expert\\_offset[i+1])$ contains the position of tokens belong to expert_i, and `expert_offset[i+1]` is the prefix sum of tokens from expert_0 to expert_i.

```
X_flat = X.reshape(-1, K)
for i in range(1, num_expert+1):
    X_expert_i = X_flat[expert_offset[i]: expert_offset[i+1]]
```

Shape: $(num\\_experts + 1)$

### `W`(constant): tensor(T1)

Transformation weight of all experts.

Shape: $(num\\_experts,N,K)$ or $(num\\_experts,N\\_d,K)$ for each device $d$ when $TPsize > 1$. 

### `B`(constant, optional): tensor(T2)

Transformation bias of all experts.

Shape: $(num\\_experts,N)$ or $(num\\_experts,N\\_d)$ for each device $d$ when $TPsize > 1$.

## Outputs

### `Y`

Output feature of linear transformation.

Shape: $(\*,N)$ or $(\*,N\\_d)$ for each device $d$ when `gather_output` is `False`, where $∗$ means any number of dimensions including none.

## Type Constraints

### `T1`: float32, float16, int8

### `T2`: float32, float16, int8, int32

enable accumulate with int32 when using int8 linear