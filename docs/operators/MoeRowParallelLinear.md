# MoeRowParallelLinear

Apply [RowParallelLinear](./RowParallelLinear.md) function for each expert. Variables below `num_experts` is the number of experts, and `num_experts_per_token` is the number of selected experts for each token. 

Tensor parallel is performed along the $N$ dimension, and weight $W$ will be divided into $TPsize$ parts along the $N$ dimension: $W(N,K) \rightarrow W([N_0,N_1,\cdots,N_t], K)$. Same splitting operation for the bias $B$.

After each device in the same communicate world performs linear transformation, it is necessary to perform all reduce on the $TPsize$ parts of result whose shape are $(\*,N)$ to get the final result.

$TPsize$ means communicate world size of tensor parallel.

## Attributes/Parameters

### `num_experts`: int

Number of experts.

### `in_features`: int

$K$ dim of weight.

### `out_features`: int

$N$ dim of weight

### `bias_term`: bool(default: True)

Mark that whether there is bias term. Provide convenience for graph optimization.

### `input_is_parallel`: bool(default: False)

If true, we assume that the input is already split across the devices and we do not need to split it again.

If false, input should split into $TPsize$ pieces: $(\*,K) \rightarrow (\*,[K^0,K^1,\cdots,K^t])$, and each device pick its own piece.

## Inputs

### `X`: tensor(T1)

Input feature of linear transformation.

Shape: $(\*,K)$ or $(\*, K_d)$ for each device $d$ when `input_is_parallel` is `True`, where $∗$ means any number of dimensions including none.

### `expert_offset`: tensor(int64)

Contains the offset of the first token for each expert for `X` after flattening in dimension `*`. Region $[expert\\_offset[i], expert\\_offset[i+1])$ contains the position of tokens belong to expert_i, and `expert_offset[i+1]` is the prefix sum of tokens from expert_0 to expert_i.

```
X_flat = X.reshape(-1, K)
for i in range(1, num_expert+1):
    X_expert_i = X_flat[expert_offset[i]: expert_offset[i+1]]
```

Shape: $(num\\_experts + 1)$

### `W`(constant): tensor(T1)

Transformation weight.

Shape: $(num\\_experts,N,K)$ or $(num\\_experts,N,K\\_d)$ for each device $d$ when $TPsize > 1$. 

### `B`(constant, optional): tensor(T2)

Transformation bias.

Shape: $(num\\_experts,N)$. 

## Outputs

### `Y`: tensor(T2)

Output feature of linear transformation.

Shape: $(\*,N)$, where $∗$ means any number of dimensions including none.

## Type Constraints

### `T1`: float32, float16, int8

### `T2`: float32, float16, int8, int32

enable accumulate with int32 when using int8 linear
