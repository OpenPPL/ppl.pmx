# MoeSelect

Select the experts with higher probability from all experts for each token, and sort all tokens according to their assigned expert id. For Mixtral-7B, it selects top-2 experts from 8 experts for each token.

## Attributes/Parameters

### `num_experts`: int

Number of expert.

### `num_experts_per_token`: int

Number of selected experts for each token.

## Inputs

### `X`: tensor(T1)

Input feature. 

Shape: $(\*,K)$, where $\*$ means any number of dimension including none.

### `scores`: tensor(T1)

Routing scores after moe gate of each token.

Shape: $(\*, num\\_experts)$

## Outputs

### `X_expand_permute`: tensor(T1)

Input feature X after expand and permute. Each token's size will be expanded to `num_experts_per_token`, and permuted by order of their expert id. 

Shape: $(\*, num\\_experts\\_per\\_token, K)$

### `expert_weights`: tensor(T1)

Select top `num_experts_per_token` from scores, and normalize it with softmax.

Shape: $(\*, num\\_experts\\_per\\_token)$

### `invert_permutation`: tensor(int64)

The indices of invert permutation: mapping from permuted token index to origin token index. 

Shape: $(\*, num\\_experts\\_per\\_token)$

### `expert_offset`: tensor(int64)

Contains the offset of the first token for each expert. Region $[expert\\_offset[i], expert\\_offset[i+1])$ contains the position of tokens belong to expert_i, and `expert_offset[i+1]` is the prefix sum of tokens from expert_0 to expert_i.

Shape $(num\\_experts + 1)$

## Type Constraints

### `T1`: float32, float16, int8
