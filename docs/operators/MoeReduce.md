# MoeReduce

Perform a reduce sum operation on tokens from different experts and then reorder the tokens to their original positions

## Attributes/Parameters

### `num_experts_per_token`: int

Number of selected experts for each token.

## Inputs

### `Y_expand_permute`: tensor(T1)

Input feature before invert permutation and reduce sum.

Shape: $(\*,num\\_experts\\_per\\_token,N)$

### `expert_weights`: tensor(T1)

Expert weights of each token.

Shape: $(\*,num\\_experts\\_per\\_token)$

### `invert_permutation`: tensor(int64)

The indices of invert permutation: mapping from permuted token index to origin token index. 

Shape: $(\*, num\\_experts\\_per\\_token)$

## Outputs

### `Y_reduced`: tensor(T1)

Output feature after invert permutation and reduce sum.

Shape: $(\*,N)$

## Type Constraints

### `T1`: float32, float16, int8

