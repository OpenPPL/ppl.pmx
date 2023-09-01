# dynamic_batching.MultiHeadAttention

The original definition of `MultiHeadAttention` refers to [here](../MultiHeadAttention.md).

For `dynamic_batching.MultiHeadAttention`, sequence length and key/value length of each batch are different.

Because dynamic batching will combine decoding attention and first-fill attention togather, we have to pass some information to separate input into two part by `decoding_batches`.
The first part in the front is decoding part, whose size is equal to `decoding_batches`, will never apply causal mask. And the batches remain are first-fill part, who will be apply with causal mask if `is_causal` is `True`.

## Attributes/Parameters

### `num_heads`: int

Number of heads

### `head_dim`: int

Dimension of each head, where $head\\_dim * num\\_heads = hidden\\_dim$

### `is_causal`: bool

Whether do casual mask when sequence length > 1. If `is_causal` is `True`, length of `query`, `key` and `value` within every batchs must be equal.

### `num_kv_heads`: int(default: 0)

For [Grouped-Query Attention](https://arxiv.org/pdf/2305.13245.pdf). If `num_kv_heads` and `num_heads` are not equal, we should repeat key and value `num_heads/num_kv_heads` times before applying ${\rm MHA}$ for each token. `num_heads` must be divisible by `num_kv_heads`. Default is 0, and at this point, `num_heads` is used as `num_kv_heads`.

## Inputs

### `query`: tensor(T)

Input Query tensor

Shape: $(seqstarts[B], num\\_heads, head\\_dim)$
### `key`: tensor(T)

Input Key tensor

Shape: $(kvstarts[B], num\\_kv\\_heads, head\\_dim)$

### `value`: tensor(T)

Input Value tensor

Shape: $(kvstarts[B], num\\_kv\\_heads, head\\_dim)$

### `seqstarts`: tensor(int64)

`seqstarts[:B]` contains the position of the first token in `query` for each batch. And `seqstarts[B]` contains the total length of `query`.

Note that `seqstarts[b+1]-seqstarts[b]` can calculate out the sequence length of batch $b$.

Shape: $(B+1)$

### `kvstarts`: tensor(int64)

`kvstarts[:B]` contains the position of the first token in `key` and `value` for each batch. And `kvstarts[B]` contains the total length of `key` and `value`.

Note that `kvstarts[b+1]-kvstarts[b]` can calculate out the key and value length of batch $b$.

Shape: $(B+1)$

### `decoding_batches`: scalar(int64)

Describe how many batches in front are being decoded, those who are not need causal mask.

### `max_seqlen`: scalar(int64)

Maximum sequence length of `query`, equal to `max(seqstarts[1:]-seqstarts[:B])`. For parallel computing.

### `max_kvlen`: scalar(int64)

Maximum sequence length of `key` and `value`, equal to `max(kvstarts[1:]-kvstarts[:B])`. For parallel computing.

### `attn_mask`(optional): tensor(T)

Optional custom mask.
`seqlens=seqstarts[1:]-seqstarts[:B]` is a sequence contains length of `query` for each batch.
`kvlens=kvstarts[1:]-kvstarts[:B]` is a sequence contains length of `key` and `value` for each batch.

Shape: $(num\\_heads, {\rm sum}(seqlens), {\rm sum}(kvlens))$ or $({\rm sum}(seqlens), {\rm sum}(kvlens))$

## Outputs

### `attn_output`: tensor(T)

Output feature of attention result

Shape: $(seqstarts[B], num\\_heads, head\\_dim)$

## Type Constraints

### `T`: float32, float16
