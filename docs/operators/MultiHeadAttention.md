# MultiHeadAttention

Allows the model to jointly attend to information from different representation subspaces as described in the paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

Multi-Head Attention(${\rm MHA}$) is defined as:

$${\rm MHA}(Q,K,V)=[head_1, head_2,...,head_h]$$

$$head_i={\rm softmax}(\frac{Q_iK_i^T}{\sqrt{head\\_dim}})V_i$$

$Q$ is `query`, $K$ is `key` and $V$ is `value`.

Shape of $Q$ is $(batch, num\\_heads, seqlen\\_q, head\\_dim)$ and shape of $K$ and $V$ are $(batch, num\\_kv\\_heads, seqlen\\_kv, head\\_dim)$.

But in this operator, shape of $Q$ will be $(batch, seqlen\\_q, num\\_heads, head\\_dim)$ and shape of $K$ and $V$ will be $(batch, seqlen\\_kv, num\\_kv\\_heads, head\\_dim)$. So we need to do some transpose before applying attention.

## Attributes/Parameters

### `num_heads`: int

Number of heads

### `head_dim`: int

Dimension of each head, where $head\\_dim * num\\_heads = hidden\\_dim$

### `is_causal`: bool

Whether do casual mask when sequence length > 1. If `is_causal` is `True`, `seqlen_q` must be equal to `seqlen_kv`

### `num_kv_heads`: int(default: 0)

For [Grouped-Query Attention](https://arxiv.org/pdf/2305.13245.pdf). If `num_kv_heads` and `num_heads` are not equal, we should repeat key and value `num_heads/num_kv_heads` times before applying ${\rm MHA}$ for each token. `num_heads` must be divisible by `num_kv_heads`. Default is 0, and at this point, `num_heads` is used as `num_kv_heads`.

## Inputs

### `query`: tensor(T)

Input Query tensor

Shape: $(batch, seqlen\\_q, num\\_heads, head\\_dim)$
### `key`: tensor(T)

Input Key tensor

Shape: $(batch, seqlen\\_kv, num\\_kv\\_heads, head\\_dim)$

### `value`: tensor(T)

Input Value tensor

Shape: $(batch, seqlen\\_kv, num\\_kv\\_heads, head\\_dim)$

### `attn_mask`(optional): tensor(T)

Optional custom mask. If shape is not $(batch, num\\_heads, seqlen\\_q, seqlen\\_kv)$, `attn_mask` will be broadcasted.

Shape: $(seqlen\\_q, seqlen\\_kv)$ or $(num\\_heads, seqlen\\_q, seqlen\\_kv)$ or $(batch, num\\_heads, seqlen\\_q, seqlen\\_kv)$

## Outputs

### `attn_output`: tensor(T)

Output feature of attention result

Shape: $(batch, seqlen\\_q, num\\_heads, head\\_dim)$

## Type Constraints

### `T`: float32, float16
