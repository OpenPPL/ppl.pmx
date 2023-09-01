# RotaryPositionEmbedding

Rotary Position Embedding, or RoPE, is a type of position embedding which encodes absolute positional information with rotation matrix and naturally incorporates explicit relative position dependency in self-attention formulation, as described in the paper [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4). 

Rotary Position Embedding is defined as:

$$f(q, m) = qe^{im\theta}= {\begin{pmatrix} \cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta \end{pmatrix}} {\begin{pmatrix} a \\ b \end{pmatrix}}$$

$m$ is position index，$q=\begin{pmatrix} a \\ b \end{pmatrix}$ is input features.

For each input `query` and `key`, do rotary position embedding on dimension $head\\_dim$, where each pair of elements as $q$, even element as $a$ and odd element as $b$.

$$Q=[q_0, q_1, \cdots, q_{\frac{head\\_dim}{2}}]$$

Position index is start from `start_pos`, which means value of $m$ is from `start_pos` to `start_pos + seqlen`. For example, rotary position embedding of `query` is perform as:

```python
for b in range(batch):
    for s in range(seqlen):
        for nh in range(num_heads):
            pivot = rotary_dim if rotary_dim else query.shape[-1]
            offset = start_pos + s - pad_len[b]
            rotated_query[b, s, nh, :pivot] =
                f(query[b, s, nh, :pivot], offset)
```

## Attributes/Parameters

### `rotary_dim`: int(default: 0)

How many elements in dimension $head\\_dim$ to be rotary, must be even number. Default is `0`, which means all elements should be rotary. Otherwise only rotary $Q_r = Q(:\frac{rotary\_dim}{2})$.

### `theta`: int(default: 10000)

Hyperameter $\theta$ to adjust rotate angle interval.

### `bypass_key`: bool(default: False)

Bypass rotating `key` for compatibility.

## Inputs

### `query`: tensor(T)

Input query tensor.

Shape: $(batch, seq\\_len, num\\_heads, head\\_dim)$

### `key`: tensor(T)

Input key tensor.

Shape: $(batch, seq\\_len, num\\_k\\_heads, head\\_dim)$

### `start_pos`: scalar(int64)

Start position in a sequence.

### `pad_len`(optional): tensor(int64)

Padding length of each sequence. position of each batch `b` should start from `start_pos - pad_len[b]`.

Shape: $(batch)$

## Outputs

### `rotated_query`: tensor(T)

Query tensor after rotary position embedding.

Shape: $(batch, seq\\_len, num\\_heads, head\\_dim)$

### `rotated_key`: tensor(T)

Key tensor after rotary position embedding .

Shape: $(batch, seq\\_len, num\\_k\\_heads, head\\_dim)$

### `T`: float32, float16, int8
