# Rotary2DPositionEmbedding

The definition of `RotaryPositionEmbedding` refers to [here](RotaryPositionEmbedding.md).

The difference between `Rotary2DPositionEmbedding` and `RotaryPositionEmebdding` is the dimension of positions and how they handle with hidden features dimensions. The former splits the feature dimension into two parts, and applies Rotary Position Embedding with their position index to each part separately. Afterwards, the two parts are merged back together.

As the code below, The position mapping of two part are different.

```python
for b in range(batch):
    for s in range(seqlen):
        for nh in range(num_heads):
            pivot = query.shape[-1] // 2
            offset = start_pos + s
            prompt_len = first_seqlen - pad_len[b]
            if offset < pad_len[b]:
                pos0 = 0
                pos1 = 0
            elif offset < first_seqlen - 1:
                pos0 = offset - pad_len[b]
                pos1 = 0
            else:
                pos0 = prompt_len - 2
                pos1 = offset - prompt_len + 2
            rotated_query[b, s, nh, :pivot]
                = f(query[b, s, nh, :pivot], pos0)
            rotated_query[b, s, nh, pivot:]
                = f(query[b, s, nh, pivot:], pos1)
```

## Attributes/Parameters

### `theta`: float(default: 10000.0)

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
