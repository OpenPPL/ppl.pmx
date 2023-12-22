# dynamic_batching.Rotary2DPositionEmbedding

The original definition of `Rotary2DPositionEmbedding` refers to [here](../Rotary2DPositionEmbedding.md).

For `dynamic_batching.Rotary2DPositionEmbedding`, start postion and sequence length of each batch are different. And there is no need for `pad_len` because paddings are removed in dynamic batching term.

## Attributes/Parameters

### `theta`: float(default: 10000.0)

Hyperameter $\theta$ to adjust rotate angle interval.

### `bypass_key`: bool(default: False)

Bypass rotating `key` for compatibility.

## Inputs

### `query`: tensor(T)

Input query tensor.

Shape: $(seqstarts[batch], num\\_heads, head\\_dim)$

### `key`: tensor(T)

Input key tensor.

Shape: $(seqstarts[batch], num\\_k\\_heads, head\\_dim)$

### `seqstarts`: tensor(int64)

`seqstarts[:batch]` contains the position of the first token in `current_key` and `current_value` for each batch. And `seqstarts[batch]` contains the total length of `current_key` and `current_value`.

Note that `seqstarts[b+1]-seqstarts[b]` can calculate out the sequence length of batch $b$.

Shape: $(batch+1)$

### `start_pos`: tensor(int64)

Sequence position of each batch.

Shape: $(batch)$

### `max_seqlen`: scalar(int64)

Maximum sequence length of `query` and `key`, equal to `max(seqstarts[1:]-seqstarts[:batch])`. For parallel computing.

### `first_seqlen`: tensor(int64)

Prefill tokens length of each batch

Shape: $(batch)$

## Outputs

### `rotated_query`: tensor(T)

Query tensor after rotary position embedding.

Shape: $(seqstarts[batch], num\\_heads, head\\_dim)$

### `rotated_key`: tensor(T)

Key tensor after rotary position embedding .

Shape: $(seqstarts[batch], num\\_k\\_heads, head\\_dim)$

### `T`: float32, float16, int8