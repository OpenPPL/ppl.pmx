
# dynamic_batching.PositionIndex

`dynamic_batching.PositionIndex`, uses `seqstarts` to record the sequence begining position of each batch.

Generating tokens' position indeces of each batch $b$, where $p=start\\_pos[b]$ and $e=p+seqstarts[b+1]-seqstarts[b]$

$$ postion\\_idx = (p, p + 1, \cdots, e)$$

## Inputs

### `sequences`: tensor(int64)

input sequences

Shape: `seqstarts[batch]`

### `seqstarts`: tensor(int64)

`seqstarts[:batch]` contains the position of the first token in `sequences` of each batch. And `seqstarts[batch]` contains the total length of `sequences`.

Note that `seqstarts[b+1]-seqstarts[b]` can calculate out the sequence length of batch $b$.

Shape: $(B+1)$

### `start_pos`: tensor(int64)

Sequence position of each batch.

Shape: $(B)$

### `max_seqlen`: scalar(int64)

Maximum sequence length of `sequences`, equal to `max(seqstarts[1:]-seqstarts[:batch])`. For parallel computing.

## Outputs

### `position_idx`: tensor(int64)

Shape: `seqstarts[batch]`
