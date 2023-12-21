# dynamic_batching.InsertEmbedding

The original definition of `InsertEmbedding` refers to [here](../InsertEmbedding.md).

The difference between `dynamic_batching.InsertEmbedding` and `InsertEmbedding` is that the sequence length, offset of each batch are different. 

`dynamic_batching.InsertEmbedding` uses `seqstarts`, `outstarts`, and `patstarts` to record the sequence begining position of each batch. 

## Inputs

### `X`: tensor(T)

Input sequences.

Shape: $(seqstarts[-1],E)$

### `P`: tensor(T)

patch sequences.

Shape: $(patstarts[-1],E)$

### `seqstarts`: tensor(int64)

`seqstarts[:batch]` contains the position of the first token in `X` and for each batch.
And `seqstarts[batch]` contains the total length of `X`.

Note that `seqstarts[b+1]-seqstarts[b]` can calculate out the sequence length of batch $b$.

Shape: $(batch+1)$

### `outstarts`: tensor(int64)

`outstarts[:batch]` contains the position of the first token in `Y` and for each batch.
And `outstarts[batch]` contains the total length of `Y`.

Note that `outstarts[b+1]-outstarts[b]` can calculate out the sequence length of batch $b$.

Usually, `outstarts[-1]=seqstarts[-1]+patstarts[-1]` or `outstarts[-1]=seqstarts[-1]+batch*patstarts[-1]` when $P$ is broadcasted.

Shape: $(batch+1)$

### `patstarts`: tensor(int64)

`patstarts[:batch]` contains the position of the first token in `P` and for each batch.
And `patstarts[batch]` contains the total length of `P`.

Note that `patstarts[b+1]-patstarts[b]` can calculate out the sequence length of batch $b$.

When size of `patstarts` is $(2)$, it will be broadcasted.

Shape: $(batch+1)$ or $(2)$ to be broadcasted.

### `offset`: tensor(int64)

Insert offset.

Shape: $(batch)$ or $(1)$ to be broadcasted.

### `max_outlen`: scalar(int64)

Maximum sequence length of `Y`, equal to `max(outstarts[1:]-outstarts[:batch])`. For parallel computing.

## Outputs

### `Y`: tensor(T)

Output sequences.

Shape: $(outstarts[-1],E)$, 

## Type Constraints

### `T1`: float32, float16