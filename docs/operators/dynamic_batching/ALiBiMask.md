
# dynamic_batching.ALiBiMask

The original definition of `ALiBi` refers to [here](../ALiBi.md).
`dynamic_batching.ALiBi`, uses `seqstarts` and `kvstarts` to record the sequence begining position of each batch. The mask is similar to a block diagonal matrix. We generate the mask for each batch and fill it in the corresponding position. The following code shows the calculation process.

```python
alibi_mask = torch.zeros([seqstarts[-1], kvstarts[-1]], dtype=data_type)

seqlens = seqstarts[1:] - seqstarts[:-1]
kvlens = kvstarts[1:] - kvstarts[:-1]
for batch_idx, seqlen in enumerate(seqlens):
    kvlen = kvlens[batch_idx]
    seqbeg = seqstarts[batch_idx]
    seqend = seqstarts[batch_idx+1]
    kvbeg = kvstarts[batch_idx]
    kvend = kvstarts[batch_idx+1]

    tmp_alibi_mask = torch.full((seqlen, kvlen), float('inf'), dtype=data_type)
    # generate masks for each batch, the process is the same as static batch alibi
    for i in range(xx):
        for j in range(xx):
            ...

    alibi_mask[seqbeg:seqend, kvbeg:kvend] = tmp_alibi_mask

alibi_mask = alibi_mask.unsqueeze(0).expand(num_heads, -1, -1)
# alibi_mask shape -> (num_heads, sum(seqlens), sum(kvlens))
# slopes_m shape -> (num_heads, 1, 1)
alibi_mask = slopes_m * alibi_mask
```
## Attributes/Parameters

### `num_heads`: int

Number of heads

### `data_type`: int

Data type of ALiBi mask

## Inputs

### `seqstarts`: tensor(int64)

`seqstarts[:B]` contains the position of the first token in `query` for each batch. And `seqstarts[B]` contains the total length of `query`.

Note that `seqstarts[b+1]-seqstarts[b]` can calculate out the sequence length of batch $b$.

Shape: $(B+1)$

### `kvstarts`: tensor(int64)

`kvstarts[:B]` contains the position of the first token in `key` and `value` for each batch. And `kvstarts[B]` contains the total length of `key` and `value`.

Note that `kvstarts[b+1]-kvstarts[b]` can calculate out the key and value length of batch $b$.

Shape: $(B+1)$

### `attn_mask`(optional): tensor(T)

Optional custom mask.
`seqlens=seqstarts[1:]-seqstarts[:B]` is a sequence contains length of `query` for each batch.
`kvlens=kvstarts[1:]-kvstarts[:B]` is a sequence contains length of `key` and `value` for each batch.

Shape: $(num\\_heads, {\rm sum}(seqlens), {\rm sum}(kvlens))$ or $({\rm sum}(seqlens), {\rm sum}(kvlens))$

## Outputs

### `alibi_mask`: tensor(T)

Output mask of ALiBi.

Shape: $(num\\_heads, {\rm sum}(seqlens), {\rm sum}(kvlens))$ 
## Type Constraints

### `T`: float32, float16
