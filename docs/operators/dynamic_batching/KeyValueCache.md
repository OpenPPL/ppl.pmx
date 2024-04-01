# dynamic_batching.KeyValueCache

The original definition of `KeyValueCache` refers to [here](../KeyValueCache.md).

The difference between `dynamic_batching.KeyValueCache` and `KeyValueCache` is that the sequence length, the value of `start_pos` and the length of key and value caches of each batch are different. 

`dynamic_batching.KeyValueCache` uses `seqstarts` and `kvstarts` to record the sequence begining position of each batch. And the ability to map batches to different locations in the cache is provided by `cachestarts`.

In the description below
 - $L$ is number of attention layers(`num_layer`)
 - $B$ is batch size
 - $MaxT$ is max tokens length `cache` could hold(i.e, it could be over 10,000,000 in some case)
 - $MaxP$ is the max number of pages of sequences in [Paged Attention](https://arxiv.org/abs/2309.06180) mode.
 - $H$ is `num_heads` of transformer
 - $Dh$ is `dims_per_head` or `head_dim` of transformer.

> NOTE：`cache` and `scale` are used as in-out tensor, so it is recommended to use them as model inputs, and let the user set the shape by themselves (mainly because `MaxT` need to be configured separately).

## Attributes/Parameters

### `num_layer`: int(default: 1)

Number of attention layers.

### `layer_idx`: int(default: 0)

Attention layer index for cache and scale.

### `quant_bit`: int(default: 0)

Quantize bit for cache compression. For example, 8 means int8 compression. `0` means disabled.

### `quant_group`: int(default: 8)

Quantize scale shared group size. $2^n$ and $n > 2$ is recommanded for hardware implementation.

### `num_repeat`: int(default: 1)

For [Grouped-Query Attention](https://arxiv.org/pdf/2305.13245.pdf). Repeat key and value `num_repeat` time on axis `num_heads` to construct an input compatiltable with non-grouped MultiHeadAttention.

### `cache_mode`: int(default: 0)

Define cache indexing mode. Default is zero.
- When `cache_mode` is `0`, `cache` is indexed by offset mode. Shape of `cachestarts` is $(B)$. For each batch $b$, `cachestarts[b]` mapping cache begining index in $MaxT$ of `cache` and `scale`. Note that `cachestarts[b+1]-cachestarts[b]` can **not** calculate out the cache length of batch $b$.
- When `cache_mode` is `1`, `cache` is indexed by page table mode, which called Paged Attention. Shape of `cachestarts` is $(B, MaxP)$. For each batch $b$, `cachestarts[b, :]` contains pages' begining index in $MaxT$ of `cache` and `scale`.<br>
Example for `batch = 2, page_size = 256`:
$$cachestarts=[[0,256,\cdots],[1024,2048,\cdots]]$$

### `cache_layout`: int(default: 0)

Define data layout of `cache` and `scale`. Default is zero.

Meaning of numbers:
- `0`: $cache(MaxT,L,2,H,Dh)$ and $scale(MaxT,L,2,H,Dh/quant\\_group)$
- `1`: $cache(L,MaxT,2,H,Dh)$ and $scale(L,MaxT,2,H,Dh/quant\\_group)$
- `2`: $cache(L,2,MaxT,H,Dh)$ and $scale(L,2,MaxT,H,Dh/quant\\_group)$
- `3`: $cache(L,2,H,MaxT,Dh)$ and $scale(L,2,H,MaxT,Dh/quant\\_group)$

### `page_size`: int(default: 128)

Page size in Paged Attention(when `cache_mode` is `1`)

## Inputs

### `current_key`: tensor(T1)

Shape: $(seqstarts[B],H,Dh)$

### `current_value`: tensor(T1)

Shape: $(seqstarts[B],H,Dh)$

### `seqstarts`: tensor(int64)

`seqstarts[:B]` contains the position of the first token in `current_key` and `current_value` for each batch.
And `seqstarts[B]` contains the total length of `current_key` and `current_value`.

Note that `seqstarts[b+1]-seqstarts[b]` can calculate out the sequence length of batch $b$.

Shape: $(B+1)$

### `kvstarts`: tensor(int64)

`kvstarts[:B]` contains the position of the first token in `key` and `value` for each batch.
And `kvstarts[B]` contains the total length of `key` and `value`.

Note that `kvstarts[b+1]-kvstarts[b]` can calculate out the key and value length of batch $b$.

Shape: $(B+1)$

### `cachestarts`: tensor(int64)

Indexing cache position in $MaxT$ of `cache` and `scale`. Behavior is determinated by `cache_mode`.

Shape: $(B)$ or $(B, MaxP)$

### `start_pos`: tensor(int64)

Sequence position where `current_key` and `current_value` begining to store of each batch.

Shape: $(B)$

### `max_seqlen`: scalar(int64)

Maximum sequence length of `current_key` and `current_value`, equal to `max(seqstarts[1:]-seqstarts[:B])`. For parallel computing.

### `max_kvlen`: scalar(int64)

Maximum sequence length of `key` and `value`, equal to `max(kvstarts[1:]-kvstarts[:B])`. For parallel computing.

### `cache`: tensor(T2)

Shape: Determinated by `cache_layout`.

Contains key and value caches of attention layer. When `cache_layout` is `0`, subspace $(:,:,0,:,:,:)$ contains key caches and subspace $(:,:,1,:,:,:)$ contains value caches. **Data in this tensor will be modified.**

### `scale`(optional): tensor(T3)

Shape: determinate by `cache_layout`.

Contains key and value cache quantize scales of attention layer. When `cache_layout` is `0`, subspace $(:,:,0,:,:,:)$ contains key cache scales and subspace $(:,:,1,:,:,:)$ contains value cache scales. Must appear if `quant_bit` is not zero. **Data in this tensor will be modified.**

## Outputs

### `key`: tensor(T1)

Shape: $(kvstarts[B],H*num\\_repeat,Dh)$

Packed current key and all pass key. If `quant_bit` is not `0`, it should be decompressed.

### `value`: tensor(T1)

Shape: $(kvstarts[B],H*num\\_repeat,Dh)$

Packed contains current value and all pass value. If `quant_bit` is not `0`, it should be decompressed.

## Type Constraints

### `T1`: float32, float16, int8

### `T2`: float32, float16, int8, int4

### `T3`: float32, float16
