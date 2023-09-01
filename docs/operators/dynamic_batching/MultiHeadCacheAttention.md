# dynamic_batching.MultiHeadCacheAttention

The original definition of `dynamic_batching.MultiHeadAttention` refers to [here](MultiHeadAttention.md).

The original definition of `dynamic_batching.KeyValueCache` refers to [here](KeyValueCache.md).

For `dynamic_batching.MultiHeadCacheAttention`, it is just fuse `dynamic_batching.MultiHeadAttention` and `dynamic_batching.KeyValueCache` togather.

## Attributes/Parameters

### `num_heads`: int

Number of heads

### `head_dim`: int

Dimension of each head, where $head\_dim * num\_heads = hidden\_dim$

### `is_causal`: bool

Whether do casual mask when sequence length > 1. If `is_causal` is `True`, length of `query`, `key` and `value` within every batchs must be equal.

### `num_kv_heads`: int(default: 0)

For [Grouped-Query Attention](https://arxiv.org/pdf/2305.13245.pdf). If `num_kv_heads` and `num_heads` are not equal, we should repeat key and value `num_heads/num_kv_heads` times before applying ${\rm MHA}$ for each token. `num_heads` must be divisible by `num_kv_heads`. Default is 0, and at this point, `num_heads` is used as `num_kv_heads`.

### `num_layer`: int(default: 1)

Number of attention layers.

### `layer_idx`: int(default: 0)

Attention layer index for cache and scale.

### `quant_bit`: int(default: 0)

Quantize bit for cache compression. For example, 8 means int8 compression. `0` means disabled.

### `quant_group`: int(default: 8)

Quantize scale shared group size. $2^n$ and $n > 2$ is recommanded for hardware implementation.

### `cache_mode`: int(default: 0)

Define cache indexing mode. Default is zero.
- When `cache_mode` is `0`, cache is indexed by offset mode. Shape of `cachestarts` is $(B)$. For each batch $b$, `cachestarts[b]` mapping cache begining index in $MaxT$ of `cache` and `scale`. Note that `cachestarts[b+1]-cachestarts[b]` can **not** calculate out the cache length of batch $b$.
- When `cache_mode` is `1`, cache is indexed by table mode. Shape of `cachestarts` is $(kvstarts[B])$. For each batch $b$, `cachestarts[kvstarts[b]:kvstarts[b+1]]` contains cache indices of tokens in $MaxT$ of `cache` and `scale`.

### `cache_layout`: int(default: 0)

Define data layout of `cache` and `scale`. Default is zero.

Meaning of numbers:
- `0`: $cache(MaxT,L,2,H,Dh)$ and $scale(MaxT,L,2,H,Dh/quant\_group)$
- `1`: $cache(L,MaxT,2,H,Dh)$ and $scale(L,MaxT,2,H,Dh/quant\_group)$
- `2`: $cache(L,H,MaxT,2,Dh)$ and $scale(L,H,MaxT,2,Dh/quant\_group)$
- `3`: $cache(2,L,H,MaxT,Dh)$ and $scale(2,L,H,MaxT,Dh/quant\_group)$

## Inputs

### `query`: tensor(T1)

Input Query tensor

Shape: $(seqstarts[B], num\_heads, head\_dim)$

### `current_key`: tensor(T1)

Input Key tensor

Shape: $(seqstarts[B], num\_kv\_heads, head\_dim)$

### `current_value`: tensor(T1)

Input Value tensor

Shape: $(seqstarts[B], num\_kv\_heads, head\_dim)$

### `seqstarts`: tensor(int64)

`seqstarts[:B]` contains the position of the first token in `query` for each batch. And `seqstarts[B]` contains the total length of `query`.

Note that `seqstarts[b+1]-seqstarts[b]` can calculate out the sequence length of batch $b$.

Shape: $(B+1)$

### `kvstarts`: tensor(int64)

`kvstarts[:B]` contains the position of the first token in `key = cat(past_key, current_key)` and `value = cat(past_value, current_value)` for each batch, where `key` and `value` are originally provided by operator `KeyValueCache`. And `kvstarts[B]` contains the total length of `key` and `value`.

Note that `kvstarts[b+1]-kvstarts[b]` can calculate out the key and value length of batch $b$.

Shape: $(B+1)$

### `cachestarts`: tensor(int64)

Indexing cache position in $MaxT$ of `cache` and `scale`. Behavior is determinated by `cache_mode`.

Shape: $(B)$ or $(kvstarts[B])$

### `start_pos`: tensor(int64)

Sequence position where `current_key` and `current_value` begining to store of each batch.

Shape: $(B)$

### `decoding_batches`: scalar(int64)

Describe how many batches in front are being decoded, those who are not need causal mask.

### `max_seqlen`: scalar(int64)

Maximum sequence length of `query`, equal to `max(seqstarts[1:]-seqstarts[:B])`. For parallel computing.

### `max_kvlen`: scalar(int64)

Maximum sequence length of `key` and `value`, equal to `max(kvstarts[1:]-kvstarts[:B])`. For parallel computing.

### `cache`: tensor(T2)

Shape: Determinated by `cache_layout`.

Contains key and value caches of attention layer. When `cache_layout` is `0`, subspace $(:,:,0,:,:,:)$ contains key caches and subspace $(:,:,1,:,:,:)$ contains value caches. **Data in this tensor will be modified.**

### `scale`(optional): tensor(T3)

Shape: determinate by `cache_layout`.

Contains key and value cache quantize scales of attention layer. When `cache_layout` is `0`, subspace $(:,:,0,:,:,:)$ contains key cache scales and subspace $(:,:,1,:,:,:)$ contains value cache scales. Must appear if `quant_bit` is not zero. **Data in this tensor will be modified.**

### `attn_mask`(optional): tensor(T1)

Optional custom mask.
`seqlens=seqstarts[1:]-seqstarts[:B]` is a sequence contains length of `query` for each batch.
`kvlens=kvstarts[1:]-kvstarts[:B]` is a sequence contains length of `key` and `value` for each batch.

Shape: $(num\_heads, {\rm sum}(seqlens), {\rm sum}(kvlens))$ or $({\rm sum}(seqlens), {\rm sum}(kvlens))$

## Outputs

### `attn_output`: tensor(T1)

Output feature of attention result

Shape: $(seqstarts[B], num\_heads, head\_dim)$

## Type Constraints

### `T1`: float32, float16

### `T2`: float32, float16, int8, int4

### `T3`: float32, float16
