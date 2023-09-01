# ParallelEmbedding

A simple lookup table that stores embeddings of a fixed dictionary and size.

This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.

$$output=gather(input, W)$$

If `paddding_idx` is non negative:

$$output[where(output==padding\\_idx)] = 0$$

$N$ is `num_embeddings`, $E$ is `embedding_dim`, and $W$ is embedding weight.

Tensor parallel is performed along the $E$ dimension, and weight $W$ will be divided into $TPsize$ parts along the $E$ dimension: $W(N,E) \rightarrow W(N,[E_0,E_1,\cdots,E_t ])$.

After the devices in the same communicate world perform embedding, all gather $TPsize$ parts of result to get the final result $(*,E)$.

$TPsize$ means communicate world size of tensor parallel.

## Attributes/Parameters

### `num_embeddings`: int

Number of embedding vector in embedding weight, marked as $N$.

### `embedding_dim`: int

Dimension of embedding weight, marked as $E$. It should be $E_d$ for each device $d$ when $TPsize > 1$.

### `padding_idx `: int(default: -1)

Enable padding when the value is non negative. The embedding vector at `padding_idx` will fill with zeros.

### `max_norm `: float(default: 0)

If greater than 0, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.

### `norm_type `: float(default: 2)

The p of the p-norm to compute for the max_norm option.

## Inputs

### `ids`: tensor(T1)

Input token ids, the value of ids should between `0` and `num_embeddings-1`.

Shape: $(*)$, where $∗$ means any number of dimensions including none.

### `W`(constant): tensor(T2)

Embedding weight.

Shape: $(N, E)$ or $(N, E_d)$ for each device $d$ when $TPsize > 1$.

## Outputs

### `output`: tensor(T2)

Shape: $(*, E)$

## Type Constraints

### `T1`: int32

### `T2`: float32, float16
