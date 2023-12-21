# InsertEmbedding

Insert patch sequences of embedding into source sequences of embedding.

Mark $X$ as the source sequence $[x_0, x_1, \cdots, x_S]$
and $P$ as the patch sequence $[p_0, p_1, \cdots, p_I]$, where $x_i$ and $p_i$ are embedding vectors with length $E$.

Then we insert $P$ into $X$ by offset $k$, the output will become
$$Y = [x_0, x_1, \cdots, x_k, p_0, p_1, \cdots, p_I, x_{k+1}, x_{k+2}, \cdots, x_S]$$

## Inputs

### `X`: tensor(T)

Input sequences.

Shape: $(batch,seqlen,E)$

### `P`: tensor(T)

patch sequences.

Shape: $(batch,patlen,E)$ or $(patlen,E)$

when $P$ is a 2d tensor, is will be broadcasted.

### `offset`: scalar(int64)

Insert offset.

## Outputs

### `Y`: tensor(T)

Output sequences.

Shape: $(batch,seqlen+patlen,E)$

## Type Constraints

### `T1`: float32, float16