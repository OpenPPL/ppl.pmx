
# ALiBiSlope

[Attention with Linear Biases ](https://ofir.io/train_short_test_long.pdf) (ALiBi)
does not add positional embeddings
to word embeddings; instead, it biases query-key attention scores with a penalty
that is proportional to their distance.
ALiBi is defined as:

$$
softmax\left(\mathbf{q}_i \mathbf{K}^{\top}+m \cdot[-(i-1), \ldots,-2,-1,0]\right)
$$

![ALiBi](ALiBi.jpeg)


The figrue offers a visualization.
ALiBi adds a constant bias (right) to each attention score ($\mathbf{q}_i \cdot \mathbf{k}_j$, left). m is a head-specific scalar that is set and not learned throughout training. 
The following code shows how to generate $m$.

`num_heads` is the number of heads in transformer model.

```python
def get_slopes(num_heads):
    result = []
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    for n in range(1, closest_power_of_2+1):
        result.append(2**(-8 * n / closest_power_of_2))
    if closest_power_of_2 < num_heads:
        for n in range(1, 2*(num_heads-closest_power_of_2)+1, 2):
            result.append(2**(-4 * n / closest_power_of_2))
    return tmp
```

> NOTE: This OP's result will be used by Attention to fuse ALiBiMask.

## Attributes/Parameters

### `num_heads`: int

Number of heads

## Inputs

> None

## Outputs

### `slopes`: tensor(float)

Shape: $(num\\_heads)$
