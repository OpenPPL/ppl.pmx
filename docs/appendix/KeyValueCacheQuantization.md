# KeyValueCache Quantization

## Linear Symmetry Quantization

For every input array, perform linear symmertry quantization, give the output quantized array and output scale. `quant_group` denotes the number of elements in input data. `quant_bit` denotes bits of quantization and determinates the number of steps.

$$scale={\rm MAX}(\frac{{\rm MAX}(input)}{2^{quant\_bit-1}-1}, eps)$$
$$output={\rm ROUND}(\frac{input}{scale})$$