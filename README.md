# About PMX

PPL Model Exchange (PMX) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. PMX provides an open source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types.

Currently PMX focus on the capabilities and hardware friendliness needed for Large Language Model(LLM) inferencing.

# Operator spec

Table of Contents: [Link](docs/OperatorsTOC.md)

About add new operator: [Link](docs/AddNewOp.md)

About update an operator's version: [Link](docs/UpdateOp.md)

# Use PMX Python API

PMX provides functional API based on `torch.autograd.Function`.

Clone the PMX repo, and import `torch_function` like this:

```python
import pmx_llm.torch_function as PMX
```

And then use it as Pytorch's functional API:

```python
norm, skip_out = PMX.skip_rms_norm(x, weight, skip_in, -1, eps)
```

We can use these API in pytorch to custom your own model.

All PMX function could be exported as custom operators by `torch.onnx.export`.

# Model Zoo

Some opensource model are provided in our model zoo.

Currently models:

- [LLaMA 1/2](model_zoo/llama)

# Exporting LLM model in the model zoo

- [Facebook LLaMA](model_zoo/llama/facebook/README.md)
