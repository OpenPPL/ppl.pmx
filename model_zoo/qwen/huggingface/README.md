# Usage example

## Download model from Hugging Face

Download the model file from the [Hugging Face](https://huggingface.co/Qwen/Qwen-7B-Chat).

## Convert model params

Due to the inconsistency with the implementation of Hugging Face's RotaryPositionEmbedding function, we need to convert the weight parameters.

```
python ConvertWeightToOpmx.py --input_dir <hf_model_dir> --output_dir <pmx_model_dir>
```

you can find opmx model file in `<pmx_model_dir>` after the conversion.

## Spliting model

[SplitModel.py](https://github.com/openppl-public/ppl.opmx/blob/master/model_zoo/llama/modeling/SplitModel.py) is a Python script that splits a OPMX model's weights into multiple shards. The script reads a OPMX model's weights and divides them into specified shards, creating separate models for each shard.

```bash
python SplitModel.py --input_dir <input_directory_path> --num_shards <number_of_shards> --output_dir <output_directory_path>
```

- `input_dir`: Location of OPMX model weights. Ensure that the directory contains the file 'opmx_params.json'.
- `num_shards`: Number of shards to split the weights into.
- `output_dir`: Directory to save the resulting shard models.

## Merging model

[MergeModel.py](https://github.com/openppl-public/ppl.opmx/blob/master/model_zoo/llama/modeling/MergeModel.py) is a Python script that merges weights of a sharded model into a single model. The script reads the weights from multiple shards of a model and creates a consolidated model with combined weights.

```bash
python MergeModel.py --input_dir <input_directory_path> --num_shards <number_of_shards> --output_dir <output_directory_path>
```

- `input_dir`: Location of model weights, containing multiple files ending in '.pth'.
- `num_shards`: Number of shards to merge.
- `output_dir`: Directory to write the merged OPMX model.

## Testing Model

The `Demo.py` script provides functionality to test the model for correctness before exporting.

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node $num_gpu Demo.py --ckpt_dir <llama_dir> --tokenizer_path <llama_tokenizer_dir>/tokenizer.model --fused_qkv 1 --fused_kvcache 1 --auto_causal 1 --quantized_cache 1 --dynamic_batching 1
```

- `OMP_NUM_THREADS`: This parameter determines the number of OpenMP threads. It is set to 1 to prevent excessive CPU core usage. Each PyTorch process opens an OpenMP thread pool, and setting it to 1 avoids occupying too many CPU cores.
- `--nproc_per_node`: Specifies the number of model slices per node.

## Exporting Model

To export a model, you will use the `Export.py` script provided. Here's an example command for exporting a 13B model with 1 GPU:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node $num_gpu Export.py --ckpt_dir <llama_dir> --tokenizer_path <llama_tokenizer_dir>/tokenizer.model --fused_qkv 1 --fused_kvcache 1 --auto_causal 1 --quantized_cache 1 --dynamic_batching 1 --export_path <export_dir>
```

Make sure to replace `$num_gpu` with the actual number of GPUs you want to use.

## Generating Test Data

This script demonstrates how to generate test data for steps 0, 1, and 255 using the specified command.

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node $num_gpu Demo.py --ckpt_dir <llama_dir> --tokenizer_path <llama_tokenizer_dir>/tokenizer.model --fused_qkv 1 --fused_kvcache 1 --auto_causal 1 --quantized_cache 1 --dynamic_batching 1 --seqlen_scale_up 1 --max_gen_len 256 --dump_steps 0,1,255 --dump_tensor_path <dump_dir>  --batch 1
```

- `seqlen_scale_up`: Scale factor for input byte size (sequence length scaled up by 8).
- `max_gen_len`: Specifies the maximum generated output length in bytes.
- `dump_steps`: Steps at which to dump the test data.
- `dump_tensor_path`: Path to store the dumped test data.
- `batch`: Specifies the batch size for data processing.

Make sure to replace `<llama_dir>` , `<llama_tokenizer_dir>` and `<dump_tensor_path>`with the actual directory paths in your environment.
