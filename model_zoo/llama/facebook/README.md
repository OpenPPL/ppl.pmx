# About

Modeling LLaMA from fackbook official, compatible with LLaMA2.

Model Source: [facebookresearch/llama](https://github.com/facebookresearch/llama/)

Model Download: [LLaMA](https://github.com/facebookresearch/llama/tree/llama_v1#llama) [LLaMA2](https://github.com/facebookresearch/llama/#download)

# Usage example

## Download model from facebook

Click the "Model Download" link above and follow the documents provided by facebook.

## Install requirements

```bash
pip install -r requirements.txt
```

## Convert model params

We recommand converting facebook's `params.json` to opmx's `params.json` before running or exporting the model. Because it is needed by the model merging and spliting tools.

```bash
ConvertParamsToPmx.py -ckpt_dir <llama_dir> --tokenizer_path <llama_tokenizer_dir>/tokenizer.model
```

You can find `opmx_params.json` in `<llama_dir>` after the conversion.

## Run model for testing

You can test wether the model is corret before exporting.

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node <MP> \
Demo.py --ckpt_dir <llama_dir> \
--tokenizer_path <llama_tokenizer_dir>/tokenizer.model \
--fused_qkv 1 --fused_kvcache 1 --auto_causal 1 \
--quantized_cache 1 --dynamic_batching 1 \
# --dump_tensor_path <dir_to_dump_input_outputs>
# --dump_steps <step numbers separated by comma, 0,1,2...>
```

If you want to dump some input/output data, uncomment the line above. And **careful for your disk space**.

## Export

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node <MP> \
Export.py --ckpt_dir <llama_dir> \
--tokenizer_path <llama_tokenizer_dir>/tokenizer.model \
--export_path <path_to_store_exported_llama7b>
--fused_qkv 1 --fused_kvcache 1 --auto_causal 1 \
--quantized_cache 1 --dynamic_batching 1 \
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| LLaMA-7B     | 1  |
| LLaMA-13B    | 2  |
| LLaMA-33B    | 4  |
| LLaMA-65B    | 8  |
| LLaMA2-7B     | 1  |
| LLaMA2-70B    | 8  |
