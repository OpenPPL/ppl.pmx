import argparse
import gc
import json
import os
import shutil
import warnings

from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sentencepiece.sentencepiece_model_pb2 as model

"""
Sample usage:

```
python convert_hf_weights_to_pmx.py \
    --model_path /path/to/tokenizer.model
    --config_path /path/to/tokenizer_config.json
    --output_path /output/path/to/tokenizer.model
```

we add added_tokens_decoder in config into the tokenizer.model


example json:

{
  "add_bos_token": true,
  "add_eos_token": false,
  "added_tokens_decoder": {
    "0": {
      "content": "<unk>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "1": {
      "content": "<s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "2": {
      "content": "</s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "92538": {
      "content": "<|plugin|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "92539": {
      "content": "<|interpreter|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "92540": {
      "content": "<|action_end|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "92541": {
      "content": "<|action_start|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "92542": {
      "content": "<|im_end|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "92543": {
      "content": "<|im_start|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    }
  },
  "additional_special_tokens": [
    "<|im_start|>",
    "<|im_end|>",
    "<|action_start|>",
    "<|action_end|>",
    "<|interpreter|>",
    "<|plugin|>"
  ],
  "auto_map": {
    "AutoTokenizer": [
      "tokenization_internlm2.InternLM2Tokenizer",
      "tokenization_internlm2_fast.InternLM2TokenizerFast"
    ]
  },
  "bos_token": "<s>",
  "chat_template": "{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
  "clean_up_tokenization_spaces": false,
  "decode_with_prefix_space": false,
  "eos_token": "</s>",
  "model_max_length": 1000000000000000019884624838656,
  "pad_token": "</s>",
  "sp_model_kwargs": null,
  "tokenizer_class": "InternLM2Tokenizer",
  "unk_token": "<unk>"
}

"""

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def added_tokens_decoder(model_path, config_path, output_path):
    assert os.path.isfile(model_path), model_path
    assert os.path.isfile(config_path), config_path

    m = model.ModelProto()
    m.ParseFromString(open(model_path, "rb").read())

    cfg = read_json(config_path)
    add_tokens = cfg.get('added_tokens_decoder', None)
    if add_tokens is not None:
        for key in add_tokens.keys():
            origin_word = m.pieces[int(key)].piece
            replace_word = add_tokens[key]['content']
            m.pieces[int(key)].piece = replace_word
            print(f"{key}: \"{origin_word}\" -> \"{m.pieces[int(key)].piece}\"")

    with open(output_path, 'wb') as f:
        f.write(m.SerializeToString())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        help="Location of tokenizer.model",
    )
    parser.add_argument(
        "--config_path",
        help="Location of tokenizer_config.json of HF model",
    )
    parser.add_argument(
        "--output_path",
        help="Location to output the modified tokenizer.model",
    )
    args = parser.parse_args()
    added_tokens_decoder(
        model_path=args.model_path,
        config_path=args.config_path,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()