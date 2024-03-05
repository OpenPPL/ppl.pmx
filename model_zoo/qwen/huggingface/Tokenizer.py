import sys
import os
from tokenizers import Tokenizer as TokenizerFast
from logging import getLogger
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

from model_zoo.ModelUtils import __Tokenizer__

logger = getLogger()


class Tokenizer(__Tokenizer__):
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = TokenizerFast.from_file(model_path)

        self.bos_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|endoftext|>"
        self.im_start_token = "<|im_start|>"
        self.im_end_token = "<|im_end|>"


        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.get_vocab_size()
        self.bos_id: int = self.sp_model.token_to_id(self.bos_token)
        self.eos_id: int = self.sp_model.token_to_id(self.eos_token)
        self.pad_id: int = self.sp_model.token_to_id(self.pad_token)
        self.im_start_id: int = self.sp_model.token_to_id(self.im_start_token)
        self.im_end_id: int = self.sp_model.token_to_id(self.im_end_token)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s).ids
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t, skip_special_tokens=False)

    def vocab_size(self):
        return self.n_words

    def get_bos_id(self):
        return self.bos_id

    def get_eos_id(self):
        return self.eos_id

    def get_pad_id(self):
        return self.pad_id
    

def make_context(
    tokenizer: Tokenizer,
    query: str,
):
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role, content):
        return f"{role}\n{content}", tokenizer.encode(role) + nl_tokens + tokenizer.encode(content)

    system = "You are a helpful assistant."
    system_text, system_tokens_part = _tokenize_str("system", system)
    system_tokens = [tokenizer.im_start_id] + system_tokens_part + [tokenizer.im_end_id]

    raw_text = ""
    context_tokens = []

    context_tokens = system_tokens + context_tokens
    raw_text = f"{tokenizer.im_start_token}{system_text}{tokenizer.im_end_token}" + raw_text
    context_tokens += (
        nl_tokens
        + [tokenizer.im_start_id]
        + _tokenize_str("user", query)[1]
        + [tokenizer.im_end_id]
        + nl_tokens
        + [tokenizer.im_start_id]
        + tokenizer.encode("assistant")
        + nl_tokens
    )
    raw_text += f"\n{tokenizer.im_start_token}user\n{query}{tokenizer.im_end_token}\n{tokenizer.im_start_token}assistant\n"

    return raw_text, context_tokens


def decode_context(
    tokens: List[int],
    *,
    tokenizer: Tokenizer,
    raw_text_len: int,
    context_length: int,
):
    eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id]
    eod_token_idx = context_length
    for eod_token_idx in range(context_length, len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            break
    
    trim_decode_tokens = tokenizer.decode(tokens[:eod_token_idx])[raw_text_len:]
    trim_decode_tokens = trim_decode_tokens.strip()

    return trim_decode_tokens