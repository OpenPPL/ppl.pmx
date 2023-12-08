import sys
import os
from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

from ModelUtils import __Tokenizer__

logger = getLogger()

class Tokenizer(__Tokenizer__):
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")
        
        self.bos_token = '<sop>'
        self.eos_token='<eop>'
        self.pad_token='<pad>'

        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        self.special_ids = {}
        for i, token in enumerate(special_tokens):
            self.special_ids[self.sp_model.vocab_size() + i] = token

        self.n_words: int = self.sp_model.vocab_size() + len(self.special_ids)
        self.bos_id = self.sp_model.PieceToId(self.bos_token)
        self.eos_id = self.sp_model.PieceToId(self.eos_token)
        self.pad_id = self.sp_model.PieceToId(self.pad_token)
        self.gmask_id = self.sp_model.vocab_size() + 1  # 64790
        self.sop_id = self.sp_model.vocab_size() + 3    # 64792

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - gMASK ID: {self.gmask_id} - SOP ID: {self.sop_id} ")
        
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        t = [self.gmask_id, self.sop_id] + t
        return t

    def decode(self, t: List[int]) -> str:
        # 跳过special tokens
        if isinstance(t, int):
            if t in self.special_ids:
                return ""

        filtered_tokens = []
        for token in t:
            if token in self.special_ids:
                continue
            filtered_tokens.append(token)
        return self.sp_model.decode(filtered_tokens)

    def vocab_size(self):
        return self.n_words

    def get_bos_id(self):
        return self.bos_id

    def get_eos_id(self):
        return self.eos_id
    
    def get_pad_id(self):
        return self.pad_id
    
    def get_gmask_id(self):
        return self.gmask_id
