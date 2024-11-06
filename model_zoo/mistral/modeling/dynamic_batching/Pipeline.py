from typing import List
import sys
import os
import torch
import json

from .Model import Transformer, TensorDumper

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from ModelUtils import __Tokenizer__, __TextGenerator__
from llama.modeling.dynamic_batching.Pipeline import LLaMA as Mistral


class BatchState:
    def __init__(self):
        self.tid = 0
        self.input_tokens = []
        self.start_pos = 0
        self.cache_starts = 0
        self.output_tokens = []
        self.is_decoding = False