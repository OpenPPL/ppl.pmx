from typing import List
import sys
import os
import torch
import json

from .Model import Transformer, TensorDumper

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from ModelUtils import __TextGenerator__

from llama.modeling.static_batching.Pipeline import LLaMA as Mistral