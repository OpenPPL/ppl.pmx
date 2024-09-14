import sys
import os

import torch
from torch import nn
import torch.distributed as dist

from typing import Mapping, Any, Optional

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

import torch_function as OPMX
from ModelParams import ModelParams
import ModelUtils
from ModelParallel import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from ModelLayers import SkipRMSNorm

TensorDumper = ModelUtils.__TensorDumper__()

# TODO
