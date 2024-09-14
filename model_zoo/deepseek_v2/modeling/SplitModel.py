import argparse
import gc
import json
import os
import shutil
import warnings

import torch

from pathlib import Path


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

# TODO
