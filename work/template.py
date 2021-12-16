import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import copy
import numpy as np
import pickle
from itertools import chain

from type_def import *
from models.model_utils import get_init_params
from evaluate.evaluator import BaseEvaluator
from utils import tools
from utils.data import SimpleDataset
from analysis.recorder import NaiveRecorder

"""
Model
    input(train):
    input(eval):
    output(train):
    output(eval):
Loss
    input:
    output:
Evaluator
    input:
    output:
"""

# model

# loss

# evaluator

# dataset

# sample_to_eval_format

# eval_output_to_read_format

# train_output_to_loss_format

# UseModel


model_registry = {
    "model": None,
    'loss': None,
    "evaluator": None,
    "dataset": None,
    "sample_to_eval_format": None,
    'eval_output_to_read_format': None,
    "train_output_to_loss_format": None,
    "UseModel": None,
    "args": [],
    "recorder": None,
}


