from type_def import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import numpy as np
from itertools import chain
import pickle
import copy

from evaluate.evaluator import BaseEvaluator, tokenspans2events, EE_F1Evaluator, CcksEvaluator
from work.EE.PLMEE.sentence_representation_layer import SentenceRepresentation, TriggeredSentenceRepresentation
from work.EE.PLMEE.trigger_extraction_model import TriggerExtractionLayer_woSyntactic
from work.EE.PLMEE.argument_extraction_model import ArgumentExtractionModel_woSyntactic
from work.EE.EE_utils import *
from work.EE.JointEE_rebuild import jointee_settings
from utils import tools
from utils.data import SimpleDataset
from analysis.recorder import NaiveRecorder
from models.model_utils import get_init_params


class JointEE(nn.Module):
    def __init__(self,
                 plm_path=jointee_settings.plm_path,
                 n_head=jointee_settings.n_head,
                 d_head=jointee_settings.d_head,
                 hidden_dropout_prob=0.3,
                 plm_lr=EE_settings.plm_lr,
                 others_lr=EE_settings.others_lr,
                 trigger_threshold=jointee_settings.trigger_extraction_threshold,
                 argument_threshold=jointee_settings.argument_extraction_threshold):
        super(JointEE, self).__init__()
        self.init_params = get_init_params(locals())  # 默认模型中包含这个东西。也许是个不好的设计
        # store init params
        self.plm_path = plm_path
        self.n_head = n_head
        self.d_head = d_head
        self.hidden_dropout_prob = hidden_dropout_prob
        self.plm_lr = plm_lr
        self.others_lr = others_lr
        self.trigger_threshold = trigger_threshold
        self.argument_threshold = argument_threshold

        # initiate network structures
        #   Sentence Representation
        self.sentence_representation = SentenceRepresentation(self.plm_path)
        self.hidden_size = self.sentence_representation.hidden_size
        #   Trigger Extraction
        self.tem = TriggerExtractionLayer_woSyntactic(
            num_heads=self.n_head,
            hidden_size=self.hidden_size,
            d_head=self.d_head,
            dropout_prob=self.hidden_dropout_prob)
        #   Triggered Sentence Representation
        self.trigger_sentence_representation = TriggeredSentenceRepresentation(self.hidden_size)
        #   Argument Extraction
        self.aem = ArgumentExtractionModel_woSyntactic(
            n_head=self.n_head,
            d_head=self.d_head,
            hidden_size=self.hidden_size,
            dropout_prob=self.hidden_dropout_prob)

        self.trigger_spanses = []
        self.argument_spanses = []