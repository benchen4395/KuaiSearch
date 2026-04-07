#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import os
import numpy as np
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)
from sentence_transformers.losses import ContrastiveLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import SentenceEvaluator

from transformers import EarlyStoppingCallback
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import *
from transformers import set_seed
import torch.distributed as dist
from parameters import load_parameters
set_seed(42)
model_args, data_args, training_args = load_parameters()
training_args.batch_sampler=BatchSamplers.NO_DUPLICATES

train_dataset = load_dataset(data_args.train_file)
eval_dataset = load_dataset(data_args.validation_file)

model = SentenceTransformer(model_args.model_name_or_path)
model.max_seq_length = data_args.max_seq_length
loss = ContrastiveLoss(model)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

if dist.is_available() and dist.is_initialized():
    dist.barrier() 
if trainer.is_world_process_zero():
    trainer.save_model(training_args.output_dir+'/best')
    print(f"Done. Saved to: {training_args.output_dir}")
    test(
        data_args.train_file,
        data_args.validation_file,
        model,
        query_prompt_name="query",
        optimize_metric="macro_f1",
        batch_size=training_args.per_device_eval_batch_size,
    )

