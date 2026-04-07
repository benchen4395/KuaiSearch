#!/bin/bash
torchrun --nproc_per_node=8 relevance/embedding/train.py relevance/embedding/config.json
