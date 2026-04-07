#!/bin/bash
torchrun --nproc_per_node=8 relevance/GR/train.py relevance/GR/config.json
