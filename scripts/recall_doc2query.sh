#!/bin/bash
torchrun --nproc_per_node=8 recall/BM25/doc2query.py recall/BM25/config.json
