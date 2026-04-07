#!/bin/bash
torchrun --nproc_per_node=8 recall/GR/train.py \
  --corpus_file recall/data/item_text_codes.json \
  --train_file recall/data/query_item_code_train.json \
  --test_file recall/data/query_item_code_test.json \
  --code_file recall/data/embeddings/item_code.pt
