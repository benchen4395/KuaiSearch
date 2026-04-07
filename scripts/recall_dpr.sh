#!/bin/bash
accelerate launch recall/dpr/main.py \
  --corpus_path data/corpus.jsonl \
  --queries_file recall/data/test.queries.tsv \
  --qrels_file recall/data/test.qrels.tsv
