#!/bin/bash
python recall/BM25/bm25.py \
  --corpus_file data/corpus.jsonl \
  --queries_file recall/data/test.queries.tsv \
  --qrels_file recall/data/test.qrels.tsv \
  --pseudo_query_file recall/data/pseudo_query/pseudo_query.all.jsonl \
  --index_name kuaisearch \
  --top_k 100 \
  --cutoffs 10 20 50 100
