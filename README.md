# KuaiSearch


<div align="center">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <a href="https://arxiv.org/abs/2602.11518"><img src="https://img.shields.io/badge/arXiv-2602.11518-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/benchen4395/KuaiSearch"><img src="https://img.shields.io/badge/🤗%20Dataset-KuaiSearch-blue" alt="HuggingFace Dataset"></a>
</div>

**KuaiSearch** is a large-scale e-commerce search dataset and full-stack benchmark system built from real user search interactions on the [Kuaishou](https://www.kuaishou.com) platform. It covers the three core stages of modern industrial search pipelines: **Recall**, **Relevance**, and **Ranking**. Each stage provides multiple algorithmic baselines, allowing researchers to systematically evaluate and compare methods

> 📄 **Paper**: [KuaiSearch: A Large-Scale E-Commerce Search Dataset for Recall, Ranking, and Relevance](https://arxiv.org/abs/2602.11518)
> Yupeng Li\*, Ben Chen\*, Mingyue Cheng, Zhiding Liu, Xuxin Zhang, Chenyi Lei, Wenwu Ou
>
> 🤗 **Dataset**: [huggingface.co/datasets/benchen4395/KuaiSearch](https://huggingface.co/datasets/benchen4395/KuaiSearch)

---

## Table of Contents

- [Overview](#overview)
- [Dataset Statistics](#dataset-statistics)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Recall](#recall)
  - [Relevance](#relevance)
  - [Ranking](#ranking)
- [Supported Models](#supported-models)
- [Benchmark Results](#benchmark-results)
- [Notes](#notes)
- [Citation](#citation)
- [References](#references)

---

## Overview

KuaiSearch provides a large-scale e-commerce search dataset together with a complete benchmark system covering three modular, independently trainable stages:

| Stage | Description | Methods |
|---|---|---|
| 🔍 **Recall** | Retrieve candidate documents from a large corpus | BM25, DPR (Dense Retrieval), Generative Retrieval (GR) |
| ✅ **Relevance** | Score query–document semantic relevance | Cross-Encoder, Bi-Encoder Embedding, GR |
| 📊 **Ranking** | Learn-to-rank candidates with user features | DNN, Wide&Deep, DCNv1, DCNv2, DIN |

KuaiSearch is, to the best of our knowledge, **the largest e-commerce search dataset currently available**, built upon real user search interactions from the Kuaishou platform. It retains authentic user queries and natural-language product texts, covers cold-start users and long-tail products, and spans all three key stages of the search pipeline.

---

## Dataset Statistics

### Scale Comparison with Existing Datasets

| Dataset | # Users | # Items | # Queries | Text Form |
|---|---|---|---|---|
| Amazon | 192,403 | 63,001 | 3,221 | text (heuristic queries) |
| JDsearch | 173,831 | 12,872,736 | 171,728 | anonymized |
| **KuaiSearch-Lite** | **102,086** | **6,634,118** | **555,553** | **text** |
| **KuaiSearch** | **331,930** | **18,605,582** | **2,574,949** | **text** |

### Data Schema

| Table | Size | Key Fields |
|---|---|---|
| **User** | 331,930 | `user_id`, `gender`, `age`, `location` |
| **Item** | 18,605,582 | `item_id`, `title`, `brand`, `seller`, `category L1/L2/L3` |
| **Recall** | 2,574,949 | `user_id`, `session_id`, `query`, `impressed_item_ids`, `clicked_item_ids`, `purchased_item_ids` |
| **Ranking** | 81,401,477 | `user_id`, user stats, `session_id`, `query`, `search_entrance`, recent clicked/purchased items, target item features, `is_clicked`, `is_purchased` |
| **Relevance** | 46,422 | `query`, `title`, `brand_name`, `seller_name`, `attribute`, `score` (0–3) |

> **KuaiSearch-Lite** is a lightweight subset designed for rapid model validation and ablation studies. All experiments in the paper are conducted on KuaiSearch-Lite.

---



## Installation

### Requirements

- Python 3.8+
- CUDA 11.7+ (recommended for GPU training)

### Install Dependencies

```bash
pip install -r requirements.txt
```


---

## Data Preparation

### Download the Dataset

The dataset is publicly available on HuggingFace:

```bash
# Install HuggingFace Hub CLI (if not already installed)
pip install huggingface_hub

# Download KuaiSearch dataset
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='benchen4395/KuaiSearch',
    repo_type='dataset',
    local_dir='./data'
)
"
```

Or manually download from: **https://huggingface.co/datasets/benchen4395/KuaiSearch**

---

## Usage

> ⚠️ All commands must be run from the `KuaiSearch/` project root.

---

### Recall

#### Step 1: Data Preprocessing

```bash
bash scripts/recall_data_process.sh
```

#### Method 1: BM25

```bash
bash scripts/recall_bm25_eval.sh
```
#### Method 2: DocT5Query

```bash
bash scripts/recall_doc2query.sh
bash scripts/recall_docT5query_eval.sh
```

#### Method 3: Dense Retrieval (DPR)

```bash
bash scripts/recall_dpr.sh
```

#### Method 4: Generative Retrieval (GR)

```bash
bash scripts/recall_gr.sh
```

---

### Relevance

#### Step 1: Data Preprocessing

```bash
bash scripts/relevance_data_process.sh
```

#### Method 1: Cross-Encoder

```bash
bash scripts/relevance_crossencoder.sh
```

#### Method 2: Bi-Encoder (Embedding)

```bash
bash scripts/relevance_embedding.sh
```

#### Method 3: Generative Relevance (GR)

```bash
bash scripts/relevance_gr.sh
```

---

### Ranking

#### Step 1: Data Preprocessing

```bash
bash scripts/ranking_data_process.sh
```

#### Step 2: Train

```bash
# Default model: DCNv1
bash scripts/ranking_train.sh
```

#### Switch Model

```bash
# Options: DNN | WideDeep | DCNv1 | DCNv2 | DIN
accelerate launch ranking/main.py --model DCNv2
```

---

## Benchmark Results

All experiments are conducted on **KuaiSearch-Lite**. Results are from the paper ([arXiv:2602.11518](https://arxiv.org/abs/2602.11518)).

### Recall Task (Recall@K / HitRate@K)

| Method | R@10 | HR@10 | R@20 | HR@20 | R@50 | HR@50 |
|---|---|---|---|---|---|---|
| BM25 | 0.0706 | 0.1001 | 0.1037 | 0.1427 | 0.1564 | 0.2088 |
| DocT5Query (BM25+Doc2Query) | 0.0784 | 0.1098 | 0.1156 | 0.1594 | 0.1772 | 0.2381 |
| DPR-ADE | 0.0818 | 0.1184 | 0.1254 | 0.1745 | 0.2026 | 0.2709 |
| **DPR-SDE** | **0.0826** | **0.1210** | **0.1293** | **0.1814** | **0.2079** | **0.2769** |
| DSI (GR) | 0.0623 | 0.0965 | 0.0892 | 0.1344 | 0.1369 | 0.2018 |
| LTRGR (GR) | 0.0688 | 0.1049 | 0.0986 | 0.1477 | 0.1501 | 0.2184 |

### Ranking Task (CTR Prediction)

| Method | Logloss ↓ | ROC-AUC ↑ |
|---|---|---|
| **DNN** | **0.1588** | 0.6258 |
| Wide & Deep | 0.1598 | 0.6217 |
| DCN | 0.1611 | 0.6194 |
| DCNv2 | 0.1603 | 0.6239 |
| **DIN** | 0.1606 | **0.6262** |

### Relevance Task (ROC-AUC / PR-AUC)

| Model | Size | ROC-AUC ↑ | PR-AUC ↑ |
|---|---|---|---|
| BGE-Base (Bi-Encoder) | 0.1B | 0.7475 | 0.5791 |
| BGE-Large (Bi-Encoder) | 0.32B | 0.7531 | 0.6052 |
| BERT-Chinese (Cross-Encoder) | 0.11B | 0.7606 | 0.6041 |
| BERT-Multilingual (Cross-Encoder) | 0.11B | 0.7737 | 0.6383 |
| XLM-RoBERTa-Base (Cross-Encoder) | 0.27B | 0.7941 | 0.6658 |
| XLM-RoBERTa-Large (Cross-Encoder) | 0.55B | 0.8005 | 0.6756 |
| Llama3.2-1B (GR + LoRA) | 1.0B | 0.7602 | 0.5927 |
| Llama3.2-3B (GR + LoRA) | 3.0B | 0.8093 | 0.6696 |
| Qwen3-0.6B (GR + LoRA) | 0.6B | 0.7994 | 0.6524 |
| **Qwen3-1.7B (GR + LoRA)** | **1.7B** | **0.8215** | **0.6966** |

---

## Supported Models

### Recall

| Model | Directory | Backbone | Key Feature |
|---|---|---|---|
| BM25 + Doc2Query | `recall/BM25/` | `google/mt5-base` | Sparse retrieval + pseudo-query expansion |
| DPR | `recall/dpr/` | `bert-base-chinese` | Bi-encoder dense retrieval, FAISS index |
| Generative Retrieval | `recall/GR/` | `MT5` | Seq2seq + Trie-constrained decoding |

### Relevance

| Model | Directory | Backbone | Key Feature |
|---|---|---|---|
| Cross-Encoder | `relevance/crossencoder/` | `bert-base-chinese` | Concat input, LoRA optional |
| Bi-Encoder | `relevance/embedding/` | `XLM-RoBERTa-Base` | Contrastive, offline doc encoding |
| GR (LLM) | `relevance/GR/` | `Llama-3.2-3B` | Generative scoring, LoRA fine-tuning |

### Ranking

| Model | `--model` Flag | Description |
|---|---|---|
| DNN | `DNN` | Baseline deep neural network |
| Wide & Deep | `WideDeep` | Wide linear + deep neural components |
| DCN v1 | `DCNv1` | Deep & Cross Network with explicit feature crosses |
| DCN v2 | `DCNv2` | Improved DCN with matrix-based cross layers |
| DIN | `DIN` | Deep Interest Network with attention over user history |

All ranking models share the same feature schema:

| Feature Group | Fields |
|---|---|
| **User features** | `user_id`, `gender`, `age` |
| **Item features** | `item_id`, `category_level1/2/3_id`, `item_text` |
| **Query features** | Query text (pre-encoded via BERT embeddings) |
| **Behavior features** | User interaction history sequence (used in DIN attention) |
| **Text embeddings** | Pre-computed title/query embeddings (`.npy` memmap) |

---

## Notes

1. **Working directory**: Always run scripts from `KuaiSearch/`. All paths in configs and scripts are relative to the project root.
2. **GPU memory**: Adjust `batch_size` in scripts to fit your GPU VRAM. Multi-GPU training is supported via `accelerate` and `torchrun`.
3. **Model checkpoints**: Trained models are saved to the `model/` subdirectory of each module (configurable via `output_dir` in config files).
4. **Mixed precision**: Ranking trainer uses `bf16` by default; change via `--mixed_precision` flag. Relevance/Recall trainers use standard HuggingFace `TrainingArguments`.
5. **Reproducibility**: All modules fix `seed=42` by default.
6. **FAISS variant**: Install `faiss-gpu` for GPU-accelerated retrieval, or `faiss-cpu` for CPU-only environments.
7. **LLM access**: The GR relevance module requires access to `meta-llama/Llama-3.2-3B`. Ensure you have accepted the model license on HuggingFace and set `HF_TOKEN` if needed.

---

## Citation

If you use KuaiSearch in your research, please cite our paper:

```bibtex
@article{li2026kuaisearch,
  title     = {KuaiSearch: A Large-Scale E-Commerce Search Dataset for Recall, Ranking, and Relevance},
  author    = {Yupeng Li and Ben Chen and Mingyue Cheng and Zhiding Liu and Xuxin Zhang and Chenyi Lei and Wenwu Ou},
  journal   = {arXiv preprint arXiv:2602.11518},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.11518}
}
```

---

## References

- [KuaiSearch Paper (arXiv:2602.11518)](https://arxiv.org/abs/2602.11518)
- [KuaiSearch Dataset (HuggingFace)](https://huggingface.co/datasets/benchen4395/KuaiSearch)
- [DPR: Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [Generative Retrieval (DSI)](https://arxiv.org/abs/2202.05144)
- [Doc2Query: Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375)
- [Cross-Encoder for Re-ranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [DCN v2: Deep & Cross Network](https://arxiv.org/abs/2008.13535)
- [DIN: Deep Interest Network](https://arxiv.org/abs/1706.06978)
- [Wide & Deep Learning](https://arxiv.org/abs/1606.07792)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [sentence-transformers](https://www.sbert.net/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
