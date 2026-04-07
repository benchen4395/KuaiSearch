# KuaiSearch


<div align="center">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <a href="https://arxiv.org/abs/2602.11518"><img src="https://img.shields.io/badge/arXiv-2602.11518-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/benchen4395/KuaiSearch"><img src="https://img.shields.io/badge/🤗%20Dataset-KuaiSearch-blue" alt="HuggingFace Dataset"></a>
</div>

**KuaiSearch** is a large-scale e-commerce search dataset and full-stack benchmark system built from real user search interactions on the [Kuaishou](https://www.kuaishou.com) platform. It covers the three core stages of modern industrial search pipelines: **Recall**, **Relevance**, and **Ranking**. Each stage provides multiple algorithmic baselines, allowing researchers to systematically evaluate and compare methods

## ! All the codes and datasets are now public !

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
- CUDA 11.7+

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

