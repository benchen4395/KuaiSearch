import csv
import faiss       
import numpy as np 
from tqdm import tqdm
from transformers import BertModel,BertTokenizer
import torch
import time
import transformers
from typing import Dict, List
import json
transformers.logging.set_verbosity_error()
def normalize_query(question: str) -> str:
    question = question.replace("’", "'")
    return question

def load_msmarco_queries(queries_file: str) -> Dict[str, str]:
    """TSV: qid \t query_text"""
    queries = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            qid, qtext = row[0], row[1]
            queries[qid] = normalize_query(qtext)
    return queries


def load_msmarco_qrels(qrels_file: str) -> Dict[str, Dict[str, int]]:
    """
    TSV: qid \t ? \t docid \t rel
    返回: {qid: {docid: rel_int}}
    """
    from collections import defaultdict
    qrels = defaultdict(dict)
    with open(qrels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[:4]
            qrels[qid][docid] = int(rel)
    return dict(qrels)


def load_msmarco_corpus_ids(corpus_file: str) -> List[str]:
    """
    {"id": "DOCID", "text": "..."}

    """
    doc_ids = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="loading corpus doc_ids"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_ids.append(str(obj["id"]))
    return doc_ids

def compute_msmarco_metrics(
    qids: List[str],
    I: np.ndarray,
    doc_ids: List[str],
    qrels: Dict[str, Dict[str, int]],
    top_k_list: List[int] = [10, 100],
):

    from collections import defaultdict
    import numpy as np

    max_k = I.shape[1]
    top_k_list = sorted(set(k for k in top_k_list if k <= max_k))

    metrics = defaultdict(list) 
    qid_missing=0
    rel_missing=0
    for qi, qid in enumerate(qids):
        if qid not in qrels:
            qid_missing+=1
            continue

        rel_docs = {docid for docid, rel in qrels[qid].items() if rel > 0}
        if qi==0:
            print("rel_docs", rel_docs, type(next(iter(rel_docs))))
        if not rel_docs:
            rel_missing+=1
            continue

        retrieved_internal = I[qi]  # shape: [top_k_max]
        retrieved_docids = [doc_ids[idx] for idx in retrieved_internal]
        if qi==0:    
            print("retrieval_docs", retrieved_docids)
        for k in top_k_list:
            cand_docids = retrieved_docids[:k]
            cand_set = set(cand_docids)
            # Recall@k
            hit_rels = len(cand_set & rel_docs)
            recall_k = hit_rels / len(rel_docs)
            metrics[f"recall@{k}"].append(recall_k)

            # Precision@k
            precision_k = hit_rels / len(cand_docids) if cand_docids else 0.0
            metrics[f"precision@{k}"].append(precision_k)

            # MRR@k
            mrr_k = 0.0
            for rank, docid in enumerate(cand_docids, start=1):
                if docid in rel_docs:
                    mrr_k = 1.0 / rank
                    break
            metrics[f"mrr@{k}"].append(mrr_k)

            #HR@k
            hr_k = 1.0 if hit_rels > 0 else 0.0
            metrics[f"hr@{k}"].append(hr_k)

    avg_metrics = {name: (float(np.mean(vals)) if len(vals) > 0 else 0.0)
                   for name, vals in metrics.items()}
    return avg_metrics

def evaluate_model(args):
    queries_dict = load_msmarco_queries(args.queries_file)
    qrels = load_msmarco_qrels(args.qrels_file)


    qids = list(queries_dict.keys())
    queries_all = [queries_dict[qid] for qid in qids]


    queries_batches = [
        queries_all[i:i + args.encoding_batch_size]
        for i in range(0, len(queries_all), args.encoding_batch_size)
    ]
    print("queries--",queries_batches[0])
    print("qids--",qids[:args.encoding_batch_size])

    doc_ids = []

    embedding_dimension = 768
    index = faiss.IndexFlatIP(embedding_dimension)


    print("building FAISS index from doc embeddings...")
    for shard_idx in tqdm(range(args.num_shards), desc="loading doc embedding shards"):
        shard_path = f"{args.embedding_dir}/corpus_shard_{shard_idx}.npy"
        ids_path = f"{args.embedding_dir}/corpus_shard_{shard_idx}_ids.npy"

        data = np.load(shard_path)
        ids = np.load(ids_path)

        
        index.add(data)
        doc_ids.extend([str(x) for x in ids])
    print("docids",doc_ids[:10],type(doc_ids[0]))

    query_encoder = BertModel.from_pretrained(args.pretrained_model_path, add_pooling_layer=False)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_encoder.to(device).eval()


    print("encoding queries...")
    query_embeddings_list = []
    with torch.no_grad():
        for batch_queries in tqdm(queries_batches, desc="encoding batches"):
            
            inputs = tokenizer(
                batch_queries,
                max_length=64,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            outputs = query_encoder(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            query_embeddings_list.append(emb.cpu().numpy())

    query_embeddings = np.concatenate(query_embeddings_list, axis=0)

    assert query_embeddings.shape[0] == len(qids)


    print("searching index...", end=" ")
    start_time = time.time()
    top_k = args.top_k
    _, I = index.search(query_embeddings, top_k)
    print(f"takes {time.time() - start_time:.2f} s")

    metrics = compute_msmarco_metrics(
        qids=qids,
        I=I,
        doc_ids=doc_ids,
        qrels=qrels,
        top_k_list=[10,20,50,100],  
    )

    print("\n=== MS MARCO evaluation metrics ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_file",
        type=str,
        default="data/corpus.jsonl",
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default="recall/data/test.queries.tsv",
    )
    parser.add_argument(
        "--qrels_file",
        type=str,
        default="recall/data/test.qrels.tsv",
    )
    
    parser.add_argument("--encoding_batch_size", type=int, default=512)
    parser.add_argument("--num_shards", type=int, default=8,
                        help="doc embedding 分成多少个 shard_n.npy")
    parser.add_argument("--embedding_dir",default="./doc_embedding_s")
    parser.add_argument("--pretrained_model_path",default="./model_signle/best/encoder")
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    evaluate_model(args)