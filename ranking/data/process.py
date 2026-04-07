import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

"""
Merged vector encoding script:
1. Extracts queries from samples_.reindex.jsonl and encodes them into query_emb.npy and session_id2idx.json
2. Reads item_titles from corpus_.reindex.jsonl and encodes them into item_title_emb.npy and item_id2idx.json
Reuses the same model and tokenizer to avoid redundant loading.
"""

# ====================
# Configuration Parameters
# ====================
DATA_DIR = "data"
SAMPLES_FILE = os.path.join(DATA_DIR, "rank.jsonl")
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")

QUERY_EMB_NPY = os.path.join('./', "query_emb.npy")
SESSION_ID2IDX = os.path.join('./', "session_id2idx.json")

ITEM_EMB_NPY = os.path.join('./', "item_title_emb.npy")
ITEM_ID2IDX = os.path.join('./', "item_id2idx.json")

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MAX_LEN = 32
BATCH_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collect_session_queries():
    """Collect Session ID to Query mapping (takes the first query of each session)"""
    session2query = {}
    if not os.path.exists(SAMPLES_FILE):
        print(f"[WARN] File not found: {SAMPLES_FILE}. Skipping Query processing.")
        return session2query
        
    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj["session_id"]
            q = obj["query"]
            if sid not in session2query:
                session2query[sid] = q
    return session2query


def collect_items():
    """Collect Item ID and Item Title"""
    items = []
    if not os.path.exists(CORPUS_FILE):
        print(f"[WARN] File not found: {CORPUS_FILE}. Skipping Item processing.")
        return items
        
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append((obj["item_id"], obj.get("item_title", "")))
    return items


def encode_and_save(texts, ids, tokenizer, model, npy_path, json_path, desc):
    """General text encoding and saving logic"""
    if not texts:
        return

    emb_list = []
    id2idx = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
            batch_texts = texts[start:start + BATCH_SIZE]
            batch_ids = ids[start:start + BATCH_SIZE]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)

            outputs = model(**enc)
            # Extract [CLS] vector
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu()

            for i, unique_id in enumerate(batch_ids):
                idx = len(emb_list)
                id2idx[unique_id] = idx
                emb_list.append(cls_emb[i].numpy())

    # Stack and convert to float16 to save space
    emb_mat = np.stack(emb_list, axis=0).astype(np.float16)
    np.save(npy_path, emb_mat)

    # Save mapping dictionary
    with open(json_path, "w", encoding="utf-8") as fidx:
        json.dump(id2idx, fidx, ensure_ascii=False, indent=2)

    print(f"[INFO] Successfully saved {os.path.basename(npy_path)}, shape={emb_mat.shape}")
    print(f"[INFO] Successfully saved {os.path.basename(json_path)} with {len(id2idx)} entries\n")


def main():
    print(f"[INFO] Loading model {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    model.eval()
    print("[INFO] Model loaded successfully!\n")

    # ====================
    # 1. Process Queries (Sessions)
    # ====================
    print(f"[INFO] Start processing Queries -> {SAMPLES_FILE}")
    session2query = collect_session_queries()
    if session2query:
        print(f"[INFO] Found {len(session2query)} unique sessions")
        session_ids = list(session2query.keys())
        queries = [session2query[sid] for sid in session_ids]
        encode_and_save(queries, session_ids, tokenizer, model, QUERY_EMB_NPY, SESSION_ID2IDX, "Encoding Queries")

    # ====================
    # 2. Process Items (Corpus)
    # ====================
    print(f"[INFO] Start processing Items -> {CORPUS_FILE}")
    items_data = collect_items()
    if items_data:
        print(f"[INFO] Found {len(items_data)} items")
        item_ids = [x[0] for x in items_data]
        titles = [x[1] for x in items_data]
        encode_and_save(titles, item_ids, tokenizer, model, ITEM_EMB_NPY, ITEM_ID2IDX, "Encoding Items")

    print("[INFO] All encoding tasks completed!")


if __name__ == "__main__":
    main()