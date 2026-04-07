import json
import numpy as np
import torch
from tqdm import tqdm

# ====== Path Configuration - Update these paths to your actual file locations ======
CORPUS_PATH = "data/corpus.jsonl"                         # Product corpus containing item_id / item_title / brand_name / seller_name etc.
CORPUS_IDS_PATH = "recall/data/embeddings/item_code.pt"        # Your saved corpus_ids (codes+dup) file - do NOT use wildcards

# Train data paths
QUERIES_TSV_TRAIN = "recall/data/train.queries.tsv"              # Format: qid\tquery
QRELS_TSV_TRAIN = "recall/data/train.qrels.tsv"                  # Format: qid Q0 docid rel

# Test data paths
QUERIES_TSV_TEST = "recall/data/test.queries.tsv"                # Format: qid\tquery
QRELS_TSV_TEST = "recall/data/test.qrels.tsv"                    # Format: qid Q0 docid rel

# Output file paths
ITEM2CODE_OUTPUT_JSONL = "recall/data/item_text_codes.json"             # Mapping: item → code
QUERY2CODE_OUTPUT_JSONL_TRAIN = "recall/data/query_item_code_train.json"  # Train mapping: query → item_code
QUERY2CODE_OUTPUT_JSONL_TEST = "recall/data/query_item_code_test.json"    # Test mapping: query → item_code


def load_corpus_texts_and_ids(path: str):
    """
    Maintain consistency with original encoding logic:
    - Read item_title from corpus.jsonl
    - Concatenate into text string
    - Record item_id to maintain 1:1 alignment with text
    
    Important: Only count samples with valid text (matches original encoding behavior).
    
    Returns:
        Tuple of (texts_list, item_ids_list)
    """
    texts = []
    item_ids = []  # Item IDs aligned with texts list
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = obj.get("item_title", "") or ""
            # Only keep non-empty components
            parts = [p for p in [title] if p]
            if not parts:
                # Skip samples with no valid text (consistent with encoding logic)
                continue

            text = " ".join(parts)
            texts.append(text)

            item_id = obj.get("item_id")
            # Convert to string to align with qrels docid format; None for missing item_id
            item_ids.append(str(item_id) if item_id is not None else None)

    print(f"[load_corpus_texts_and_ids] Number of usable items = {len(texts)}")
    return texts, item_ids


def load_queries(queries_tsv: str) -> dict:
    """
    Load queries from TSV file.
    
    Format:
        qid\tquery_text
        
    Returns:
        Dictionary mapping query ID to query text {qid: query_text}
    """
    qid2query = {}
    with open(queries_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            qid, query = parts[0], parts[1]
            qid2query[qid] = query
    print(f"[load_queries] Total queries loaded = {len(qid2query)}")
    return qid2query


def build_query2code(
    qrels_path: str,
    queries_path: str,
    output_path: str,
    itemid2idx: dict,
    corpus_ids: np.ndarray,
    split_name: str = "train",
):
    """
    Build query-to-code mappings with different handling for train/test splits:
    - Train: Write individual records (one per relevant item)
    - Test: Aggregate codes by query ID (list of codes per query)
    
    Args:
        qrels_path: Path to qrels TSV file
        queries_path: Path to queries TSV file
        output_path: Path for output JSON file
        itemid2idx: Mapping from item ID to corpus index
        corpus_ids: Numpy array of corpus codes (shape: [num_items, 4])
        split_name: Split type ("train" or "test")
    """
    print(f"[{split_name}] Loading queries from {queries_path} ...")
    qid2query = load_queries(queries_path)

    print(f"[{split_name}] Reading qrels and collecting records...")

    # --------- Key Difference 1: Data structure for train vs test ---------
    if split_name == "test":
        # Test: qid -> list of code strings (aggregated)
        qid2codes = {}
    else:
        # Train: list of individual records
        records = []

    # Counters for statistics
    n_total = 0
    n_skip_no_query = 0
    n_skip_no_item = 0
    n_skip_nonrel = 0

    with open(qrels_path, "r", encoding="utf-8") as f_in:
        for line in tqdm(f_in, desc=f"Processing qrels ({split_name})"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            qid, _, docid, rel = parts
            n_total += 1

            # Validate relevance score
            try:
                rel_val = float(rel)
            except ValueError:
                rel_val = 0.0
            if rel_val <= 0:
                n_skip_nonrel += 1
                continue

            # Check if query exists
            query = qid2query.get(qid)
            if query is None:
                n_skip_no_query += 1
                continue

            # Map document ID to corpus index
            idx = itemid2idx.get(str(docid))
            if idx is None:
                n_skip_no_item += 1
                continue

            # Extract code components [c0, c1, c2, dup]
            row = corpus_ids[idx]
            c0, c1, c2, dup = [int(x) for x in row.tolist()]
            code_str = f"{c0} {c1} {c2} {dup}"

            # --------- Key Difference 2: Record handling ---------
            if split_name == "test":
                # Test: Aggregate codes by query ID
                if qid not in qid2codes:
                    qid2codes[qid] = []
                qid2codes[qid].append(code_str)
            else:
                # Train: Add individual record
                records.append({
                    "text": query,
                    "output": code_str,
                })

    # --------- Key Difference 3: File writing logic ---------
    with open(output_path, "w", encoding="utf-8") as f_out:
        if split_name == "test":
            # Test: One record per query with list of codes
            out_records = []
            for qid, codes in qid2codes.items():
                query = qid2query.get(qid)
                if query is None:
                    continue
                out_records.append({
                    "text": query,
                    "output": codes,      # List of code strings
                })
            json.dump(out_records, f_out, ensure_ascii=False, indent=2)
        else:
            # Train: Write individual records
            json.dump(records, f_out, ensure_ascii=False, indent=2)

    # Print statistics
    written_count = len(qid2codes) if split_name == 'test' else len(records)
    print(f"[{split_name}] Statistics - Total: {n_total}, "
          f"Written: {written_count}, "
          f"Skipped (non-relevant): {n_skip_nonrel}, "
          f"Skipped (no query): {n_skip_no_query}, "
          f"Skipped (no item): {n_skip_no_item}")


def main():
    """Main execution pipeline"""
    # 1. Load corpus IDs (codes + duplication flag)
    print("[main] Loading corpus_ids...")
    corpus_ids = torch.load(CORPUS_IDS_PATH)
    
    # Convert to numpy array (handle both tensor and array inputs)
    if isinstance(corpus_ids, torch.Tensor):
        corpus_ids = corpus_ids.cpu().numpy()
    else:
        corpus_ids = np.array(corpus_ids)
    print("[main] corpus_ids shape:", corpus_ids.shape)
    # Expected shape: (num_items, 4) → [c0, c1, c2, dup]

    # 2. Load corpus texts and item IDs (must match encoding order exactly)
    print("[main] Loading corpus texts & item_ids...")
    item_texts, item_ids = load_corpus_texts_and_ids(CORPUS_PATH)

    # Critical validation: text count must match corpus IDs count
    assert len(item_texts) == corpus_ids.shape[0], \
        f"Text count ({len(item_texts)}) does not match corpus_ids row count ({corpus_ids.shape[0]})! " \
        f"Verify encoding/filtering logic consistency."

    # 3. Write item-to-code mapping
    print(f"[main] Writing item2code JSON to {ITEM2CODE_OUTPUT_JSONL}")
    item_records = []
    for text, row in tqdm(
        zip(item_texts, corpus_ids),
        total=len(item_texts),
        desc="Processing item2code"
    ):
        c0, c1, c2, dup = [int(x) for x in row.tolist()]
        item_records.append({
            "text": text,
            "output": f"{c0} {c1} {c2} {dup}",
        })
    
    with open(ITEM2CODE_OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        json.dump(item_records, f_out, ensure_ascii=False, indent=2)
    
    print("[main] Item-to-code mapping completed.")

    # 4. Build item ID to corpus index mapping (for qrels to code lookup)
    print("[main] Building item_id -> index mapping...")
    itemid2idx = {}
    for idx, iid in enumerate(item_ids):
        if iid is None:
            continue
        itemid2idx[iid] = idx
    print(f"[main] itemid2idx mapping size = {len(itemid2idx)}")

    # 5. Build train query-to-code mapping
    build_query2code(
        qrels_path=QRELS_TSV_TRAIN,
        queries_path=QUERIES_TSV_TRAIN,
        output_path=QUERY2CODE_OUTPUT_JSONL_TRAIN,
        itemid2idx=itemid2idx,
        corpus_ids=corpus_ids,
        split_name="train",
    )

    # 6. Build test query-to-code mapping
    build_query2code(
        qrels_path=QRELS_TSV_TEST,
        queries_path=QUERIES_TSV_TEST,
        output_path=QUERY2CODE_OUTPUT_JSONL_TEST,
        itemid2idx=itemid2idx,
        corpus_ids=corpus_ids,
        split_name="test",
    )


if __name__ == "__main__":
    main()