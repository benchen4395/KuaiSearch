import json
import random
from typing import Dict, List, Tuple

# ========== Configuration ==========
QUERIES_FILE = "recall/data/train.queries.tsv"
QRELS_FILE = "recall/data/train.qrels.tsv"
CORPUS_FILE = "data/corpus.jsonl"   # Your product corpus JSONL file
OUTPUT_QA_FILE = "recall/data/train_dpr.json"

# Number of negative samples to sample per query (random from all non-positive samples)
NUM_NEG_PER_QUERY = 1

random.seed(42)  # Fixed seed for reproducibility


# ====== Utility Functions: Data Loading ======

def load_queries_from_tsv(path: str) -> Dict[str, str]:
    """
    Load queries from TSV file:
    Format per line: qid \t query_text
    Returns: Dictionary mapping query ID to query text {qid: query_text}
    """
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            qid, query_text = parts[0], parts[1]
            queries[qid] = query_text
    return queries


def load_qrels_from_tsv(path: str) -> Dict[str, Dict[str, int]]:
    """
    Load relevance judgments (qrels) from TSV file:
    Format per line: qid \t Q0 \t docid \t relevance_score
    Returns: Nested dictionary {qid: {docid: relevance_score}}
    """
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[:4]
            rel = int(rel)
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = rel
    return qrels


def safe_get(doc: dict, key: str) -> str:
    """
    Safely retrieve value from dictionary with fallback.
    Returns empty string if key is missing/None, and converts value to string.
    
    Args:
        doc: Input dictionary to retrieve value from
        key: Target key to look up
        
    Returns:
        String value (empty string if missing/None)
    """
    val = doc.get(key, "")
    if val is None:
        return ""
    return str(val)


def build_passage_text(doc: dict) -> str:
    """
    Construct passage text from product document.
    Extracts and returns item_title field (main text content).
    
    Args:
        doc: Product document dictionary
        
    Returns:
        Cleaned item title text
    """
    text = safe_get(doc, "item_title")
    return text


def load_corpus(path: str) -> Tuple[Dict[str, dict], List[str]]:
    """
    Load product corpus from JSONL file:
    Each line contains a JSON object with item_id and other product fields.
    
    Returns:
        Tuple containing:
        - doc_dict: Dictionary mapping document ID to full document {docid(str): original_doc}
        - all_docids: List of all valid document IDs in corpus
    """
    doc_dict: Dict[str, dict] = {}
    all_docids: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Corpus JSON parse error at line {ln}, skipped")
                continue

            item_id = obj.get("item_id")
            if item_id is None:
                continue

            docid = str(item_id)
            doc_dict[docid] = obj
            all_docids.append(docid)

    return doc_dict, all_docids


# ====== Main Logic: Build QA Dataset ======

def build_qa_data(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    corpus: Dict[str, dict],
    all_docids: List[str],
    num_neg_per_query: int = 10,
    max_pos_per_query: int = None,  # Optional: limit number of positive samples per query
) -> List[dict]:
    """
    Build QA dataset samples for DPR training.
    
    Each sample contains:
    {
        "query": str,
        "positive_item": [{"text": str}, ...],
        "negative_item": [{"text": str}, ...]
    }
    
    Args:
        queries: Dictionary of {qid: query_text}
        qrels: Nested dictionary of relevance judgments {qid: {docid: rel_score}}
        corpus: Dictionary of {docid: full_document}
        all_docids: List of all document IDs in corpus
        num_neg_per_query: Number of negative samples to generate per query
        max_pos_per_query: Maximum positive samples to use per query (None = use all)
        
    Returns:
        List of QA samples ready for DPR training
    """
    qa_samples: List[dict] = []

    # Optimize lookups: list for random sampling, set for membership checks
    all_docids_list = all_docids
    all_docids_len = len(all_docids_list)
    all_docids_set = set(all_docids_list)

    # Process only queries with relevance judgments
    for qid, rels_for_q in qrels.items():
        # Skip if query text is missing
        query_text = queries.get(qid)
        if not query_text:
            continue

        # Get positive document IDs (relevance score > 0)
        pos_docids = [docid for docid, rel in rels_for_q.items() if rel > 0]

        if not pos_docids:
            continue  # Skip queries with no positive samples

        # Optional: limit number of positive samples per query
        if max_pos_per_query is not None and len(pos_docids) > max_pos_per_query:
            pos_docids = pos_docids[:max_pos_per_query]

        # Build positive contexts
        positive_ctxs = []
        for docid in pos_docids:
            doc = corpus.get(docid)
            if doc is None:
                continue
            text = build_passage_text(doc)
            if not text:
                continue
            positive_ctxs.append({"text": text})

        if not positive_ctxs:
            continue  # Skip if no valid positive samples

        # -------- Efficient negative sampling: global random + rejection sampling --------
        pos_set = set(pos_docids)
        negative_ctxs = []
        chosen_negs = set()  # Prevent duplicate negative samples

        # Maximum trials to avoid infinite loops (edge case: small corpus)
        max_trials = num_neg_per_query * 20 if num_neg_per_query > 0 else 0
        trials = 0

        while len(negative_ctxs) < num_neg_per_query and trials < max_trials:
            trials += 1
            # Randomly sample document from corpus
            docid = all_docids_list[random.randint(0, all_docids_len - 1)]

            # Skip: positive docs / already chosen negatives / missing docs
            if docid in pos_set or docid in chosen_negs:
                continue

            doc = corpus.get(docid)
            if doc is None:
                continue

            text = build_passage_text(doc)
            if not text:
                continue

            chosen_negs.add(docid)
            negative_ctxs.append({"text": text})

        # Skip if insufficient negative samples
        if not negative_ctxs:
            continue

        # Create final sample
        sample = {
            "query": query_text,
            "positive_item": positive_ctxs,
            "negative_item": negative_ctxs,
        }
        qa_samples.append(sample)

    return qa_samples


def main():
    """Main execution pipeline"""
    print("Loading queries...")
    queries = load_queries_from_tsv(QUERIES_FILE)
    print(f"Loaded {len(queries)} queries")

    print("Loading relevance judgments (qrels)...")
    qrels = load_qrels_from_tsv(QRELS_FILE)
    print(f"Loaded qrels for {len(qrels)} queries")

    print("Loading product corpus...")
    corpus, all_docids = load_corpus(CORPUS_FILE)
    print(f"Loaded {len(all_docids)} documents in corpus")

    print("Building QA dataset for DPR training...")
    qa_data = build_qa_data(
        queries=queries,
        qrels=qrels,
        corpus=corpus,
        all_docids=all_docids,
        num_neg_per_query=NUM_NEG_PER_QUERY,
    )
    print(f"Built {len(qa_data)} QA samples")

    print(f"Saving output to {OUTPUT_QA_FILE}...")
    with open(OUTPUT_QA_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()