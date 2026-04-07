import json
from collections import defaultdict
from typing import Dict, List, Tuple

# ========= Config =========
INPUT_FILE = "data/recall.jsonl"  # Path to your JSONL log file

# Output files (TSV format)
TRAIN_QUERIES_FILE = "recall/data/train.queries.tsv"
TRAIN_QRELS_FILE = "recall/data/train.qrels.tsv"
TEST_QUERIES_FILE = "recall/data/test.queries.tsv"
TEST_QRELS_FILE = "recall/data/test.qrels.tsv"


def get_interacted_items(rec: dict) -> List[int]:
    """
    Collect interacted item IDs from click and purchase behavior.
    Both clicked and purchased items are treated as relevant (rel=1).
    
    Args:
        rec: Single record dict from JSONL file
        
    Returns:
        List of unique interacted item IDs (clicked + purchased)
    """
    clicked = rec.get("clicked_item_ids") or []
    purchased = rec.get("purchased_item_ids") or []

    # Union of clicked and purchased items (remove duplicates)
    items = list(set(clicked + purchased))
    return items


def process_file(
    input_path: str,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Parse JSONL file and build train/test splits using native 'split' field.
    
    Args:
        input_path: Path to input JSONL file
        
    Returns:
        Tuple containing:
        - train_queries: {session_id: query_text}
        - train_qrels: {session_id: {item_id: relevance_score}}
        - test_queries: {session_id: query_text}
        - test_qrels: {session_id: {item_id: relevance_score}}
    """
    train_queries: Dict[str, str] = {}
    train_qrels: Dict[str, Dict[str, int]] = defaultdict(dict)

    test_queries: Dict[str, str] = {}
    test_qrels: Dict[str, Dict[str, int]] = defaultdict(dict)

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] JSON parse error at line {line_num}, skipped.")
                continue

            # Extract core fields from record
            session_id = rec.get("session_id")
            query_text = rec.get("query")
            split = rec.get("split")  # Native split label (train/test)

            # Validate required fields
            if not session_id or not query_text or not split:
                print(f"[WARN] Missing core fields (session_id/query/split) at line {line_num}, skipped.")
                continue

            # Convert session_id to string to avoid type issues
            qid = str(session_id)

            # Get interacted items (clicked + purchased)
            interacted_items = get_interacted_items(rec)
            if not interacted_items:
                # Only keep sessions with interaction behavior
                continue

            # Assign to train/test split based on native split field
            if split.lower() == "train":
                # Overwrite if duplicate session_id exists (keep last occurrence)
                train_queries[qid] = query_text
                for item_id in interacted_items:
                    docid = str(item_id)
                    train_qrels[qid][docid] = 1  # Click/purchase = relevant (rel=1)
            elif split.lower() == "test":
                test_queries[qid] = query_text
                for item_id in interacted_items:
                    docid = str(item_id)
                    test_qrels[qid][docid] = 1
            else:
                # Skip records with invalid split values
                print(f"[WARN] Invalid split value '{split}' at line {line_num}, skipped.")
                continue

    return train_queries, train_qrels, test_queries, test_qrels


def write_queries(queries: Dict[str, str], out_path: str) -> None:
    """
    Write query TSV file with format: qid \t query_text
    
    Args:
        queries: Dictionary of {session_id: query_text}
        out_path: Output file path
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, qtext in queries.items():
            # Clean special characters to maintain valid TSV format
            qtext_clean = str(qtext).replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{qtext_clean}\n")


def write_qrels(qrels: Dict[str, Dict[str, int]], out_path: str) -> None:
    """
    Write QRELs TSV file with standard TREC format: qid \t Q0 \t docid \t rel
    
    Compatible with loader logic: qid, _, docid, rel = parts[:4]
    
    Args:
        qrels: Dictionary of {session_id: {item_id: relevance_score}}
        out_path: Output file path
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for docid, rel in docs.items():
                f.write(f"{qid}\tQ0\t{docid}\t{rel}\n")


def main():
    """Main processing pipeline"""
    (
        train_queries,
        train_qrels,
        test_queries,
        test_qrels,
    ) = process_file(INPUT_FILE)

    # Print processing statistics
    print(f"Train queries: {len(train_queries)}")
    print(f"Train qrels sessions: {len(train_qrels)}")
    print(f"Test queries: {len(test_queries)}")
    print(f"Test qrels sessions: {len(test_qrels)}")

    # Write output files
    write_queries(train_queries, TRAIN_QUERIES_FILE)
    write_qrels(train_qrels, TRAIN_QRELS_FILE)

    write_queries(test_queries, TEST_QUERIES_FILE)
    write_qrels(test_qrels, TEST_QRELS_FILE)

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()