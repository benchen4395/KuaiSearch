import json
import random
from pathlib import Path

# Set random seed for reproducible train/validation split
random.seed(42)

# Input/output configuration
INPUT_PATH = "data/relevance.jsonl"
SAVE_DIR = "relevance/data"
OUTPUT_PREFIX = "rank"      # Output prefix: rank.train.jsonl / rank.valid.jsonl / rank.test.jsonl
TEST_EVERY = 10             # Sample 1 out of every N records for test set (1/10)
VALIDATION_FRACTION = 0.01  # 1% of training data for validation set

# Define which fields compose each item's text (order can be adjusted)
FIELDS_IN_TEXT = [
    "item_title",
    "brand",
    "seller_name",
    "attr_value",
]

def build_text(obj: dict) -> str:
    """
    Construct clean text string from item fields with validation.
    
    Args:
        obj: Dictionary containing item metadata
        
    Returns:
        Combined text string of non-empty, valid fields (space-separated)
    """
    parts = []
    for k in FIELDS_IN_TEXT:
        v = obj.get(k, "")
        if v is None:
            continue
        v = str(v).strip()
        if not v:
            continue

        # Critical modification: Skip brand field if value is "无品牌" (no brand in Chinese)
        if k == "brand" and v == "无品牌":
            continue

        parts.append(v)
    return " ".join(parts)

def main():
    """Main processing function: Split relevance data into train/validation/test sets with text construction"""
    # Initialize file paths
    in_path = Path(INPUT_PATH)
    train_path = Path(f"{SAVE_DIR}/{OUTPUT_PREFIX}.train.jsonl")
    valid_path = Path(f"{SAVE_DIR}/{OUTPUT_PREFIX}.valid.jsonl")  # New validation set
    test_path = Path(f"{SAVE_DIR}/{OUTPUT_PREFIX}.test.jsonl")

    # Initialize counters
    cnt = 0
    train_raw_cnt = 0
    test_cnt = 0
    
    # Temporary storage for training data (to split validation later)
    train_records = []

    # First pass: Split raw data into train (temporary) and test sets
    with in_path.open("r", encoding="utf-8") as fin, \
         test_path.open("w", encoding="utf-8") as ftest:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Extract and validate query text
            query = str(obj.get("query", "")).strip()
            if not query:
                continue

            # Build cleaned item text from configured fields
            text = build_text(obj)
            if not text:
                continue

            # Extract relevance label (default to 0 if missing)
            rel = int(obj.get("label", 0))

            # Create final record
            record = {
                "query": query,
                "text": text,
                "label": rel
            }

            # Split into train (temp storage) and test sets (1 out of TEST_EVERY to test)
            if cnt % TEST_EVERY == 0:
                ftest.write(json.dumps(record, ensure_ascii=False) + "\n")
                test_cnt += 1
            else:
                train_records.append(record)
                train_raw_cnt += 1

            cnt += 1

    # Second pass: Split training data into train (99%) and validation (1%)
    # Shuffle first for random distribution
    random.shuffle(train_records)
    valid_size = int(len(train_records) * VALIDATION_FRACTION)
    
    # Split the shuffled training records
    valid_records = train_records[:valid_size]
    final_train_records = train_records[valid_size:]

    # Write train and validation sets to files
    with train_path.open("w", encoding="utf-8") as ftrain:
        for record in final_train_records:
            ftrain.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    with valid_path.open("w", encoding="utf-8") as fvalid:
        for record in valid_records:
            fvalid.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print final statistics and file paths
    print(
        f"Processing completed.\n"
        f"  Total samples processed   = {cnt}\n"
        f"  Raw training samples      = {train_raw_cnt}\n"
        f"  Final train samples saved = {len(final_train_records)}\n"
        f"  Validation samples saved  = {len(valid_records)}\n"
        f"  Test samples saved        = {test_cnt}\n"
        f"\nFile paths:\n"
        f"  Train file      = {train_path.resolve()}\n"
        f"  Validation file = {valid_path.resolve()}\n"
        f"  Test file       = {test_path.resolve()}"
    )

if __name__ == "__main__":
    main()