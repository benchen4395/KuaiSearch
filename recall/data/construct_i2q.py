import json

INPUT_PATH = "recall/data/train_dpr.json"           # Input: Your DPR training data (JSON list)
OUTPUT_PATH = "recall/data/docTquery_train_data.jsonl"  # Output: Converted to JSONL (one sample per line)

PROMPT_PREFIX = "为下面的商品标题生成一个用户搜索query："

def main():
    # 1. Read train_dpr.json (it's a JSON list)
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    count_input = 0
    count_output = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for example in data:
            count_input += 1
            query = example.get("query", "").strip()
            positive_items = example.get("positive_item", []) or []

            for pos in positive_items:
                # Field compatibility: prioritize "text", fallback to "title"
                text = (
                    (pos.get("text") if isinstance(pos, dict) else None)
                    or (pos.get("title") if isinstance(pos, dict) else None)
                    or ""
                )
                text = text.strip()
                if not text:
                    continue

                output_obj = {
                    "query": query,
                    "item": f"{PROMPT_PREFIX}{text}",
                }
                fout.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
                count_output += 1

    print(f"Number of input samples read: {count_input}")
    print(f"Number of output lines written (positive samples): {count_output}")
    print(f"Output file saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()