import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration - Update these paths to your actual file locations
CORPUS_PATH = "data/corpus.jsonl"  # Path to your product corpus JSONL file
EMB_SAVE_DIR = "recall/data/embeddings"   # Directory to save generated embeddings


def load_item_texts(path: str, max_doc: int = None) -> list:
    """
    Load and preprocess product texts from corpus JSONL file.
    Extracts the item_title field and constructs clean text strings.
    
    Args:
        path: Path to the corpus JSONL file
        max_doc: Optional maximum number of documents to load (None = load all)
        
    Returns:
        List of processed product texts (item titles)
    """
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Stop loading if max document limit is reached
            if max_doc and len(texts) >= max_doc:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed JSON lines
                continue

            # Extract item title (primary text field)
            title = obj.get("item_title", "") or ""

            # Keep only non-empty text components
            parts = [p for p in [title] if p]
            if not parts:
                continue

            # Combine components into final text string
            text = " ".join(parts)
            texts.append(text)

    return texts


# Main execution flow
print("Preparing product texts...")
item_texts = load_item_texts(CORPUS_PATH)
print(f"Number of valid items loaded: {len(item_texts)}")
print("Example texts (first 5):", item_texts[:5])

# Initialize embedding model (BGE small Chinese v1.5 for Chinese text)
model_name = "BAAI/bge-small-zh-v1.5"
model = SentenceTransformer(
    model_name,
    model_kwargs={"device_map": "cuda:0", "torch_dtype": "auto"},
)

print("Ready for text encoding...")

# Generate embeddings for product texts
embeddings = model.encode(
    item_texts,
    batch_size=256,
    device="cuda:0",
    show_progress_bar=True,
)

# Verify embedding dimensions
print(f"Embedding generation complete. Shape: {embeddings.shape}")

# Save embeddings to disk
os.makedirs(EMB_SAVE_DIR, exist_ok=True)
embedding_filename = f"{EMB_SAVE_DIR}/item_emb_{embeddings.shape[1]}.npy"
np.save(embedding_filename, embeddings)

print(f"Embeddings saved to: {embedding_filename}")
print("All processing completed successfully.")