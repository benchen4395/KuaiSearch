__all__ = [
    "SparseRetriever",
]

import os
from pathlib import Path
from .sparse_retriever.sparse_retriever import SparseRetriever

# Set environment variables ----------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if "RETRIV_BASE_PATH" not in os.environ: # allow user to set a different path in .bash_profile
    os.environ["RETRIV_BASE_PATH"] = str(Path.home() / ".retriv")

def set_base_path(path: str):
    os.environ["RETRIV_BASE_PATH"] = path