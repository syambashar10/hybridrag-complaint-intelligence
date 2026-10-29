from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "complaints.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "complaints_clean.csv"
INDEX_DIR = DATA_DIR / "indexes"
FAISS_INDEX_PATH = INDEX_DIR / "complaints_faiss.index"
EMBEDDINGS_PATH = INDEX_DIR / "complaints_embeddings.npy"
INDEX_METADATA_PATH = INDEX_DIR / "complaints_metadata.json"

EMBEDDING_MODEL = os.getenv("HYBRIDRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("HYBRIDRAG_TOP_K", "5"))
DEFAULT_CANDIDATE_K = int(os.getenv("HYBRIDRAG_CANDIDATE_K", "30"))
RRF_K = int(os.getenv("HYBRIDRAG_RRF_K", "60"))
