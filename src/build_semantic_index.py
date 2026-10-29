from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from .config import (
        EMBEDDING_MODEL,
        EMBEDDINGS_PATH,
        FAISS_INDEX_PATH,
        INDEX_DIR,
        INDEX_METADATA_PATH,
        PROCESSED_DATA_PATH,
    )
except ImportError:
    from config import (
        EMBEDDING_MODEL,
        EMBEDDINGS_PATH,
        FAISS_INDEX_PATH,
        INDEX_DIR,
        INDEX_METADATA_PATH,
        PROCESSED_DATA_PATH,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index(data_path: Path = PROCESSED_DATA_PATH) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}. Run prepare_dataset.py first."
        )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    if "clean_text" not in df.columns:
        raise ValueError("clean_text column not found in processed dataset.")
    texts = df["clean_text"].fillna("").astype(str).tolist()
    if not texts:
        raise ValueError("Processed dataset is empty.")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Embedding {len(texts):,} complaints...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(EMBEDDINGS_PATH, embeddings)

    metadata = {
        "model_name": EMBEDDING_MODEL,
        "num_vectors": int(index.ntotal),
        "embedding_dimension": dimension,
        "index_type": "IndexFlatIP",
        "normalized_embeddings": True,
        "data_path": str(data_path),
        "dataset_sha256": sha256_file(data_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    INDEX_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved FAISS index: {FAISS_INDEX_PATH}")
    print(f"Saved embeddings: {EMBEDDINGS_PATH}")
    print(f"Saved metadata: {INDEX_METADATA_PATH}")
    return metadata


def main() -> None:
    try:
        build_index()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
