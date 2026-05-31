import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/processed/complaints_clean.csv")
INDEX_DIR = Path("data/indexes")

FAISS_INDEX_PATH = INDEX_DIR / "complaints_faiss.index"
EMBEDDINGS_PATH = INDEX_DIR / "complaints_embeddings.npy"
METADATA_PATH = INDEX_DIR / "complaints_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading clean dataset...")
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("")

    print(f"Loaded {len(df)} complaints.")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    embeddings = model.encode(
        df["clean_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("Saving FAISS index...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print("Saving embeddings...")
    np.save(EMBEDDINGS_PATH, embeddings)

    metadata = {
        "model_name": MODEL_NAME,
        "num_vectors": int(index.ntotal),
        "embedding_dimension": int(dimension),
        "data_path": str(DATA_PATH),
        "faiss_index_path": str(FAISS_INDEX_PATH),
        "embeddings_path": str(EMBEDDINGS_PATH),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"Embeddings saved to: {EMBEDDINGS_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Vectors: {index.ntotal}")
    print(f"Dimension: {dimension}")


if __name__ == "__main__":
    main()