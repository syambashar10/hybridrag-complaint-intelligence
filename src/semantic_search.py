from __future__ import annotations

import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from .config import FAISS_INDEX_PATH, INDEX_METADATA_PATH, PROCESSED_DATA_PATH
except ImportError:
    from config import FAISS_INDEX_PATH, INDEX_METADATA_PATH, PROCESSED_DATA_PATH


class SemanticSearchEngine:
    """Loads a prebuilt FAISS index; it never re-embeds the full corpus at query time."""

    def __init__(
        self,
        data_path: Path = PROCESSED_DATA_PATH,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = INDEX_METADATA_PATH,
    ):
        self.data_path = data_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.df: pd.DataFrame | None = None
        self.model: SentenceTransformer | None = None
        self.index = None
        self.metadata: dict = {}

    def load(self) -> "SemanticSearchEngine":
        missing = [
            str(path)
            for path in (self.data_path, self.index_path, self.metadata_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing semantic-search artifacts: "
                + ", ".join(missing)
                + ". Run prepare_dataset.py and build_semantic_index.py first."
            )

        self.df = pd.read_csv(self.data_path)
        self.df["clean_text"] = self.df["clean_text"].fillna("").astype(str)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(self.index_path))

        if int(self.index.ntotal) != len(self.df):
            raise ValueError(
                "FAISS index and processed dataset have different row counts. "
                "Rebuild the semantic index."
            )

        model_name = self.metadata.get("model_name")
        if not model_name:
            raise ValueError("Semantic index metadata is missing model_name.")
        self.model = SentenceTransformer(model_name)
        return self

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        if self.df is None or self.model is None or self.index is None:
            raise RuntimeError("Semantic index is not ready. Call load() first.")

        top_k = max(1, min(int(top_k), len(self.df)))
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results: list[dict] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            row = self.df.iloc[int(idx)]
            results.append(
                {
                    "row_index": int(idx),
                    "rank": rank,
                    "score": float(score),
                    "complaint_id": str(row.get("complaint_id", "")),
                    "product": row.get("product", ""),
                    "issue": row.get("issue", ""),
                    "company": row.get("company", ""),
                    "date_received": row.get("date_received", ""),
                    "text": row.get("clean_text", ""),
                }
            )
        return results


def main() -> None:
    engine = SemanticSearchEngine().load()
    print(f"Saved semantic index ready with {len(engine.df):,} complaints.")
    print("Type a complaint. Type 'exit' to stop.\n")
    while True:
        query = input("Complaint query: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            print("Please enter a complaint.\n")
            continue
        for item in engine.search(query, top_k=5):
            print("=" * 80)
            print(f"#{item['rank']} score={item['score']:.4f}")
            print(f"Product: {item['product']}")
            print(f"Issue: {item['issue']}")
            print(f"Company: {item['company']}")
            print(item["text"][:700])


if __name__ == "__main__":
    main()
