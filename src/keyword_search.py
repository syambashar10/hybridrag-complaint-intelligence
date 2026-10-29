from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from .config import PROCESSED_DATA_PATH
except ImportError:
    from config import PROCESSED_DATA_PATH


class KeywordSearchEngine:
    def __init__(self, data_path: Path = PROCESSED_DATA_PATH):
        self.data_path = data_path
        self.df: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None

    def load_data(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {self.data_path}. Run prepare_dataset.py first."
            )
        self.df = pd.read_csv(self.data_path)
        if "clean_text" not in self.df.columns:
            raise ValueError("clean_text column not found in processed dataset.")
        self.df["clean_text"] = self.df["clean_text"].fillna("").astype(str)

    def build_index(self) -> None:
        if self.df is None:
            self.load_data()
        assert self.df is not None
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["clean_text"])

    def load(self) -> "KeywordSearchEngine":
        self.load_data()
        self.build_index()
        return self

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        if self.df is None or self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("Keyword index is not ready. Call load() first.")

        top_k = max(1, min(int(top_k), len(self.df)))
        query_vector = self.vectorizer.transform([query])
        scores = linear_kernel(query_vector, self.tfidf_matrix).ravel()
        top_indices = scores.argsort()[::-1][:top_k]

        results: list[dict] = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.df.iloc[int(idx)]
            results.append(
                {
                    "row_index": int(idx),
                    "rank": rank,
                    "score": float(scores[idx]),
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
    engine = KeywordSearchEngine().load()
    print(f"Keyword index ready with {len(engine.df):,} complaints.")
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
