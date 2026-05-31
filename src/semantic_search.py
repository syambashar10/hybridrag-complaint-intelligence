import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

DATA_PATH = Path("data/processed/complaints_clean.csv")


class SemanticSearchEngine:
    def __init__(self, data_path: Path, model_name: str = "all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.model_name = model_name
        self.df = None
        self.model = None
        self.index = None
        self.embeddings = None

    def load_data(self):
        print("Loading clean dataset...")
        self.df = pd.read_csv(self.data_path)

        if "clean_text" not in self.df.columns:
            raise ValueError("clean_text column not found in dataset.")

        self.df["clean_text"] = self.df["clean_text"].fillna("")
        print(f"Loaded {len(self.df)} complaints.")

    def load_model(self):
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded.")

    def build_index(self):
        print("Creating embeddings. This may take a few minutes...")

        texts = self.df["clean_text"].tolist()

        self.embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.embeddings = self.embeddings.astype("float32")

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(self.embeddings)

        print(f"Semantic FAISS index ready with {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 5):
        if self.index is None:
            raise ValueError("Index not built. Run build_index() first.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            row = self.df.iloc[idx]

            results.append({
                "rank": rank,
                "score": float(score),
                "complaint_id": row.get("complaint_id", ""),
                "product": row.get("product", ""),
                "issue": row.get("issue", ""),
                "company": row.get("company", ""),
                "text": row.get("clean_text", "")[:700]
            })

        return results


def main():
    engine = SemanticSearchEngine(DATA_PATH)
    engine.load_data()
    engine.load_model()
    engine.build_index()

    print("\nSemantic search engine is ready.")
    print("Type a customer complaint. Type 'exit' to stop.\n")

    while True:
        query = input("Complaint query: ")

        if query.lower().strip() == "exit":
            break

        results = engine.search(query, top_k=5)

        print("\nTop semantic results:")
        for item in results:
            print("=" * 80)
            print(f"Rank: {item['rank']}")
            print(f"Score: {item['score']:.4f}")
            print(f"Product: {item['product']}")
            print(f"Issue: {item['issue']}")
            print(f"Company: {item['company']}")
            print(f"Text: {item['text']}")
        print("=" * 80)


if __name__ == "__main__":
    main()