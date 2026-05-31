import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("data/processed/complaints_clean.csv")


class KeywordSearchEngine:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None

    def load_data(self):
        print("Loading clean dataset...")
        self.df = pd.read_csv(self.data_path)

        if "clean_text" not in self.df.columns:
            raise ValueError("clean_text column not found in dataset.")

        self.df["clean_text"] = self.df["clean_text"].fillna("")
        print(f"Loaded {len(self.df)} complaints.")

    def build_index(self):
        print("Building TF-IDF keyword index...")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50_000,
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["clean_text"])
        print("Keyword index ready.")

    def search(self, query: str, top_k: int = 5):
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Index not built. Run build_index() first.")

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        top_indices = scores.argsort()[::-1][:top_k]

        results = []

        for rank, idx in enumerate(top_indices, start=1):
            row = self.df.iloc[idx]

            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "complaint_id": row.get("complaint_id", ""),
                "product": row.get("product", ""),
                "issue": row.get("issue", ""),
                "company": row.get("company", ""),
                "text": row.get("clean_text", "")[:700]
            })

        return results


def main():
    engine = KeywordSearchEngine(DATA_PATH)
    engine.load_data()
    engine.build_index()

    print("\nKeyword search engine is ready.")
    print("Type a customer complaint. Type 'exit' to stop.\n")

    while True:
        query = input("Complaint query: ")

        if query.lower().strip() == "exit":
            break

        results = engine.search(query, top_k=5)

        print("\nTop results:")
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