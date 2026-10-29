from __future__ import annotations

try:
    from .config import DEFAULT_CANDIDATE_K, DEFAULT_TOP_K, RRF_K
    from .fusion import reciprocal_rank_fusion
    from .keyword_search import KeywordSearchEngine
    from .semantic_search import SemanticSearchEngine
except ImportError:
    from config import DEFAULT_CANDIDATE_K, DEFAULT_TOP_K, RRF_K
    from fusion import reciprocal_rank_fusion
    from keyword_search import KeywordSearchEngine
    from semantic_search import SemanticSearchEngine


class HybridSearchEngine:
    def __init__(self, keyword_weight: float = 1.0, semantic_weight: float = 1.0):
        self.keyword = KeywordSearchEngine()
        self.semantic = SemanticSearchEngine()
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.loaded = False

    def load(self) -> "HybridSearchEngine":
        self.keyword.load()
        self.semantic.load()
        self.loaded = True
        return self

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> list[dict]:
        if not self.loaded:
            raise RuntimeError("Hybrid search engine is not ready. Call load() first.")
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        top_k = max(1, int(top_k))
        candidate_k = max(top_k, int(candidate_k))
        keyword_results = self.keyword.search(query, top_k=candidate_k)
        semantic_results = self.semantic.search(query, top_k=candidate_k)

        return reciprocal_rank_fusion(
            {"keyword": keyword_results, "semantic": semantic_results},
            weights={
                "keyword": self.keyword_weight,
                "semantic": self.semantic_weight,
            },
            rrf_k=RRF_K,
            top_k=top_k,
        )


def main() -> None:
    engine = HybridSearchEngine().load()
    print("Hybrid search ready (TF-IDF + semantic FAISS + Reciprocal Rank Fusion).")
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
            print(f"#{item['rank']} hybrid={item['hybrid_score']:.4f}")
            print(
                f"Keyword rank: {item.get('keyword_rank', '-')} | "
                f"Semantic rank: {item.get('semantic_rank', '-')}"
            )
            print(f"Product: {item['product']}")
            print(f"Issue: {item['issue']}")
            print(f"Company: {item['company']}")
            print(item["text"][:700])


if __name__ == "__main__":
    main()
