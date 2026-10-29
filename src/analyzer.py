from __future__ import annotations

from collections import defaultdict

try:
    from .hybrid_search import HybridSearchEngine
except ImportError:
    from hybrid_search import HybridSearchEngine


def _weighted_vote(results: list[dict], field: str) -> tuple[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for item in results:
        value = item.get(field)
        if value is None:
            continue
        label = str(value).strip()
        if not label or label.lower() == "nan":
            continue
        totals[label] += max(float(item.get("hybrid_score", 0.0)), 1e-9)

    if not totals:
        return "Unknown", 0.0
    winner, winner_score = max(totals.items(), key=lambda pair: pair[1])
    total_score = sum(totals.values())
    return winner, (winner_score / total_score if total_score else 0.0)


class ComplaintAnalyzer:
    """Retrieval-grounded classifier; no generative model or invented confidence."""

    def __init__(self, engine: HybridSearchEngine | None = None):
        self.engine = engine or HybridSearchEngine()

    def load(self) -> "ComplaintAnalyzer":
        self.engine.load()
        return self

    def analyze(self, complaint: str, evidence_k: int = 8) -> dict:
        results = self.engine.search(complaint, top_k=evidence_k)
        product, product_confidence = _weighted_vote(results, "product")
        issue, issue_confidence = _weighted_vote(results, "issue")
        return {
            "predicted_product": product,
            "product_confidence": product_confidence,
            "predicted_issue": issue,
            "issue_confidence": issue_confidence,
            "evidence": results,
            "method": "TF-IDF + dense semantic retrieval + RRF weighted vote",
        }
