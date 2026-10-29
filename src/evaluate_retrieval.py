from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from .hybrid_search import HybridSearchEngine
except ImportError:
    from hybrid_search import HybridSearchEngine


def _without_self(results: list[dict], complaint_id: str, top_k: int) -> list[dict]:
    return [r for r in results if str(r.get("complaint_id")) != complaint_id][:top_k]


def _metrics(results: list[dict], target_product: str, target_issue: str) -> dict:
    product_ranks = [
        i for i, item in enumerate(results, start=1) if item.get("product") == target_product
    ]
    issue_ranks = [
        i for i, item in enumerate(results, start=1) if item.get("issue") == target_issue
    ]
    return {
        "product_hit": float(bool(product_ranks)),
        "product_mrr": 1.0 / product_ranks[0] if product_ranks else 0.0,
        "issue_hit": float(bool(issue_ranks)),
        "issue_mrr": 1.0 / issue_ranks[0] if issue_ranks else 0.0,
    }


def evaluate(sample_size: int = 100, top_k: int = 5, seed: int = 42) -> pd.DataFrame:
    engine = HybridSearchEngine().load()
    assert engine.keyword.df is not None
    df = engine.keyword.df.dropna(subset=["clean_text", "product", "issue"]).copy()
    if df.empty:
        raise ValueError("No labeled complaints are available for evaluation.")

    sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    aggregate: dict[str, list[float]] = {}

    for row in sample.itertuples(index=False):
        query = str(row.clean_text)
        complaint_id = str(row.complaint_id)
        candidate_k = top_k + 8

        result_sets = {
            "keyword": _without_self(
                engine.keyword.search(query, candidate_k), complaint_id, top_k
            ),
            "semantic": _without_self(
                engine.semantic.search(query, candidate_k), complaint_id, top_k
            ),
            "hybrid": _without_self(
                engine.search(query, top_k=candidate_k, candidate_k=max(30, candidate_k * 3)),
                complaint_id,
                top_k,
            ),
        }

        for method, results in result_sets.items():
            metrics = _metrics(results, str(row.product), str(row.issue))
            for metric, value in metrics.items():
                aggregate.setdefault(f"{method}_{metric}", []).append(value)

    rows = []
    for method in ("keyword", "semantic", "hybrid"):
        rows.append(
            {
                "method": method,
                f"product_hit@{top_k}": np.mean(aggregate[f"{method}_product_hit"]),
                "product_mrr": np.mean(aggregate[f"{method}_product_mrr"]),
                f"issue_hit@{top_k}": np.mean(aggregate[f"{method}_issue_hit"]),
                "issue_mrr": np.mean(aggregate[f"{method}_issue_mrr"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval using CFPB labels as proxy relevance.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    table = evaluate(args.sample_size, args.top_k, args.seed)
    print("\nProxy retrieval evaluation")
    print("A result is considered relevant when product/issue labels match the query record.\n")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
