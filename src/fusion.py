from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    result_sets: Mapping[str, Sequence[dict]],
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """Fuse ranked lists without assuming their raw scores share the same scale."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    weights = dict(weights or {})
    merged: dict[int, dict] = {}
    active_weight_sum = 0.0

    for source, results in result_sets.items():
        weight = float(weights.get(source, 1.0))
        if weight <= 0:
            continue
        active_weight_sum += weight

        for position, item in enumerate(results, start=1):
            row_index = int(item["row_index"])
            if row_index not in merged:
                merged[row_index] = {
                    "row_index": row_index,
                    "complaint_id": item.get("complaint_id", ""),
                    "product": item.get("product", ""),
                    "issue": item.get("issue", ""),
                    "company": item.get("company", ""),
                    "date_received": item.get("date_received", ""),
                    "text": item.get("text", ""),
                    "rrf_score_raw": 0.0,
                }

            fused = merged[row_index]
            fused["rrf_score_raw"] += weight / (rrf_k + position)
            fused[f"{source}_rank"] = position
            fused[f"{source}_score"] = float(item.get("score", 0.0))

    if not merged:
        return []

    theoretical_best = active_weight_sum / (rrf_k + 1) if active_weight_sum else 1.0
    ranked = sorted(
        merged.values(),
        key=lambda item: (item["rrf_score_raw"], -item["row_index"]),
        reverse=True,
    )

    output: list[dict] = []
    for rank, item in enumerate(ranked[:top_k], start=1):
        item = dict(item)
        item["rank"] = rank
        item["hybrid_score"] = (
            item.pop("rrf_score_raw") / theoretical_best if theoretical_best else 0.0
        )
        output.append(item)
    return output
