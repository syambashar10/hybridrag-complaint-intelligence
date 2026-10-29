from src.fusion import reciprocal_rank_fusion


def _item(row_index, rank, score):
    return {
        "row_index": row_index,
        "rank": rank,
        "score": score,
        "complaint_id": str(row_index),
        "product": "P",
        "issue": "I",
        "company": "C",
        "date_received": "2026-01-01",
        "text": "text",
    }


def test_rrf_rewards_documents_ranked_by_both_retrievers():
    keyword = [_item(1, 1, 0.9), _item(2, 2, 0.8)]
    semantic = [_item(2, 1, 0.95), _item(3, 2, 0.7)]

    fused = reciprocal_rank_fusion(
        {"keyword": keyword, "semantic": semantic}, rrf_k=60, top_k=3
    )

    assert fused[0]["row_index"] == 2
    assert fused[0]["keyword_rank"] == 2
    assert fused[0]["semantic_rank"] == 1
    assert 0 < fused[0]["hybrid_score"] <= 1
