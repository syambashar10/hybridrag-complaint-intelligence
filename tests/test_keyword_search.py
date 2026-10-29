import pandas as pd

from src.keyword_search import KeywordSearchEngine


def test_keyword_search_returns_relevant_document(tmp_path):
    path = tmp_path / "complaints.csv"
    pd.DataFrame(
        [
            {
                "complaint_id": "1",
                "product": "Credit reporting",
                "issue": "Incorrect information",
                "company": "A",
                "date_received": "2026-01-01",
                "clean_text": "An account on my credit report does not belong to me.",
            },
            {
                "complaint_id": "2",
                "product": "Mortgage",
                "issue": "Payment issue",
                "company": "B",
                "date_received": "2026-01-02",
                "clean_text": "My mortgage payment was processed late.",
            },
        ]
    ).to_csv(path, index=False)

    engine = KeywordSearchEngine(path).load()
    results = engine.search("unknown account on credit report", top_k=1)

    assert results[0]["complaint_id"] == "1"
    assert results[0]["score"] > 0
