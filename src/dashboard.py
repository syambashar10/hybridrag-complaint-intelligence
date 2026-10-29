from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from .analyzer import ComplaintAnalyzer
    from .config import FAISS_INDEX_PATH, PROCESSED_DATA_PATH
except ImportError:
    from analyzer import ComplaintAnalyzer
    from config import FAISS_INDEX_PATH, PROCESSED_DATA_PATH

st.set_page_config(page_title="HybridRAG Complaint Intelligence", page_icon="🔎", layout="wide")
st.title("HybridRAG Complaint Intelligence")
st.caption("CFPB complaint retrieval with TF-IDF, semantic embeddings, FAISS, and Reciprocal Rank Fusion.")

if not PROCESSED_DATA_PATH.exists() or not FAISS_INDEX_PATH.exists():
    st.error("Processed data or FAISS index is missing.")
    st.code(
        "python -m src.prepare_dataset\n"
        "python -m src.build_semantic_index\n"
        "streamlit run src/dashboard.py"
    )
    st.stop()


@st.cache_resource(show_spinner="Loading search indexes...")
def load_analyzer() -> ComplaintAnalyzer:
    return ComplaintAnalyzer().load()


analyzer = load_analyzer()

search_tab, data_tab = st.tabs(["Complaint analysis", "Dataset overview"])

with search_tab:
    example = "Someone opened accounts using my identity and they are damaging my credit report."
    complaint = st.text_area("Complaint", value=example, height=150)
    evidence_k = st.slider("Evidence complaints", min_value=3, max_value=15, value=8)

    if st.button("Analyze complaint", type="primary", use_container_width=True):
        try:
            result = analyzer.analyze(complaint, evidence_k=evidence_k)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            left, right = st.columns(2)
            left.metric("Predicted product", result["predicted_product"])
            left.caption(f"Retrieval agreement: {result['product_confidence']:.1%}")
            right.metric("Predicted issue", result["predicted_issue"])
            right.caption(f"Retrieval agreement: {result['issue_confidence']:.1%}")

            st.subheader("Evidence")
            for item in result["evidence"]:
                title = (
                    f"#{item['rank']} · {item['product']} · "
                    f"hybrid {item['hybrid_score']:.3f}"
                )
                with st.expander(title, expanded=item["rank"] <= 2):
                    st.write(f"**Issue:** {item['issue']}")
                    st.write(f"**Company:** {item['company']}")
                    st.write(
                        f"**Keyword rank:** {item.get('keyword_rank', '—')} · "
                        f"**Semantic rank:** {item.get('semantic_rank', '—')}"
                    )
                    st.write(item["text"])

with data_tab:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    a, b, c = st.columns(3)
    a.metric("Complaints", f"{len(df):,}")
    b.metric("Products", f"{df['product'].nunique(dropna=True):,}")
    c.metric("Issues", f"{df['issue'].nunique(dropna=True):,}")

    st.subheader("Top products")
    st.bar_chart(df["product"].value_counts().head(12))

    st.subheader("Top issues")
    st.bar_chart(df["issue"].value_counts().head(12))
