from __future__ import annotations

import pandas as pd

try:
    from .config import RAW_DATA_PATH
except ImportError:
    from config import RAW_DATA_PATH

SAMPLE_ROWS = 5_000
TEXT_COL_CANDIDATES = [
    "Consumer complaint narrative",
    "consumer_complaint_narrative",
    "narrative",
    "complaint_narrative",
]


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise SystemExit(
            f"File not found: {RAW_DATA_PATH}\n"
            "Download the CFPB complaints CSV and place it at data/raw/complaints.csv."
        )

    print(f"Loading first {SAMPLE_ROWS:,} rows from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH, nrows=SAMPLE_ROWS, low_memory=False)

    print("\n=== Sample Shape ===")
    print(df.shape)
    print("\n=== Columns ===")
    for column in df.columns:
        print("-", column)

    print("\n=== Missing Values: Top 20 ===")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
    if text_col is None:
        print("\nNo recognized complaint narrative column was found.")
    else:
        narratives = df[text_col].dropna().astype(str)
        print(f"\n=== Narrative Column: {text_col} ===")
        print(f"Rows with complaint text: {len(narratives):,}")
        if not narratives.empty:
            print("\n=== Sample Complaint ===")
            print(narratives.iloc[0][:1_000])

    useful = [
        "Product",
        "Issue",
        "Sub-product",
        "Sub-issue",
        "Company response to consumer",
        "Timely response?",
    ]
    print("\n=== Useful Label Columns ===")
    for column in useful:
        if column in df.columns:
            print(f"\n{column}:")
            print(df[column].value_counts(dropna=False).head(10))


if __name__ == "__main__":
    main()
