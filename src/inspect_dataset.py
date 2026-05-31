import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/complaints.csv")

def main():
    if not DATA_PATH.exists():
        print(f"File not found: {DATA_PATH}")
        print("Make sure your dataset is named complaints.csv and placed inside data/raw/")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, nrows=5000, low_memory=False)

    print("\n=== Dataset Shape ===")
    print(df.shape)

    print("\n=== Columns ===")
    for col in df.columns:
        print("-", col)

    print("\n=== Missing Values: Top 20 ===")
    print(df.isnull().sum().sort_values(ascending=False).head(20))

    print("\n=== First 3 Rows ===")
    print(df.head(3))

    possible_text_cols = [
        "Consumer complaint narrative",
        "consumer_complaint_narrative",
        "narrative",
        "complaint_narrative"
    ]

    text_col = None
    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col:
        print(f"\n=== Narrative Column Found: {text_col} ===")
        print("Rows with complaint text:", df[text_col].notna().sum())

        sample_text = df[text_col].dropna().iloc[0]
        print("\n=== Sample Complaint Text ===")
        print(sample_text[:1000])
    else:
        print("\nNo obvious complaint narrative column found.")

    likely_label_cols = [
        "Product",
        "Issue",
        "Sub-product",
        "Sub-issue",
        "Company response to consumer",
        "Timely response?"
    ]

    print("\n=== Useful Label Columns Found ===")
    for col in likely_label_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).head(10))

if __name__ == "__main__":
    main()