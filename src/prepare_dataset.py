import re
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/complaints.csv")
OUTPUT_PATH = Path("data/processed/complaints_clean.csv")

CHUNKSIZE = 50_000
MAX_ROWS = 20_000

TEXT_COL = "Consumer complaint narrative"

KEEP_COLUMNS = [
    "Complaint ID",
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Submitted via",
    "Company response to consumer",
    "Timely response?",
]


def clean_text(text: str) -> str:
    text = str(text)

    # CFPB redacts private info like XXXX, XX/XX/XXXX, etc.
    text = re.sub(r"X{2,}", " ", text)

    # Remove extra spaces and newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    if not RAW_PATH.exists():
        print(f"File not found: {RAW_PATH}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cleaned_chunks = []
    total_kept = 0

    print("Starting dataset preparation...")

    for chunk_id, chunk in enumerate(pd.read_csv(RAW_PATH, chunksize=CHUNKSIZE, low_memory=False), start=1):
        print(f"Processing chunk {chunk_id}...")

        available_cols = [col for col in KEEP_COLUMNS if col in chunk.columns]
        chunk = chunk[available_cols]

        # Keep only rows that have real complaint narrative text
        chunk = chunk.dropna(subset=[TEXT_COL])

        # Clean text
        chunk["clean_text"] = chunk[TEXT_COL].apply(clean_text)

        # Remove very short complaints
        chunk = chunk[chunk["clean_text"].str.len() >= 40]

        # Rename columns to easier names
        chunk = chunk.rename(columns={
            "Complaint ID": "complaint_id",
            "Date received": "date_received",
            "Product": "product",
            "Sub-product": "sub_product",
            "Issue": "issue",
            "Sub-issue": "sub_issue",
            "Consumer complaint narrative": "original_text",
            "Company": "company",
            "State": "state",
            "Submitted via": "submitted_via",
            "Company response to consumer": "company_response",
            "Timely response?": "timely_response",
        })

        cleaned_chunks.append(chunk)
        total_kept += len(chunk)

        print(f"Rows kept so far: {total_kept}")

        if total_kept >= MAX_ROWS:
            break

    if not cleaned_chunks:
        print("No valid complaint narratives found.")
        return

    df_clean = pd.concat(cleaned_chunks, ignore_index=True)

    # Limit final size
    df_clean = df_clean.head(MAX_ROWS)

    df_clean.to_csv(OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Saved clean dataset to: {OUTPUT_PATH}")
    print(f"Final shape: {df_clean.shape}")

    print("\nProduct distribution:")
    print(df_clean["product"].value_counts().head(10))

    print("\nExample cleaned complaint:")
    print(df_clean["clean_text"].iloc[0][:1000])


if __name__ == "__main__":
    main()