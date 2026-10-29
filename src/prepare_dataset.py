from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

try:
    from .config import PROCESSED_DATA_PATH, RAW_DATA_PATH
except ImportError:
    from config import PROCESSED_DATA_PATH, RAW_DATA_PATH

CHUNKSIZE = int(os.getenv("HYBRIDRAG_CHUNKSIZE", "50000"))
MAX_ROWS = int(os.getenv("HYBRIDRAG_MAX_ROWS", "20000"))
MIN_TEXT_LENGTH = int(os.getenv("HYBRIDRAG_MIN_TEXT_LENGTH", "40"))
TEXT_COL = "Consumer complaint narrative"
ID_COL = "Complaint ID"

KEEP_COLUMNS = [
    ID_COL,
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    TEXT_COL,
    "Company",
    "State",
    "Submitted via",
    "Company response to consumer",
    "Timely response?",
]

RENAME_COLUMNS = {
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
}


def clean_text(text: object) -> str:
    """Apply conservative cleaning while preserving complaint meaning."""
    if pd.isna(text):
        return ""
    value = str(text)
    value = re.sub(r"X{2,}", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _validate_schema(columns: list[str]) -> None:
    missing = [column for column in (TEXT_COL, ID_COL) if column not in columns]
    if missing:
        raise ValueError(
            "Dataset is missing required CFPB columns: " + ", ".join(missing)
        )


def prepare_dataset(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
    max_rows: int = MAX_ROWS,
) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_path}. Place the CFPB CSV there first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_chunks: list[pd.DataFrame] = []
    total_kept = 0
    seen_ids: set[str] = set()
    schema_checked = False

    print(f"Reading {raw_path}")
    print(f"Target: up to {max_rows:,} usable complaint narratives")

    reader = pd.read_csv(raw_path, chunksize=CHUNKSIZE, low_memory=False)
    for chunk_id, chunk in enumerate(reader, start=1):
        if not schema_checked:
            _validate_schema(list(chunk.columns))
            schema_checked = True

        available_cols = [column for column in KEEP_COLUMNS if column in chunk.columns]
        chunk = chunk.loc[:, available_cols].copy()

        chunk["clean_text"] = (
            chunk[TEXT_COL]
            .astype("string")
            .fillna("")
            .map(clean_text)
            .astype("string")
        )
        chunk = chunk.loc[chunk["clean_text"].str.len().ge(MIN_TEXT_LENGTH)].copy()

        if chunk.empty:
            print(f"Chunk {chunk_id}: no usable narratives; skipped")
            continue

        chunk[ID_COL] = (
            chunk[ID_COL]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
        chunk = chunk.loc[chunk[ID_COL].notna() & chunk[ID_COL].ne("")].copy()
        chunk = chunk.drop_duplicates(subset=[ID_COL])
        chunk = chunk.loc[~chunk[ID_COL].isin(seen_ids)].copy()
        if chunk.empty:
            print(f"Chunk {chunk_id}: only duplicate/missing complaint IDs; skipped")
            continue

        seen_ids.update(chunk[ID_COL].dropna().astype(str).tolist())
        chunk = chunk.drop_duplicates(subset=["clean_text"])
        chunk = chunk.rename(columns=RENAME_COLUMNS)

        remaining = max_rows - total_kept
        if remaining <= 0:
            break
        chunk = chunk.head(remaining)

        cleaned_chunks.append(chunk)
        total_kept += len(chunk)
        print(f"Chunk {chunk_id}: kept {len(chunk):,}; total {total_kept:,}")

        if total_kept >= max_rows:
            break

    if not cleaned_chunks:
        raise ValueError("No usable complaint narratives were found in the dataset.")

    df_clean = pd.concat(cleaned_chunks, ignore_index=True)

    if "date_received" in df_clean.columns:
        parsed = pd.to_datetime(df_clean["date_received"], errors="coerce", utc=True)
        df_clean["date_received"] = parsed.dt.strftime("%Y-%m-%d")

    df_clean.to_csv(output_path, index=False)

    print("\nDataset preparation complete")
    print(f"Saved: {output_path}")
    print(f"Shape: {df_clean.shape}")
    if "product" in df_clean.columns:
        print("\nTop products:")
        print(df_clean["product"].value_counts().head(10))

    return df_clean


def main() -> None:
    try:
        prepare_dataset()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
