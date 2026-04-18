"""
convert_to_parquet.py
=====================
Run this ONCE on your local machine before deploying.

    python convert_to_parquet.py

It reads D:\\Insp_Ds\\Insp-ds-qc.csv  (your original 700 MB CSV),
cleans it, and writes  data/Insp-ds-qc.parquet  (~60-120 MB).

Then upload data/Insp-ds-qc.parquet to Google Drive and follow
the deployment instructions in README.md.
"""

import pandas as pd
from pathlib import Path

# ── INPUT / OUTPUT PATHS ──────────────────────────────────────────
CSV_PATH     = Path(r"D:\Insp_Ds\Insp-ds-qc.csv")
OUTPUT_DIR   = Path("data")
OUTPUT_PATH  = OUTPUT_DIR / "Insp-ds-qc.parquet"

MONTH_ORDER = ["Oct 2025", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026"]
MONTH_MAP = {
    "Oct": "Oct 2025", "Nov": "Nov 2025", "Dec": "Dec 2025",
    "Jan": "Jan 2026", "Feb": "Feb 2026", "Mar": "Mar 2026",
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(
        CSV_PATH,
        dtype={
            "APPOINTMENT_ID": "int32",
            "INSP_MONTH":     "category",
            "INSP_APP_FIELD": "category",
            "DS_OUTPUT":      "object",
        },
        engine="c",
    )
    df.columns = df.columns.str.strip()
    print(f"  Rows loaded : {len(df):,}")

    # Clean DS_OUTPUT
    df["DS_OUTPUT"] = (
        df["DS_OUTPUT"]
        .astype(str)
        .str.replace('"', "", regex=False)
        .str.strip()
    )
    df["DS_OUTPUT"] = pd.to_numeric(df["DS_OUTPUT"], errors="coerce")
    df.dropna(subset=["DS_OUTPUT"], inplace=True)
    df["DS_OUTPUT"] = df["DS_OUTPUT"].astype("int8")
    df = df[df["DS_OUTPUT"].isin([0, 1, 2, 3])].copy()

    # Map month labels
    df["MONTH_LABEL"] = (
        df["INSP_MONTH"]
        .map(MONTH_MAP)
        .astype(pd.CategoricalDtype(categories=MONTH_ORDER, ordered=True))
    )
    df.dropna(subset=["MONTH_LABEL"], inplace=True)
    df.drop(columns=["INSP_MONTH"], inplace=True)

    # Keep only the columns the dashboard needs
    df = df[["APPOINTMENT_ID", "INSP_APP_FIELD", "DS_OUTPUT", "MONTH_LABEL"]].copy()

    print(f"  Rows after cleaning : {len(df):,}")
    print(f"Writing {OUTPUT_PATH} ...")
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False, compression="snappy")

    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    print(f"  Done. File size: {size_mb:.1f} MB  (was ~700 MB CSV)")
    print()
    print("Next steps:")
    print("  1. Upload data/Insp-ds-qc.parquet to Google Drive")
    print("  2. Right-click -> Share -> 'Anyone with the link' -> Viewer")
    print("  3. Copy the file ID from the share URL")
    print("     URL looks like: https://drive.google.com/file/d/FILE_ID_HERE/view")
    print("  4. Follow README.md for the rest of the deployment steps")


if __name__ == "__main__":
    main()