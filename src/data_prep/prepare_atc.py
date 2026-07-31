"""
Data preparation for ATC embeddings.

Reads data/raw/ATC.csv and produces two files:

  data/prepared_data/data_atc.csv
    One row per ATC drug entry. STY semantic-type nodes are excluded.
    Class ID shortened to the bare code (e.g. D11AX06).
    Added columns:
      ontology_type    : always "ATC"
      cui_is_duplicate : True when the same CUI appears on multiple ATC rows

  data/prepared_data/data_atc_links.csv
    Semantic Types column exploded so each STY link gets its own row.
    Only rows that actually have a Semantic Type are included.

Duplicate CUI semantics
-----------------------
ATC classifies by therapeutic indication, so the same molecule can appear
under multiple ATC codes (e.g. mesna as uroprotective AND as mucolytic).
This is by design — not dirty data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_ATC, DATA_ATC_PREPARED, DATA_ATC_LINKS

import pandas as pd

RAW_PATH = RAW_ATC
OUT_ATC = DATA_ATC_PREPARED
OUT_LINKS = DATA_ATC_LINKS


def shorten_class_id(series: pd.Series) -> pd.Series:
    """Extract the bare code from a BioPortal URI, e.g. D11AX06 from .../ATC/D11AX06."""
    return series.str.rsplit("/", n=1).str[-1]


def add_ontology_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Extract namespace from the *original* full URI before shortening
    df["ontology_type"] = df["Class ID"].str.extract(r"ontology/([^/]+)/")
    return df


def add_cui_duplicate_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cui_counts = df["CUI"].map(df["CUI"].value_counts())
    df["cui_is_duplicate"] = (cui_counts > 1) & df["CUI"].notna()
    return df


def build_atc(df: pd.DataFrame) -> pd.DataFrame:
    df = add_ontology_type(df)
    df = add_cui_duplicate_flag(df)
    df["Class ID"] = shorten_class_id(df["Class ID"])
    return df


def build_links(df: pd.DataFrame) -> pd.DataFrame:
    """Explode pipe-separated Semantic Types into one row per link."""
    df = add_ontology_type(df)
    df = add_cui_duplicate_flag(df)
    df["Class ID"] = shorten_class_id(df["Class ID"])

    df = df.dropna(subset=["Semantic Types"])
    df = df.assign(**{"Semantic Types": df["Semantic Types"].str.split("|")})
    df = df.explode("Semantic Types").reset_index(drop=True)
    return df


def report(df_atc: pd.DataFrame, df_links: pd.DataFrame) -> None:
    n_dupes = df_atc["cui_is_duplicate"].sum()
    unique_cui_dupes = df_atc.loc[df_atc["cui_is_duplicate"], "CUI"].nunique()
    print(
        f"CUI duplicates: {n_dupes} rows, {unique_cui_dupes} distinct CUIs"
        "\n  => same drug in multiple ATC categories (expected)"
    )
    print(f"\ndata_atc.csv   : {len(df_atc)} rows, {df_atc.shape[1]} columns")
    print(f"data_atc_links : {len(df_links)} rows, {df_links.shape[1]} columns")


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows from {RAW_PATH}")

    df = df[~df["Class ID"].str.contains("/STY/", regex=False)].reset_index(drop=True)
    print(f"After removing STY rows: {len(df)} rows\n")

    df_atc = build_atc(df)
    df_links = build_links(df)

    report(df_atc, df_links)

    df_atc.to_csv(OUT_ATC, index=False)
    df_links.to_csv(OUT_LINKS, index=False)
    print(f"\nWrote {OUT_ATC}")
    print(f"Wrote {OUT_LINKS}")


if __name__ == "__main__":
    main()
