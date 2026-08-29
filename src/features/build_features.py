"""
Build the final model-ready dataset for TRANSLATION PREDICTION.

Merges translation_labels.csv (candidate pool + labels + cutoffs) with
pre_cutoff_stats.csv (leakage-free review aggregates) and derives the feature
matrix, then writes a temporal train/val/test split.

Features (all knowable before the cutoff year)
----------------------------------------------
Popularity (pre-cutoff, work-aggregated):
- pre_cutoff_ratings_count, log_pre_cutoff_ratings_count
- pre_cutoff_avg_rating           (imputed 3.5 when no ratings)
- has_precutoff_signal            (>= 5 pre-cutoff ratings)
- log_pre_cutoff_text_reviews

Genre (shares over genre-bucket shelf counts, from the candidate pool):
- shelf_fiction, shelf_mystery, shelf_romance, shelf_scifi,
  shelf_nonfiction, shelf_ya, shelf_classics

Language of the original (dummies): lang_eng, lang_ger, lang_fre,
  lang_spa, lang_ita, lang_swe  (everything else / unknown = all zeros)

Book age: age_at_cutoff = cutoff_year - pub_year

Split key (NOT a model feature): cutoff_year

Metadata kept in dataset.csv only (leakage audit): ratings_count_2017

Temporal split
--------------
train: cutoff_year <= 2015 | val: 2016 | test: 2017
The task is "predict future acquisitions from past ones", so the split is by
time, never random.

Inputs  : data/interim/translation_labels.csv, data/interim/pre_cutoff_stats.csv
Outputs : data/processed/dataset.csv           (features + label + metadata)
          data/processed/splits/{X,y}_{train,val,test}.csv

Usage
-----
    python src/features/build_features.py       # seconds
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO    = Path(__file__).resolve().parents[2]
INTERIM = REPO / "data" / "interim"
PROC    = REPO / "data" / "processed"
SPLITS  = PROC / "splits"

LABELS_CSV = INTERIM / "translation_labels.csv"
STATS_CSV  = INTERIM / "pre_cutoff_stats.csv"

LABEL_COL = "translated_cz"
YEAR_COL  = "cutoff_year"

TRAIN_MAX_YEAR = 2015
VAL_YEAR       = 2016
TEST_YEAR      = 2017

LANGS = ["eng", "ger", "fre", "spa", "ita", "swe"]
SHELF_COLS = ["shelf_fiction", "shelf_mystery", "shelf_romance", "shelf_scifi",
              "shelf_nonfiction", "shelf_ya", "shelf_classics"]

# Books with zero pre-cutoff ratings are typically obscure; imputing the global
# median (~4.0) would overstate their quality. 3.5 = mid-scale neutral.
IMPUTE_AVG_RATING = 3.5

# Kept in dataset.csv for auditing, never in X (temporal leakage / identifiers).
METADATA_COLS = ["work_id", "rep_book_id", "title", "language", "pub_year",
                 "n_editions", "ratings_count_2017", "text_reviews_2017",
                 LABEL_COL]


def build_dataset() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_CSV, dtype={"work_id": str, "rep_book_id": str})
    stats  = pd.read_csv(STATS_CSV,  dtype={"work_id": str})

    df = labels.merge(
        stats[["work_id", "pre_cutoff_ratings_count", "pre_cutoff_avg_rating",
               "pre_cutoff_reviews_with_text"]],
        on="work_id", how="left",
    )
    assert len(df) == len(labels), "merge bloat — duplicate work_ids in stats"

    df["pre_cutoff_ratings_count"] = (
        df["pre_cutoff_ratings_count"].fillna(0).astype(int))
    df["pre_cutoff_reviews_with_text"] = (
        df["pre_cutoff_reviews_with_text"].fillna(0).astype(int))

    # ── Derived features ──────────────────────────────────────────────────────
    df["log_pre_cutoff_ratings_count"] = np.log1p(df["pre_cutoff_ratings_count"])
    df["log_pre_cutoff_text_reviews"]  = np.log1p(df["pre_cutoff_reviews_with_text"])
    df["has_precutoff_signal"] = (df["pre_cutoff_ratings_count"] >= 5).astype(int)
    df["pre_cutoff_avg_rating"] = df["pre_cutoff_avg_rating"].fillna(IMPUTE_AVG_RATING)
    df["age_at_cutoff"] = (df[YEAR_COL] - df["pub_year"]).astype(int)

    lang = df["language"].fillna("").str.lower()
    for lg in LANGS:
        df[f"lang_{lg}"] = (lang == lg).astype(int)

    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Model features — excludes label, split key, and metadata."""
    excluded = set(METADATA_COLS) | {YEAR_COL, "pre_cutoff_reviews_with_text"}
    return [c for c in df.columns if c not in excluded]


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masks = {
        "train": df[YEAR_COL] <= TRAIN_MAX_YEAR,
        "val":   df[YEAR_COL] == VAL_YEAR,
        "test":  df[YEAR_COL] == TEST_YEAR,
    }
    total = sum(int(m.sum()) for m in masks.values())
    assert total == len(df), "split masks do not partition the dataset"
    return {name: df[m].reset_index(drop=True) for name, m in masks.items()}


def main() -> pd.DataFrame:
    df = build_dataset()
    fcols = feature_cols(df)

    PROC.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / "dataset.csv", index=False)

    parts = temporal_split(df)
    print(f"Dataset: {len(df):,} rows, {int(df[LABEL_COL].sum()):,} positives "
          f"({df[LABEL_COL].mean():.2%})")
    print(f"Features ({len(fcols)}): {fcols}\n")

    for name, part in parts.items():
        part[fcols].to_csv(SPLITS / f"X_{name}.csv", index=False)
        part[[LABEL_COL]].to_csv(SPLITS / f"y_{name}.csv", index=False)
        print(f"  {name:<5} : {len(part):>8,} rows | "
              f"{int(part[LABEL_COL].sum()):>6,} pos "
              f"({part[LABEL_COL].mean():.2%}) | years "
              f"{int(part[YEAR_COL].min())}-{int(part[YEAR_COL].max())}")

    print(f"\n→ {PROC / 'dataset.csv'}")
    print(f"→ {SPLITS}/X_*.csv, y_*.csv")
    return df


if __name__ == "__main__":
    main()
