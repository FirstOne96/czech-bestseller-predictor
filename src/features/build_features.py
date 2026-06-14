"""
Build the model-ready feature matrix from training_dataset.csv + pre_cutoff_stats.csv.

Inputs (data/interim/)
-----------------------
- training_dataset.csv       — NKC + Goodreads + SCKN labels (from build_matched_dataset.py)
- pre_cutoff_stats.csv        — pre-cutoff review aggregates, no leakage (from aggregate_reviews.py)
- goodreads_author_names.json — author_id -> name (kept for reference, not used as a feature)

Outputs (data/processed/)
--------------------------
- training_features.csv — full feature matrix, label + metadata columns retained
- X_train.csv            — feature columns only (no label, no leakage-prone metadata)
- y_train.csv            — label column only (sckn_bestseller)

Known limitations
------------------
- Goodreads snapshot is from 2017. Books with czech_pub_year > 2017 have
  systematically lower pre-cutoff signal than they would in reality.
- Overall NKC -> Goodreads match rate is ~25%. The model is implicitly
  conditioned on a book being findable in the Goodreads catalogue.
- Pre-cutoff signal is sparse: a large share of training records have zero
  pre-cutoff ratings. `has_precutoff_signal` flags records with >= 5 ratings.

Usage
-----
    python src/features/build_features.py        # regenerate data/processed/*.csv

Or import the building blocks from a notebook for inspection:

    from src.features.build_features import load_inputs, merge_precutoff, build_feature_matrix
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO    = Path(__file__).resolve().parents[2]
INTERIM = REPO / "data" / "interim"
PROC    = REPO / "data" / "processed"

# gr_popular_shelves is a JSON list of {"name": ..., "count": ...} dicts. Each
# book's shelf counts are bucketed into these coarse genre groups and expressed
# as a share of the book's total shelf count.
SHELF_BUCKETS = {
    "fiction":     {"fiction", "literary-fiction", "contemporary", "literary"},
    "mystery":     {"mystery", "thriller", "crime", "suspense", "detective"},
    "romance":     {"romance", "love", "chick-lit"},
    "scifi":       {"science-fiction", "sci-fi", "fantasy", "speculative-fiction"},
    "nonfiction":  {"non-fiction", "nonfiction", "biography", "memoir",
                    "history", "self-help", "true-crime"},
    "ya":          {"young-adult", "ya", "teen", "childrens", "children"},
    "classics":    {"classics", "classic", "literary-classics"},
}

# Top source languages in the NKC cohort.
SOURCE_LANGS = ["eng", "ger", "fre", "rus"]

# Books with zero pre-cutoff ratings are typically obscure titles; imputing the
# global median (~4.0) would overstate their quality. Use 3.5 (mid-scale neutral).
IMPUTE_AVG_RATING = 3.5

# Columns kept in training_features.csv for context/leakage-auditing but
# excluded from X_train.csv.
METADATA_COLS = {"sckn_bestseller", "gr_ratings_count"}


# ── Step 1 — load & merge ────────────────────────────────────────────────────

def load_inputs(interim: Path = INTERIM):
    """Load the three raw inputs. Returns (train_df, pre_cutoff_df, author_names)."""
    train = pd.read_csv(interim / "training_dataset.csv", dtype=str)
    pre = pd.read_csv(interim / "pre_cutoff_stats.csv", dtype=str)
    with open(interim / "goodreads_author_names.json", encoding="utf-8") as f:
        author_names: dict[str, str] = json.load(f)
    return train, pre, author_names


def merge_precutoff(train: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """Left-join training rows with pre-cutoff review aggregates on matched_book_id.

    pre_cutoff_stats can contain duplicate matched_book_id rows when multiple
    training rows share the same Goodreads edition (different NKC
    original_title variants that the cascade mapped to the same book_id).
    Since the stats are work-aggregated, every duplicate carries identical
    values, so we drop them before the merge to prevent a cross-product blow-up.
    """
    pre_unique = pre.drop_duplicates(subset="matched_book_id", keep="first")

    df = train.merge(
        pre_unique[["matched_book_id", "pre_cutoff_ratings_count",
                     "pre_cutoff_avg_rating", "pre_cutoff_reviews_with_text"]],
        on="matched_book_id",
        how="left",
    )
    assert len(df) == len(train), f"merge bloat — {len(df)} != {len(train)}"

    df["pre_cutoff_ratings_count"] = pd.to_numeric(
        df["pre_cutoff_ratings_count"], errors="coerce").fillna(0).astype(int)
    df["pre_cutoff_avg_rating"] = pd.to_numeric(
        df["pre_cutoff_avg_rating"], errors="coerce")
    df["pre_cutoff_reviews_with_text"] = pd.to_numeric(
        df["pre_cutoff_reviews_with_text"], errors="coerce").fillna(0).astype(int)

    return df


# ── Step 2 — feature blocks ──────────────────────────────────────────────────

def shelf_shares(shelves_json: str) -> dict[str, float]:
    """Return genre-bucket share of total shelf counts for one book."""
    try:
        shelves = json.loads(shelves_json) if isinstance(shelves_json, str) else []
    except (json.JSONDecodeError, TypeError):
        shelves = []

    bucket_totals = {b: 0 for b in SHELF_BUCKETS}
    grand_total = 0

    for entry in shelves:
        name = str(entry.get("name", "")).lower().replace(" ", "-")
        count = int(entry.get("count", 0) or 0)
        grand_total += count
        for bucket, keywords in SHELF_BUCKETS.items():
            if name in keywords:
                bucket_totals[bucket] += count

    if grand_total == 0:
        return {f"shelf_{b}": 0.0 for b in SHELF_BUCKETS}
    return {f"shelf_{b}": round(bucket_totals[b] / grand_total, 6) for b in SHELF_BUCKETS}


def _popularity_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["pre_cutoff_ratings_count"] = df["pre_cutoff_ratings_count"]
    feat["pre_cutoff_avg_rating"] = df["pre_cutoff_avg_rating"]
    # Log-transformed count (handles heavy tail; log1p avoids log(0)).
    feat["log_pre_cutoff_ratings_count"] = np.log1p(df["pre_cutoff_ratings_count"])
    # Binary flag: >= 5 pre-cutoff ratings -> enough signal to trust the average.
    feat["has_precutoff_signal"] = (df["pre_cutoff_ratings_count"] >= 5).astype(int)
    # Overall Goodreads ratings count — metadata only, excluded from X_train
    # (temporal leakage).
    feat["gr_ratings_count"] = pd.to_numeric(df["gr_ratings_count"], errors="coerce").fillna(0)
    return feat


def _shelf_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(df["gr_popular_shelves"].apply(shelf_shares).tolist(), index=df.index)


def _language_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    src = df["source_lang"].str.strip().str.lower()
    for lang in SOURCE_LANGS:
        feat[f"lang_{lang}"] = (src == lang).astype(int)
    return feat


def _temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["czech_pub_year"] = pd.to_numeric(df["czech_pub_year"], errors="coerce").astype("Int64")
    return feat


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full feature matrix (features + metadata + label) from the
    merged training/pre-cutoff dataframe produced by ``merge_precutoff``.
    """
    feat = pd.concat([
        _popularity_features(df),
        _shelf_features(df),
        _language_features(df),
        _temporal_features(df),
    ], axis=1)

    feat["sckn_bestseller"] = (df["sckn_bestseller"].astype(str).str.lower() == "true").astype(int)

    # Fixed imputation for pre_cutoff_avg_rating (see IMPUTE_AVG_RATING docstring above).
    feat["pre_cutoff_avg_rating"] = feat["pre_cutoff_avg_rating"].fillna(IMPUTE_AVG_RATING)

    return feat


def feature_cols(feat: pd.DataFrame) -> list[str]:
    """Feature columns only — excludes label and leakage-prone metadata."""
    return [c for c in feat.columns if c not in METADATA_COLS]


# ── Step 3 — save outputs ────────────────────────────────────────────────────

def save_outputs(feat: pd.DataFrame, proc: Path = PROC) -> None:
    proc.mkdir(parents=True, exist_ok=True)

    feat.to_csv(proc / "training_features.csv", index=False)

    X = feat[feature_cols(feat)]
    X.to_csv(proc / "X_train.csv", index=False)

    y = feat[["sckn_bestseller"]]
    y.to_csv(proc / "y_train.csv", index=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    train, pre, _author_names = load_inputs()
    df = merge_precutoff(train, pre)
    feat = build_feature_matrix(df)
    save_outputs(feat)

    X = feat[feature_cols(feat)]
    print(f"Saved {len(feat):,} rows -> training_features.csv")
    print(f"Saved X_train.csv  shape={X.shape}")
    print(f"Saved y_train.csv  shape=({len(feat)}, 1)")
    return feat


if __name__ == "__main__":
    main()
