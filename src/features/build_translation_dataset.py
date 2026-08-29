"""
Attach TRANSLATION labels + temporal cutoffs to the candidate pool.

Task definition
---------------
Label: ``translated_cz = 1`` iff the Goodreads work was matched (by the
NKC -> Goodreads cascade) to at least one Czech translation record whose first
Czech publication year falls inside [LABEL_MIN_YEAR, LABEL_MAX_YEAR].
Everything else in the candidate pool is a negative: "publishers did not pick
this book (as far as we can tell)".

Temporal cutoffs (leakage control)
----------------------------------
Every row gets a ``cutoff_year``; all review-derived features must come from
STRICTLY BEFORE that year (enforced later by aggregate_reviews.py).

- Positives: cutoff = first Czech publication year. The model must only see
  what a publisher could have seen when deciding.
- Negatives: there is no translation year, so we assign a pseudo-cutoff
  sampled from the positives' cutoff-year distribution (restricted to years
  the book already existed: cutoff > pub_year). Without this, negatives would
  systematically get more review history than positives and the model would
  learn the cutoff artifact instead of the acquisition signal.

Rows dropped from the pool
--------------------------
- Works first translated BEFORE LABEL_MIN_YEAR — already translated, not
  candidates in the modeling window.
- Works first translated AFTER LABEL_MAX_YEAR — label censoring + the 2017
  snapshot cannot provide their pre-cutoff features.
- Negatives whose pub_year leaves no valid cutoff year to sample.

Known label noise (document in the thesis, do not hide)
-------------------------------------------------------
Only ~25% of NKC translation records match to Goodreads. A translated book
whose NKC record failed to match therefore looks like a NEGATIVE here. The
candidate-pool popularity floor reduces this (popular books match at ~2x the
base rate), but the negative class remains noisy. This replaces the old
task's selection-bias caveat as the headline limitation.

Inputs
------
- data/interim/candidate_pool.csv    (build_candidate_pool.py)
- data/interim/matched_dataset.csv   (build_matched_dataset.py)

Output
------
- data/interim/translation_labels.csv — one row per work:
    work_id, rep_book_id, title, language, pub_year, n_editions,
    ratings_count_2017 (metadata only), shelf_*,
    cutoff_year, translated_cz

Usage
-----
    python src/features/build_translation_dataset.py     # seconds
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO    = Path(__file__).resolve().parents[2]
INTERIM = REPO / "data" / "interim"

POOL_CSV    = INTERIM / "candidate_pool.csv"
MATCHED_CSV = INTERIM / "matched_dataset.csv"
OUT_CSV     = INTERIM / "translation_labels.csv"

# Modeling window for the label.
# LABEL_MIN_YEAR = 2008, not 2003: Goodreads launched in 2007, so pre-cutoff
# review features are structurally zero for earlier translation years — those
# rows would only teach the model "no signal -> translated anyway" noise.
LABEL_MIN_YEAR = 2008
LABEL_MAX_YEAR = 2017   # 2017 snapshot; later labels are censored

RANDOM_STATE = 0


def first_czech_year_per_work(matched: pd.DataFrame) -> pd.Series:
    """gr_work_id -> earliest known Czech publication year (int)."""
    m = matched[matched["gr_work_id"].fillna("").str.len() > 0].copy()

    # Prefer first_czech_year (earliest translation of the work, computed by
    # the cascade step); fall back to the row's czech_pub_year.
    year_col = ("first_czech_year" if "first_czech_year" in m.columns
                else "czech_pub_year")
    year = pd.to_numeric(m[year_col], errors="coerce")
    fallback = pd.to_numeric(m["czech_pub_year"], errors="coerce")
    m["_year"] = year.fillna(fallback)
    m = m.dropna(subset=["_year"])
    m["_year"] = m["_year"].astype(int)

    return m.groupby("gr_work_id")["_year"].min()


def main() -> pd.DataFrame:
    print("Loading candidate pool …", flush=True)
    pool = pd.read_csv(POOL_CSV, dtype={"work_id": str, "rep_book_id": str})
    print(f"  {len(pool):,} works in pool")

    print("Loading matched dataset …", flush=True)
    matched = pd.read_csv(MATCHED_CSV, dtype=str, keep_default_na=False)
    first_cz = first_czech_year_per_work(matched)
    print(f"  {len(first_cz):,} Goodreads works have >=1 matched Czech translation")

    pool["first_cz_year"] = pool["work_id"].map(first_cz)

    # ── Partition the pool ────────────────────────────────────────────────────
    translated_pre  = pool["first_cz_year"] < LABEL_MIN_YEAR
    translated_post = pool["first_cz_year"] > LABEL_MAX_YEAR
    positive        = pool["first_cz_year"].between(LABEL_MIN_YEAR, LABEL_MAX_YEAR)
    negative        = pool["first_cz_year"].isna()

    n_lost_positives = int((~first_cz.index.isin(pool["work_id"])).sum())

    print(f"\nPool partition:")
    print(f"  positives (translated {LABEL_MIN_YEAR}-{LABEL_MAX_YEAR}) : "
          f"{int(positive.sum()):>8,}")
    print(f"  dropped (translated <{LABEL_MIN_YEAR})          : "
          f"{int(translated_pre.sum()):>8,}")
    print(f"  dropped (translated >{LABEL_MAX_YEAR})          : "
          f"{int(translated_post.sum()):>8,}")
    print(f"  negatives (never matched)             : {int(negative.sum()):>8,}")
    print(f"  matched works NOT in pool (below floor/filtered) : "
          f"{n_lost_positives:,}")

    df = pool[positive | negative].copy()
    df["translated_cz"] = positive[positive | negative].astype(int).values

    # ── Cutoff years ──────────────────────────────────────────────────────────
    df["cutoff_year"] = df["first_cz_year"]

    pos_years = df.loc[df["translated_cz"] == 1, "cutoff_year"].astype(int)
    year_values, year_counts = np.unique(pos_years, return_counts=True)
    year_probs = year_counts / year_counts.sum()

    rng = np.random.default_rng(RANDOM_STATE)
    neg_mask = df["translated_cz"] == 0

    # Sample each negative's cutoff from the positive year distribution,
    # restricted to years strictly after the book's publication year.
    neg_pub = df.loc[neg_mask, "pub_year"].to_numpy()
    sampled = np.empty(len(neg_pub), dtype=float)
    unsatisfiable = 0
    for i, py in enumerate(neg_pub):
        valid = year_values > py
        if not valid.any():
            sampled[i] = np.nan          # book newer than all positive years
            unsatisfiable += 1
            continue
        p = year_probs[valid] / year_probs[valid].sum()
        sampled[i] = rng.choice(year_values[valid], p=p)
    df.loc[neg_mask, "cutoff_year"] = sampled

    n_before = len(df)
    df = df.dropna(subset=["cutoff_year"])
    df["cutoff_year"] = df["cutoff_year"].astype(int)
    print(f"\nNegatives with no valid cutoff (pub_year too late), dropped: "
          f"{unsatisfiable:,} ({n_before - len(df):,} rows removed)")

    # Sanity: every row's book existed before its cutoff.
    bad = (df["pub_year"] >= df["cutoff_year"]).sum()
    df = df[df["pub_year"] < df["cutoff_year"]]
    if bad:
        print(f"Dropped {bad:,} rows with pub_year >= cutoff_year "
              f"(noisy pub years on positives)")

    df = df.drop(columns=["first_cz_year"])

    # ── Save + summary ────────────────────────────────────────────────────────
    df.to_csv(OUT_CSV, index=False)
    n_pos = int(df["translated_cz"].sum())
    print(f"\n=== Summary ===")
    print(f"Rows      : {len(df):>10,}")
    print(f"Positives : {n_pos:>10,}  ({n_pos / len(df):.2%})")
    print(f"Negatives : {len(df) - n_pos:>10,}")
    print(f"Cutoff years: {int(df['cutoff_year'].min())}-"
          f"{int(df['cutoff_year'].max())}")
    print(f"→ {OUT_CSV}")
    return df


if __name__ == "__main__":
    main()
