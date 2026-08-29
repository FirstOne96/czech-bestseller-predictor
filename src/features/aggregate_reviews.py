"""
Aggregate PRE-CUTOFF Goodreads reviews for every work in translation_labels.csv.

For each labeled work we look at *all* its Goodreads editions (via
``goodreads_work_to_books.json``), accumulate reviews dated STRICTLY BEFORE the
work's ``cutoff_year``, and emit one stats row per work.

Why work-aggregation
--------------------
Goodreads stores each printing/format as a separate ``book_id``; a single
edition can have 10 ratings while the work has 30,000. What a publisher sees
on Goodreads is the work page (aggregated), so features are computed at the
work level.

Why pre-cutoff
--------------
The model must only use information available when the acquisition decision
was made (positives: first Czech publication year; negatives: assigned
pseudo-cutoff — see build_translation_dataset.py). Using the 2017 snapshot
totals would leak post-decision popularity into the features.

Input
-----
- data/interim/translation_labels.csv       (work_id, cutoff_year, …)
- data/interim/goodreads_work_to_books.json (work_id -> [book_id, …])
- data/raw/goodreads_reviews_dedup.json.gz  (15 GB reviews stream)

Output
------
- data/interim/pre_cutoff_stats.csv / .parquet — one row per work_id:
    work_id, cutoff_year,
    pre_cutoff_ratings_count, pre_cutoff_avg_rating,
    pre_cutoff_reviews_with_text

Usage
-----
    python src/features/aggregate_reviews.py     # ~15-25 min, streams 15 GB
"""
from __future__ import annotations

import gzip
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO        = Path(__file__).resolve().parents[2]
REVIEWS_GZ  = REPO / "data" / "raw"     / "goodreads_reviews_dedup.json.gz"
LABELS_CSV  = REPO / "data" / "interim" / "translation_labels.csv"
W2B_JSON    = REPO / "data" / "interim" / "goodreads_work_to_books.json"
OUT_PARQUET = REPO / "data" / "interim" / "pre_cutoff_stats.parquet"
OUT_CSV     = REPO / "data" / "interim" / "pre_cutoff_stats.csv"


def main() -> None:
    # ── Phase 1: book_id -> (work_id, cutoff) map for all labeled works ──────
    print("Loading translation labels …", flush=True)
    labels = pd.read_csv(LABELS_CSV,
                         dtype={"work_id": str, "rep_book_id": str})
    print(f"  {len(labels):,} labeled works", flush=True)

    print("Loading goodreads_work_to_books.json …", flush=True)
    with open(W2B_JSON, encoding="utf-8") as f:
        work_to_books: dict[str, list[str]] = json.load(f)

    work_cutoff: dict[str, int] = dict(
        zip(labels["work_id"], labels["cutoff_year"].astype(int)))

    book_to_work: dict[str, str] = {}
    for wid in work_cutoff:
        for bid in work_to_books.get(wid, []):
            book_to_work[bid] = wid

    print(f"  Expanded to {len(book_to_work):,} edition book_ids "
          f"({len(book_to_work) / max(len(work_cutoff), 1):.1f}x)", flush=True)

    # ── Phase 2: stream reviews, accumulate pre-cutoff stats per work ────────
    ratings_count: dict[str, int] = defaultdict(int)
    ratings_sum:   dict[str, int] = defaultdict(int)
    text_count:    dict[str, int] = defaultdict(int)

    total_lines = relevant = 0
    t0 = time.time()
    print(f"\nStreaming {REVIEWS_GZ.name} …", flush=True)

    with gzip.open(REVIEWS_GZ, "rt", encoding="utf-8") as fh:
        for line in fh:
            total_lines += 1
            if total_lines % 1_000_000 == 0:
                print(f"  {total_lines:>10,} lines | {relevant:>9,} relevant | "
                      f"{time.time() - t0:.0f}s", flush=True)

            obj = json.loads(line)
            wid = book_to_work.get(obj.get("book_id", ""))
            if wid is None:
                continue

            date_raw = obj.get("date_added", "")
            if not date_raw:
                continue
            try:
                review_year = datetime.strptime(
                    date_raw, "%a %b %d %H:%M:%S %z %Y").year
            except ValueError:
                continue

            if review_year >= work_cutoff[wid]:
                continue  # post-cutoff — drop (leakage control)

            rating = obj.get("rating", 0)
            if isinstance(rating, int) and 1 <= rating <= 5:
                ratings_count[wid] += 1
                ratings_sum[wid]   += rating
                relevant += 1

            text = obj.get("review_text", "")
            if isinstance(text, str) and text.strip():
                text_count[wid] += 1

    print(f"  {total_lines:>10,} lines | {relevant:>9,} relevant | "
          f"{time.time() - t0:.0f}s ← done", flush=True)

    # ── Phase 3: one output row per labeled work ─────────────────────────────
    rows = []
    for wid, cutoff in work_cutoff.items():
        rc = ratings_count.get(wid, 0)
        rs = ratings_sum.get(wid, 0)
        rows.append({
            "work_id": wid,
            "cutoff_year": cutoff,
            "pre_cutoff_ratings_count": rc,
            "pre_cutoff_avg_rating": round(rs / rc, 4) if rc else None,
            "pre_cutoff_reviews_with_text": text_count.get(wid, 0),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, engine="pyarrow", index=False)
    df.to_csv(OUT_CSV, index=False)

    zero = int((df["pre_cutoff_ratings_count"] == 0).sum())
    print("\n=== Summary ===")
    print(f"Works in output                    : {len(df):>10,}")
    print(f"Works with zero pre-cutoff ratings : {zero:>10,} "
          f"({zero / len(df):.1%})")
    print(f"→ {OUT_PARQUET.name}  +  {OUT_CSV.name}")


if __name__ == "__main__":
    main()
