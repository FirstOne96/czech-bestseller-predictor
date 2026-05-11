"""
Aggregate pre-cutoff Goodreads reviews for each book in the training dataset.

For each book, we only count reviews whose date_added year is strictly less than
the Czech publication year — the temporal cutoff that prevents leakage.  Books
that were translated later had less time to accumulate ratings, so this also
makes features comparable across the cohort.

Output: data/interim/pre_cutoff_stats.parquet  (and .csv backup)
"""
import gzip
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO        = Path(__file__).resolve().parents[2]
REVIEWS_GZ  = REPO / "data" / "raw"     / "goodreads_reviews_dedup.json.gz"
TRAIN_CSV   = REPO / "data" / "interim" / "training_dataset.csv"
OUT_PARQUET = REPO / "data" / "interim" / "pre_cutoff_stats.parquet"
OUT_CSV     = REPO / "data" / "interim" / "pre_cutoff_stats.csv"

# ── Build matched_book_id → czech_pub_year lookup ────────────────────────────

print("Loading training dataset …", flush=True)
train = pd.read_csv(TRAIN_CSV, dtype=str)

book_cutoffs: dict[str, int] = {}
for _, row in train.iterrows():
    bid = str(row.get("matched_book_id", "")).strip()
    try:
        year = int(float(row["czech_pub_year"]))
    except (ValueError, TypeError):
        continue
    if bid and (bid not in book_cutoffs or year < book_cutoffs[bid]):
        book_cutoffs[bid] = year

print(f"  {len(book_cutoffs):,} books with a cutoff year")

# ── Accumulate per-book pre-cutoff statistics ─────────────────────────────────

ratings_count: dict[str, int] = defaultdict(int)
ratings_sum:   dict[str, int] = defaultdict(int)
text_count:    dict[str, int] = defaultdict(int)

total_lines = relevant_reviews = 0
t0 = time.time()

print(f"\nStreaming {REVIEWS_GZ.name} …", flush=True)

with gzip.open(REVIEWS_GZ, "rt", encoding="utf-8") as fh:
    for line in fh:
        total_lines += 1

        if total_lines % 1_000_000 == 0:
            elapsed = time.time() - t0
            print(
                f"  {total_lines:>10,} lines  |  {relevant_reviews:>7,} relevant  "
                f"|  {elapsed:.0f}s elapsed",
                flush=True,
            )

        obj = json.loads(line)
        bid = obj.get("book_id", "")

        # Fast path: skip books not in our training set
        if bid not in book_cutoffs:
            continue

        # Parse date and apply temporal cutoff
        date_raw = obj.get("date_added", "")
        if not date_raw:
            continue
        try:
            review_year = datetime.strptime(date_raw, "%a %b %d %H:%M:%S %z %Y").year
        except ValueError:
            continue

        if review_year >= book_cutoffs[bid]:
            continue  # post-cutoff review — discard

        # Accumulate rating (skip shelved-without-rating entries where rating == 0)
        rating = obj.get("rating", 0)
        if isinstance(rating, int) and 1 <= rating <= 5:
            ratings_count[bid] += 1
            ratings_sum[bid]   += rating
            relevant_reviews   += 1

        # Count reviews that have text regardless of rating
        review_text = obj.get("review_text", "")
        if isinstance(review_text, str) and review_text.strip():
            text_count[bid] += 1

elapsed_total = time.time() - t0
print(f"  {total_lines:>10,} lines  |  {relevant_reviews:>7,} relevant  "
      f"|  {elapsed_total:.0f}s elapsed  ← done", flush=True)

# ── Build output dataframe (one row per training book) ────────────────────────

print("\nAssembling output …", flush=True)
rows = []
for bid, cutoff_year in book_cutoffs.items():
    rc = ratings_count.get(bid, 0)
    rs = ratings_sum.get(bid, 0)
    rows.append({
        "matched_book_id":          bid,
        "czech_pub_year":           int(cutoff_year),
        "pre_cutoff_ratings_count": rc,
        "pre_cutoff_avg_rating":    round(rs / rc, 4) if rc > 0 else None,
        "pre_cutoff_reviews_with_text": text_count.get(bid, 0),
    })

df = pd.DataFrame(rows)

# ── Save ──────────────────────────────────────────────────────────────────────

df.to_parquet(OUT_PARQUET, engine="pyarrow", index=False)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df):,} rows → {OUT_PARQUET.name}  +  {OUT_CSV.name}")

# ── Summary ───────────────────────────────────────────────────────────────────

zero_reviews = (df["pre_cutoff_ratings_count"] == 0).sum()
nonzero      = df[df["pre_cutoff_ratings_count"] > 0]["pre_cutoff_ratings_count"]

print("\n=== Summary ===")
print(f"Total lines processed          : {total_lines:>10,}")
print(f"Relevant pre-cutoff reviews    : {relevant_reviews:>10,}")
print(f"Unique books in output         : {len(df):>10,}")
print(f"Books with zero pre-cutoff reviews: {zero_reviews:>7,}  ({zero_reviews/len(df):.1%})  ← need special handling")

if len(nonzero):
    print("\npre_cutoff_ratings_count distribution (books with ≥ 1 review):")
    desc = nonzero.describe(percentiles=[.25, .5, .75])
    for stat, label in [("min","min"), ("25%","p25"), ("50%","median"), ("75%","p75"), ("max","max")]:
        print(f"  {label:<8} : {int(desc[stat]):>7,}")
