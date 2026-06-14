"""
Aggregate pre-cutoff Goodreads reviews for each training book, WORK-AGGREGATED.

For every training row we look at *all* Goodreads editions of the same work
(grouped by ``gr_work_id``), accumulate pre-cutoff reviews across them, and
assign the total back to the row's ``matched_book_id``.

Why work-aggregation
--------------------
Goodreads stores each printing/translation/format of a work as a separate
``book_id``. The cascade matcher in build_matched_dataset.py picks one edition
per NKC record (tie-break by language match → pub_year proximity → ratings
count). For non-English originals or minor English editions, this can land on
a book_id with very few ratings while the work as a whole has thousands.

Example: Wilbur Smith's *Desert God* — our matched edition has 10 ratings in
the 2017 snapshot, while the work_id covers ~30K across all editions. A model
feature computed on the matched edition would severely under-state reception.

What the publisher actually sees on Goodreads is the work page, which displays
work-level aggregated counts. Work-aggregation matches that view.

Temporal cutoff
---------------
Each work gets the cutoff year from its associated training row's
``czech_pub_year``. Reviews with ``date_added`` strictly less than that year
are counted; later reviews are dropped (no leakage).

Outputs
-------
``data/interim/pre_cutoff_stats.parquet`` (and ``.csv`` backup) with one row
per matched_book_id in training_dataset.csv:

    matched_book_id, czech_pub_year,
    pre_cutoff_ratings_count,           (work-aggregated)
    pre_cutoff_avg_rating,              (work-aggregated)
    pre_cutoff_reviews_with_text        (work-aggregated)

Schema is unchanged from v1 — feature_engineering notebook needs no edits.
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
W2B_JSON    = REPO / "data" / "interim" / "goodreads_work_to_books.json"
OUT_PARQUET = REPO / "data" / "interim" / "pre_cutoff_stats.parquet"
OUT_CSV     = REPO / "data" / "interim" / "pre_cutoff_stats.csv"


# ── Phase 1: build expanded book_id → (work_key, cutoff_year) map ────────────
#
# Books in a training row's work_id share the row's cutoff. Books without a
# work_id (rare, but possible if Goodreads dump quirks) fall back to a
# "_solo:<book_id>" key — same behaviour as edition-level aggregation.

print("Loading training dataset …", flush=True)
train = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
print(f"  {len(train):,} training rows", flush=True)

print("Loading goodreads_work_to_books.json …", flush=True)
with open(W2B_JSON, encoding="utf-8") as f:
    work_to_books: dict[str, list[str]] = json.load(f)
print(f"  {len(work_to_books):,} works in catalogue", flush=True)

work_cutoffs:  dict[str, int]  = {}    # work_key → earliest czech_pub_year
expanded_book_to_work: dict[str, str] = {}   # book_id → work_key

solo_count = 0
for _, row in train.iterrows():
    bid = row.get("matched_book_id", "").strip()
    wid = row.get("gr_work_id", "").strip()
    try:
        year = int(float(row["czech_pub_year"]))
    except (ValueError, TypeError):
        continue
    if not bid:
        continue

    if wid:
        key = wid
    else:
        key = f"_solo:{bid}"
        solo_count += 1

    # Same work could appear twice in training only if dedup let it slip —
    # defensive: keep the earlier (more restrictive) cutoff.
    if key not in work_cutoffs or year < work_cutoffs[key]:
        work_cutoffs[key] = year

# Expand: every book_id in any touched work inherits the work's cutoff.
for key in work_cutoffs:
    if key.startswith("_solo:"):
        bid = key[len("_solo:"):]
        expanded_book_to_work[bid] = key
    else:
        for bid in work_to_books.get(key, []):
            # If a book_id appears in multiple touched works (would require a
            # work_id collision in the dump — shouldn't happen, but defensive),
            # use the earliest cutoff. Since we iterate in dict order, just
            # check and keep earliest.
            prior = expanded_book_to_work.get(bid)
            if prior is None or work_cutoffs[key] < work_cutoffs[prior]:
                expanded_book_to_work[bid] = key

n_touched_works = sum(1 for k in work_cutoffs if not k.startswith("_solo:"))
print(f"  Touched works                : {n_touched_works:,}")
print(f"  Solo (no work_id) entries    : {solo_count:,}")
print(f"  Expanded book_ids (all eds.) : {len(expanded_book_to_work):,}",
      flush=True)
print(f"  Expansion factor             : "
      f"{len(expanded_book_to_work) / max(len(work_cutoffs), 1):.1f}x", flush=True)


# ── Phase 2: stream reviews dump, accumulate by work_key ─────────────────────

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

        wkey = expanded_book_to_work.get(bid)
        if wkey is None:
            continue

        # Parse date_added; cutoff is per-work.
        date_raw = obj.get("date_added", "")
        if not date_raw:
            continue
        try:
            review_year = datetime.strptime(date_raw, "%a %b %d %H:%M:%S %z %Y").year
        except ValueError:
            continue

        if review_year >= work_cutoffs[wkey]:
            continue  # post-cutoff — drop

        # Accumulate ratings (1..5 only; shelved-without-rating == 0).
        rating = obj.get("rating", 0)
        if isinstance(rating, int) and 1 <= rating <= 5:
            ratings_count[wkey] += 1
            ratings_sum[wkey]   += rating
            relevant_reviews    += 1

        # Reviews with non-empty text — counted regardless of rating presence.
        review_text = obj.get("review_text", "")
        if isinstance(review_text, str) and review_text.strip():
            text_count[wkey] += 1

elapsed_total = time.time() - t0
print(f"  {total_lines:>10,} lines  |  {relevant_reviews:>7,} relevant  "
      f"|  {elapsed_total:.0f}s elapsed  ← done", flush=True)


# ── Phase 3: project work aggregates back to per-matched_book_id rows ────────

print("\nAssembling output …", flush=True)
rows = []
for _, row in train.iterrows():
    bid = row.get("matched_book_id", "").strip()
    wid = row.get("gr_work_id", "").strip()
    try:
        cutoff_year = int(float(row["czech_pub_year"]))
    except (ValueError, TypeError):
        continue
    if not bid:
        continue

    key = wid or f"_solo:{bid}"
    rc = ratings_count.get(key, 0)
    rs = ratings_sum.get(key, 0)
    rows.append({
        "matched_book_id":              bid,
        "czech_pub_year":               cutoff_year,
        "pre_cutoff_ratings_count":     rc,
        "pre_cutoff_avg_rating":        round(rs / rc, 4) if rc > 0 else None,
        "pre_cutoff_reviews_with_text": text_count.get(key, 0),
    })

df = pd.DataFrame(rows)

# Deduplicate by matched_book_id. Two training rows can share a matched_book_id
# when the cascade mapped slightly-different NKC original_title variants to the
# same Goodreads edition (e.g., 'Animal farm' and 'Animal farm : a fairy story').
# After dedup at (author, title) in build_matched_dataset, those become separate
# training rows but still share matched_book_id. Their work-aggregated stats are
# identical, so dropping duplicates here keeps the file clean and prevents a
# cross-product blow-up when downstream code merges on matched_book_id.
n_before = len(df)
df = df.drop_duplicates(subset="matched_book_id", keep="first")
n_collapsed = n_before - len(df)
if n_collapsed:
    print(f"Deduplicated output: {n_before:,} → {len(df):,} rows  "
          f"({n_collapsed:,} duplicate matched_book_ids removed)")

df.to_parquet(OUT_PARQUET, engine="pyarrow", index=False)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df):,} rows → {OUT_PARQUET.name}  +  {OUT_CSV.name}")


# ── Summary ───────────────────────────────────────────────────────────────────

zero_reviews = (df["pre_cutoff_ratings_count"] == 0).sum()
nonzero      = df[df["pre_cutoff_ratings_count"] > 0]["pre_cutoff_ratings_count"]

print("\n=== Summary ===")
print(f"Total lines processed             : {total_lines:>10,}")
print(f"Relevant pre-cutoff reviews       : {relevant_reviews:>10,}")
print(f"Unique books in output            : {len(df):>10,}")
print(f"Books with zero pre-cutoff reviews: {zero_reviews:>10,}  "
      f"({zero_reviews/len(df):.1%})")
print(f"  (v1 edition-level value was ~48.4%; expect a drop under work-aggregation)")

if len(nonzero):
    print("\npre_cutoff_ratings_count distribution (books with ≥ 1 review):")
    desc = nonzero.describe(percentiles=[.25, .5, .75])
    for stat, label in [("min","min"), ("25%","p25"), ("50%","median"),
                        ("75%","p75"), ("max","max")]:
        print(f"  {label:<8} : {int(desc[stat]):>7,}")
