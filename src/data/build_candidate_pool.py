"""
Build the work-level CANDIDATE POOL from the Goodreads books dump.

Why this exists
---------------
The task is: "given a foreign book, will Czech publishers pick it for
translation?" To train such a model we need not only the books that WERE
translated (positives, from the NKC match) but also a pool of comparable
foreign books that were NOT translated (negatives). This script builds that
pool: every sufficiently popular foreign work in the 2017 Goodreads dump,
translated or not. Labels are attached later by
``src/features/build_translation_dataset.py``.

Level of aggregation: WORK, not edition. Goodreads stores every
printing/translation as a separate ``book_id``; one ``work_id`` groups them.
We aggregate editions into one row per work and keep the most-rated edition
as the work's representative (title, language, shelves).

Filters applied at the work level
---------------------------------
- ``ratings_count`` (2017 snapshot, summed over editions) >= POOL_MIN_RATINGS.
  This is an EXISTENCE filter only — a publisher can only discover a book that
  has some visibility. The snapshot count itself must NOT be used as a model
  feature (temporal leakage); pre-cutoff counts come from the reviews dump.
- Original publication year within [MIN_PUB_YEAR, MAX_PUB_YEAR].
- Representative-edition language not Czech/Slovak (those are not *foreign
  candidates* — they are the translations themselves or domestic originals).

Output
------
``data/interim/candidate_pool.csv`` — one row per work:

    work_id, rep_book_id, title, language, pub_year, n_editions,
    ratings_count_2017, text_reviews_2017,
    shelf_fiction, shelf_mystery, shelf_romance, shelf_scifi,
    shelf_nonfiction, shelf_ya, shelf_classics

Shelf shares use a GENRE-ONLY denominator (share of counts that fall in any
genre bucket), so they are not diluted by "to-read"/"owned" shelves.

Usage
-----
    python src/data/build_candidate_pool.py          # ~20-30 min, streams 9 GB
"""
from __future__ import annotations

import csv
import gzip
import json
import time
from pathlib import Path

REPO     = Path(__file__).resolve().parents[2]
BOOKS_GZ = REPO / "data" / "raw" / "goodreads_books.json.gz"
OUT_CSV  = REPO / "data" / "interim" / "candidate_pool.csv"

# ── Tunables ──────────────────────────────────────────────────────────────────

POOL_MIN_RATINGS = 50     # work-level 2017 ratings floor (existence filter)
MIN_PUB_YEAR     = 1950   # drop ancient works — not realistic acquisitions
MAX_PUB_YEAR     = 2017   # dump snapshot year

# Goodreads language_code values are messy; normalize to coarse groups.
LANG_MAP = {
    "eng": "eng", "en-us": "eng", "en-gb": "eng", "en-ca": "eng", "en": "eng",
    "ger": "ger", "de": "ger", "deu": "ger",
    "fre": "fre", "fr": "fre", "fra": "fre",
    "spa": "spa", "es": "spa",
    "ita": "ita", "it": "ita",
    "swe": "swe", "sv": "swe",
    "cze": "cze", "cs": "cze", "ces": "cze",
    "slo": "slo", "sk": "slo", "slk": "slo",
}
EXCLUDED_LANGS = {"cze", "slo"}   # not foreign candidates

# Same buckets as before, but shares are computed over GENRE counts only.
SHELF_BUCKETS = {
    "fiction":    {"fiction", "literary-fiction", "contemporary", "literary"},
    "mystery":    {"mystery", "thriller", "crime", "suspense", "detective"},
    "romance":    {"romance", "love", "chick-lit"},
    "scifi":      {"science-fiction", "sci-fi", "fantasy", "speculative-fiction"},
    "nonfiction": {"non-fiction", "nonfiction", "biography", "memoir",
                   "history", "self-help", "true-crime"},
    "ya":         {"young-adult", "ya", "teen", "childrens", "children"},
    "classics":   {"classics", "classic", "literary-classics"},
}
BUCKET_NAMES = list(SHELF_BUCKETS)


def norm_lang(code: str) -> str:
    """Map a raw Goodreads language_code to a coarse group ('' stays '')."""
    return LANG_MAP.get((code or "").strip().lower(), (code or "").strip().lower())


def parse_int(v) -> int:
    try:
        return int(v) if v not in (None, "") else 0
    except (ValueError, TypeError):
        return 0


def shelf_bucket_counts(shelves: list) -> list[int]:
    """Per-bucket shelf counts for one edition's popular_shelves list."""
    counts = [0] * len(BUCKET_NAMES)
    for entry in shelves:
        name = str(entry.get("name", "")).lower().replace(" ", "-")
        c = parse_int(entry.get("count"))
        for i, bucket in enumerate(BUCKET_NAMES):
            if name in SHELF_BUCKETS[bucket]:
                counts[i] += c
    return counts


def main() -> None:
    # Per-work accumulator:
    # work_id -> [best_ratings, rep_book_id, title, lang, min_pub_year,
    #             sum_ratings, sum_text_reviews, n_editions, bucket_counts]
    works: dict[str, list] = {}

    total = 0
    t0 = time.time()
    print(f"Streaming {BOOKS_GZ.name} …", flush=True)

    with gzip.open(BOOKS_GZ, "rt", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            work_id = str(obj.get("work_id") or "").strip()
            book_id = obj.get("book_id", "")
            if not work_id or not book_id:
                continue

            total += 1
            if total % 200_000 == 0:
                print(f"  {total:,} editions processed "
                      f"({time.time() - t0:.0f}s)", flush=True)

            ratings  = parse_int(obj.get("ratings_count"))
            text_rev = parse_int(obj.get("text_reviews_count"))
            pub_year = parse_int(obj.get("publication_year"))
            lang     = norm_lang(obj.get("language_code"))
            title    = (obj.get("title_without_series") or obj.get("title") or "").strip()

            w = works.get(work_id)
            if w is None:
                w = [ -1, "", "", "", 0, 0, 0, 0, [0] * len(BUCKET_NAMES) ]
                works[work_id] = w

            # Work-level sums (2017 snapshot; existence filter only).
            w[5] += ratings
            w[6] += text_rev
            w[7] += 1

            # Earliest known publication year across editions ~ original year.
            if MIN_PUB_YEAR <= pub_year <= MAX_PUB_YEAR:
                w[4] = pub_year if w[4] == 0 else min(w[4], pub_year)

            # Representative edition = the most-rated one.
            if ratings > w[0]:
                w[0] = ratings
                w[1] = book_id
                w[2] = title
                w[3] = lang
                w[8] = shelf_bucket_counts(obj.get("popular_shelves", []))

    print(f"  {total:,} editions in {len(works):,} works — done "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ── Filter + write ────────────────────────────────────────────────────────
    kept = dropped_ratings = dropped_year = dropped_lang = 0

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["work_id", "rep_book_id", "title", "language", "pub_year",
             "n_editions", "ratings_count_2017", "text_reviews_2017"]
            + [f"shelf_{b}" for b in BUCKET_NAMES]
        )
        for work_id, w in works.items():
            (_, rep_bid, title, lang, pub_year,
             sum_ratings, sum_text, n_eds, buckets) = (
                w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8])

            if sum_ratings < POOL_MIN_RATINGS:
                dropped_ratings += 1
                continue
            if not (MIN_PUB_YEAR <= pub_year <= MAX_PUB_YEAR):
                dropped_year += 1
                continue
            if lang in EXCLUDED_LANGS:
                dropped_lang += 1
                continue

            genre_total = sum(buckets)
            shares = ([round(c / genre_total, 4) for c in buckets]
                      if genre_total else [0.0] * len(BUCKET_NAMES))

            writer.writerow(
                [work_id, rep_bid, title, lang, pub_year,
                 n_eds, sum_ratings, sum_text] + shares
            )
            kept += 1

    print("\n=== Summary ===")
    print(f"Works total                  : {len(works):>10,}")
    print(f"  dropped: <{POOL_MIN_RATINGS} ratings       : {dropped_ratings:>10,}")
    print(f"  dropped: pub_year outside  : {dropped_year:>10,}")
    print(f"  dropped: cze/slo language  : {dropped_lang:>10,}")
    print(f"Kept in candidate pool       : {kept:>10,}")
    print(f"→ {OUT_CSV}")


if __name__ == "__main__":
    main()
