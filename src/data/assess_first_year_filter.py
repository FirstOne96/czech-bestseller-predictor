"""
Assessment: how many training rows would survive if we filter to books whose
FIRST Czech translation was published in the SCKN era (>= 2003)?

Motivation (supervisor feedback)
--------------------------------
SCKN charts start in 2003. Books whose first Czech edition came out long before
2003 (e.g. Gatsby — Czech first ed. ~1960) have multiple reprints in our NKC
dump. Those reprints inevitably get sckn_appearances=0 not because the book
flopped, but because everyone who wanted it already had it before SCKN began.
Labeling such reprints as "negative" is label noise, not signal.

What this script does
---------------------
1. Loads ALL nkc_translations.csv (every translation edition, not just matched).
2. Groups by (norm_author, norm_original_title) and computes min(czech_pub_year)
   — that's the "first Czech edition year" for the work.
3. Loads training_dataset.csv.
4. For each training row, looks up first_czech_year via its (author, title) key.
5. Reports impact of filter `first_czech_year >= 2003`.

Run:
    python src/data/assess_first_year_filter.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "data"))
from text_utils import normalize, norm_nkc_author  # noqa: E402

INTERIM = REPO / "data" / "interim"
RAW     = REPO / "data" / "raw"

SCKN_START_YEAR = 2003


# --------------------------------------------------------------------------- #
# 1. SCKN sanity check — confirm earliest year
# --------------------------------------------------------------------------- #

print("=" * 80)
print("SCKN earliest year check")
print("=" * 80)
years = set()
with open(RAW / "sckn_charts.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            years.add(int(row["year"]))
        except (ValueError, TypeError):
            continue
print(f"  Earliest SCKN year   : {min(years)}")
print(f"  Latest SCKN year     : {max(years)}")
print(f"  Distinct years       : {len(years)}")
assert SCKN_START_YEAR == min(years), f"hardcoded {SCKN_START_YEAR} != actual {min(years)}"


# --------------------------------------------------------------------------- #
# 2. Build first_czech_year lookup from ALL NKC records
# --------------------------------------------------------------------------- #

print()
print("=" * 80)
print("Building first_czech_year lookup from nkc_translations.csv")
print("=" * 80)

# Key: (norm_author, norm_original_title) → min(czech_pub_year)
first_year: dict[tuple[str, str], int] = {}
solo_count = 0
nkc_total = 0
with open(INTERIM / "nkc_translations.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        nkc_total += 1
        author = row.get("author", "").strip()
        orig   = row.get("original_title", "").strip()
        try:
            year = int(row.get("czech_pub_year", "").strip())
        except (ValueError, TypeError):
            continue
        if not author or not orig:
            solo_count += 1
            continue

        key = (norm_nkc_author(author), normalize(orig))
        if key not in first_year or year < first_year[key]:
            first_year[key] = year

print(f"  Total NKC records              : {nkc_total:,}")
print(f"  Records w/o author or orig     : {solo_count:,}")
print(f"  Distinct (author, title) works : {len(first_year):,}")


# --------------------------------------------------------------------------- #
# 3. Apply filter to training_dataset.csv
# --------------------------------------------------------------------------- #

print()
print("=" * 80)
print("Filter impact on training_dataset.csv")
print("=" * 80)

train_total = 0
no_key = 0
no_first_year = 0
survives = 0
dropped = 0
year_buckets: dict[str, int] = defaultdict(int)

pos_total = pos_survives = pos_dropped = 0
dropped_examples: list[tuple[str, str, int, int, int]] = []  # (author, title, czech_pub_year, first_year, sckn)
surviving_examples: list[tuple[str, str, int, int, int]] = []

with open(INTERIM / "training_dataset.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        train_total += 1
        author = row.get("author", "").strip()
        orig   = row.get("original_title", "").strip()
        try:
            czech_year = int(row.get("czech_pub_year", "").strip())
        except (ValueError, TypeError):
            czech_year = -1
        try:
            sckn = int(row.get("sckn_appearances", "0").strip())
        except (ValueError, TypeError):
            sckn = 0
        is_pos = sckn >= 1
        if is_pos:
            pos_total += 1

        if not author or not orig:
            no_key += 1
            continue

        key = (norm_nkc_author(author), normalize(orig))
        fy = first_year.get(key)
        if fy is None:
            no_first_year += 1
            continue

        # Bucket by first_year decade-ish
        if   fy < 1950: bucket = "<1950"
        elif fy < 1990: bucket = "1950-1989"
        elif fy < 2000: bucket = "1990-1999"
        elif fy < 2003: bucket = "2000-2002"
        elif fy < 2010: bucket = "2003-2009"
        elif fy < 2020: bucket = "2010-2019"
        else:           bucket = "2020+"
        year_buckets[bucket] += 1

        if fy >= SCKN_START_YEAR:
            survives += 1
            if is_pos: pos_survives += 1
            if len(surviving_examples) < 10 and is_pos:
                surviving_examples.append((author, orig, czech_year, fy, sckn))
        else:
            dropped += 1
            if is_pos: pos_dropped += 1
            if len(dropped_examples) < 15:
                dropped_examples.append((author, orig, czech_year, fy, sckn))

print(f"  Training rows total            : {train_total:,}")
print(f"  Missing (author or title)      : {no_key:,}")
print(f"  No first_czech_year lookup hit : {no_first_year:,}  (work not in NKC??)")
print(f"  → Survives (first_year >= 2003): {survives:,}  ({survives/train_total:.1%})")
print(f"  → Dropped (first_year <  2003) : {dropped:,}  ({dropped/train_total:.1%})")

print()
print("  first_czech_year distribution (training rows where we found the key):")
total_bucketed = sum(year_buckets.values())
order = ["<1950", "1950-1989", "1990-1999", "2000-2002",
         "2003-2009", "2010-2019", "2020+"]
for b in order:
    c = year_buckets.get(b, 0)
    pct = c / max(total_bucketed, 1)
    print(f"    {b:<12} : {c:>6,}  ({pct:.1%})")


# --------------------------------------------------------------------------- #
# 4. Positive rate before/after
# --------------------------------------------------------------------------- #

print()
print("=" * 80)
print("Positive rate before/after filter")
print("=" * 80)
pos_rate_before = pos_total / train_total
pos_rate_after  = pos_survives / max(survives, 1)
print(f"  BEFORE  : {pos_total:>5,} / {train_total:>6,}  = {pos_rate_before:.2%}")
print(f"  AFTER   : {pos_survives:>5,} / {survives:>6,}  = {pos_rate_after:.2%}")
print(f"  Positives LOST in filter       : {pos_dropped:,}")
print(f"  (Those are SCKN-positive books whose first Czech ed. was < 2003)")
print(f"  Ratio improvement              : {pos_rate_after / max(pos_rate_before, 0.0001):.2f}x")


# --------------------------------------------------------------------------- #
# 5. Sanity check examples
# --------------------------------------------------------------------------- #

print()
print("=" * 80)
print("Examples of DROPPED rows (first_czech_year < 2003)")
print("  — these should be reprints/classics labeled noise as 'negative'")
print("=" * 80)
for author, title, czy, fy, sckn in dropped_examples:
    label = "POS" if sckn >= 1 else "neg"
    print(f"  [{label}] {author!r} — {title!r}")
    print(f"        first_czech={fy}, this_row_czech_pub_year={czy}, sckn={sckn}")

print()
print("=" * 80)
print("Examples of SURVIVING positives (first_czech_year >= 2003)")
print("  — these are clean signal: first translated in SCKN era and hit charts")
print("=" * 80)
for author, title, czy, fy, sckn in surviving_examples:
    print(f"  [POS] {author!r} — {title!r}")
    print(f"        first_czech={fy}, this_row_czech_pub_year={czy}, sckn={sckn}")

print()
print("Assessment complete.")
