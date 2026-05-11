"""
Full NKC → Goodreads → SCKN pipeline.

Reads all 225K NKC translation records, matches each to a Goodreads book via
a two-layer cascade, joins with SCKN chart data to assign bestseller labels,
and writes two output files:

  matched_dataset.csv  — all 225K NKC records, no filters (used for inference)
  training_dataset.csv — filtered subset ready for model training (see filters below)
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import normalize, norm_nkc_author, norm_isbn

INTERIM  = REPO / "data" / "interim"
RAW      = REPO / "data" / "raw"
BOOKS_GZ = RAW / "goodreads_books.json.gz"

# ── Load inputs ───────────────────────────────────────────────────────────────

print("Loading NKC translations …", flush=True)
nkc = pd.read_csv(INTERIM / "nkc_translations.csv", dtype=str).fillna("")
nkc["_year"] = pd.to_numeric(nkc["czech_pub_year"], errors="coerce")
print(f"  {len(nkc):,} records")

print("Loading Goodreads lookup files …", flush=True)
by_author_title = json.loads((INTERIM / "goodreads_by_author_title.json").read_text())
by_title        = json.loads((INTERIM / "goodreads_by_title.json").read_text())
print(f"  by_author_title : {len(by_author_title):>8,} keys")
print(f"  by_title        : {len(by_title):>8,} keys")

print("Loading SCKN charts …", flush=True)
sckn = pd.read_csv(RAW / "sckn_charts.csv", dtype=str).fillna("")
print(f"  {len(sckn):,} chart entries")

# ── Build SCKN ISBN → appearances count ──────────────────────────────────────

print("\nBuilding SCKN ISBN lookup …", flush=True)
sckn_counts: dict[str, int] = defaultdict(int)
for raw_isbn in sckn["isbn"]:
    isbn = norm_isbn(raw_isbn.strip())
    if isbn:
        sckn_counts[isbn] += 1
print(f"  {len(sckn_counts):,} unique normalised SCKN ISBNs")


def sckn_appearances_for(czech_isbn: str) -> int:
    """Return SCKN chart appearances for an NKC record (0 if none found)."""
    if not czech_isbn.strip():
        return 0
    return max(
        (sckn_counts.get(norm_isbn(part.strip()), 0) for part in czech_isbn.split("|")),
        default=0,
    )


# ── Matching cascade ──────────────────────────────────────────────────────────

print("\nRunning matching cascade on all NKC records …", flush=True)
pending = []

for i, row in nkc.iterrows():
    if (i + 1) % 50_000 == 0:
        print(f"  {i+1:,} / {len(nkc):,} records processed …", flush=True)

    author     = row["author"].strip()
    orig_title = row["original_title"].strip()
    czech_year = int(row["_year"]) if pd.notna(row["_year"]) else None

    candidates  = []
    match_layer = 3

    # Layer 1 — author + title
    if author and orig_title:
        key  = norm_nkc_author(author) + "||" + normalize(orig_title)
        hits = by_author_title.get(key, [])
        if hits:
            candidates, match_layer = hits, 1

    # Layer 2 — title only
    if match_layer == 3 and orig_title:
        hits = by_title.get(normalize(orig_title), [])
        if hits:
            candidates, match_layer = hits, 2

    pending.append({
        "nkc_id":         row["nkc_id"],
        "oclc":           row["oclc"],
        "czech_isbn":     row["czech_isbn"],
        "czech_title":    row["czech_title"],
        "original_title": orig_title,
        "author":         author,
        "czech_pub_year": czech_year,
        "source_lang":    row["source_lang"],
        "genres":         row["genres"],
        "candidates":     candidates,
        "match_layer":    match_layer,
    })

print(f"  {len(nkc):,} / {len(nkc):,} records processed — done", flush=True)

# ── Fetch Goodreads metadata in a single pass ─────────────────────────────────

all_candidate_ids: set[str] = set()
for r in pending:
    all_candidate_ids.update(r["candidates"])

print(f"\nFetching metadata for {len(all_candidate_ids):,} candidate book_ids …", flush=True)
book_meta: dict[str, dict] = {}

with gzip.open(BOOKS_GZ, "rt", encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if len(book_meta) == len(all_candidate_ids):
            break
        obj = json.loads(line)
        bid = obj.get("book_id", "")
        if bid not in all_candidate_ids:
            continue
        try:
            pub_year = int(obj.get("publication_year") or 0)
        except (ValueError, TypeError):
            pub_year = 0
        book_meta[bid] = {
            "title":            obj.get("title_without_series") or obj.get("title", ""),
            "pub_year":         pub_year,
            "author_ids":       [a["author_id"] for a in obj.get("authors", [])],
            "ratings_count":    int(obj.get("ratings_count") or 0),
            "average_rating":   float(obj.get("average_rating") or 0.0),
            "text_reviews_count": int(obj.get("text_reviews_count") or 0),
            "popular_shelves":  json.dumps(obj.get("popular_shelves", [])),
            "language_code":    obj.get("language_code", ""),
            "is_ebook":         obj.get("is_ebook", ""),
        }
        if (i + 1) % 500_000 == 0:
            print(
                f"  {i+1:,} lines scanned, "
                f"{len(book_meta):,}/{len(all_candidate_ids):,} found …",
                flush=True,
            )

print(f"  Done. Retrieved {len(book_meta):,} / {len(all_candidate_ids):,} records.")

# ── Tie-breaking and result assembly ─────────────────────────────────────────

def pick_best(candidates: list[str], target_year: int) -> str:
    if len(candidates) == 1:
        return candidates[0]
    return min(
        candidates,
        key=lambda bid: abs((book_meta.get(bid, {}).get("pub_year") or 0) - target_year)
        if (book_meta.get(bid, {}).get("pub_year") or 0) > 0 else 9999,
    )


print("\nAssembling final dataset …", flush=True)
rows = []
for r in pending:
    target_year = (r["czech_pub_year"] or 2010) - 3

    if r["match_layer"] == 3:
        meta   = {}
        best_id = ""
    else:
        best_id = pick_best(r["candidates"], target_year)
        meta    = book_meta.get(best_id, {})

    sckn_app = sckn_appearances_for(r["czech_isbn"])

    rows.append({
        # NKC fields
        "nkc_id":           r["nkc_id"],
        "oclc":             r["oclc"],
        "czech_isbn":       r["czech_isbn"],
        "czech_title":      r["czech_title"],
        "original_title":   r["original_title"],
        "author":           r["author"],
        "czech_pub_year":   r["czech_pub_year"],
        "source_lang":      r["source_lang"],
        "genres":           r["genres"],
        # Matching
        "match_layer":      r["match_layer"],
        "matched_book_id":  best_id,
        # Goodreads
        "gr_title":               meta.get("title", ""),
        "gr_pub_year":            meta.get("pub_year", "") or "",
        "gr_ratings_count":       meta.get("ratings_count", ""),
        "gr_average_rating":      meta.get("average_rating", ""),
        "gr_text_reviews_count":  meta.get("text_reviews_count", ""),
        "gr_popular_shelves":     meta.get("popular_shelves", ""),
        "gr_language_code":       meta.get("language_code", ""),
        "gr_is_ebook":            meta.get("is_ebook", ""),
        # SCKN labels.
        # Threshold 1: one appearance already means the book outsold hundreds of
        # competitors that week, so it is a meaningful positive signal.
        "sckn_appearances": sckn_app,
        "sckn_bestseller":  sckn_app >= 1,
    })

df = pd.DataFrame(rows)

# ── Save: full dataset (all 225K, no filters — used for inference) ────────────

full_path = INTERIM / "matched_dataset.csv"
df.to_csv(full_path, index=False)
print(f"\nSaved {len(df):,} rows → {full_path}")

# ── Build training dataset (filtered) ────────────────────────────────────────
#
# Each filter is a hard requirement for reliable supervised learning:
#
#   czech_pub_year >= 2003  — SCKN data starts in 2003; earlier records can
#                             never carry a positive label, so including them
#                             would silently inflate the negative class.
#
#   gr_ratings_count >= 10  — fewer than 10 Goodreads ratings yields no
#                             reliable signal (a mean from 2 people is noise,
#                             not a feature).
#
#   match_layer != 3        — unmatched records have no Goodreads features, so
#                             they cannot be used as training examples regardless
#                             of their label.
#
# Source language is NOT filtered: German, French, etc. translations that
# charted on SCKN are valid positive examples, and excluding them would
# artificially shrink the positive class.

train = df[
    (df["czech_pub_year"] >= 2003) &                                        # SCKN coverage starts here
    (pd.to_numeric(df["gr_ratings_count"], errors="coerce") >= 10) &        # minimum Goodreads signal
    (df["match_layer"] != 3)                                                 # must have Goodreads features
].copy()

train_path = INTERIM / "training_dataset.csv"
train.to_csv(train_path, index=False)
print(f"Saved {len(train):,} rows → {train_path}")

# ── Summary ───────────────────────────────────────────────────────────────────

layer_labels = {1: "Layer 1 — author+title", 2: "Layer 2 — title only", 3: "Layer 3 — unmatched"}

print("\n=== Summary ===")
print(f"\n{'matched_dataset.csv':}")
n_full = len(df)
print(f"  Total records            : {n_full:>8,}")
print(f"  Match rate by layer:")
for layer in [1, 2, 3]:
    cnt = (df["match_layer"] == layer).sum()
    print(f"    {layer_labels[layer]:<28} : {cnt:>7,}  ({cnt/n_full:.1%})")

print(f"\n{'training_dataset.csv':}")
n_train = len(train)
n_pos = train["sckn_bestseller"].sum()
n_neg = n_train - n_pos
print(f"  Total rows               : {n_train:>8,}")
print(f"  Bestsellers (label=1)    : {n_pos:>8,}  ({n_pos/n_train:.1%})")
print(f"  Non-bestsellers (label=0): {n_neg:>8,}  ({n_neg/n_train:.1%})")
print(f"\n  Match layer breakdown:")
for layer in [1, 2]:
    cnt = (train["match_layer"] == layer).sum()
    print(f"    {layer_labels[layer]:<28} : {cnt:>7,}  ({cnt/n_train:.1%})")

print("\n  Cross-tab match_layer × sckn_bestseller:")
ct = pd.crosstab(
    train["match_layer"].map(layer_labels),
    train["sckn_bestseller"],
    margins=True,
)
ct.columns = ["not_bestseller", "bestseller", "total"]
ct.index.name = "match_layer"
print(ct.to_string())
