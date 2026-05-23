"""
Cascade matcher: NKC translations → Goodreads books, with SCKN labels.

Pipeline
--------
1. Load 6 Goodreads lookup files + book_details (built by build_goodreads_lookup.py).
2. Build inverted map ``book_id → [norm_author, ...]`` for Layer 3 author scoring.
3. Load NKC translations CSV and SCKN bestseller charts.
4. For every NKC record, run the cascade:

    Layer 1  author exact + title exact (NFKD-normalized)
    Layer 2  author exact + title fuzzy   (rapidfuzz token_set_ratio ≥ 88)
    Layer 3  author fuzzy + title exact   (rapidfuzz token_set_ratio ≥ 85)
    else     unmatched

   Within a layer, tie-break candidates by:
     (a) group by work_id, pick best edition per work
         (language_code matches source_lang → pub_year closest to czech_pub_year → highest ratings_count)
     (b) across work representatives, pick best
         (language_code match → ratings_count → pub_year proximity)

5. Stream goodreads_books.json.gz a second time to collect the heavy metadata
   fields (average_rating, text_reviews_count, popular_shelves, is_ebook, raw
   title, language_code) only for the books that won the cascade. The full
   dump has 2.4M books; we only need ~50-100K of them.

6. Join NKC + match info + GR metadata + SCKN labels into:
     data/interim/matched_dataset.csv      ← all 225K NKC records, with match_layer
     data/interim/training_dataset.csv     ← filtered (≥2003, matched, ≥10 ratings) + dedup on work_id

Schema of training_dataset.csv preserves the columns notebook 05 reads:
    nkc_id, oclc, czech_isbn, czech_title, original_title, author, czech_pub_year,
    source_lang, genres, match_layer, matched_book_id, gr_title, gr_pub_year,
    gr_ratings_count, gr_average_rating, gr_text_reviews_count,
    gr_popular_shelves, gr_language_code, gr_is_ebook,
    sckn_appearances, sckn_bestseller

Plus two new v2 columns: gr_work_id, fuzzy_score.

Important contract notes
------------------------
- ``gr_ratings_count`` and ``gr_average_rating`` are kept in training_dataset.csv
  as metadata; the X_train.csv built by notebook 05 explicitly excludes them
  to avoid temporal leakage (those numbers are post-publication).
- SCKN labels are joined on czech_isbn (NKC may list several pipe-separated
  ISBNs; we OR across all of them).
- ``sckn_bestseller`` is the string "true"/"false" — notebook 05 reads it as
  text and does ``.str.lower() == "true"``. Don't switch to int here.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz, process

REPO    = Path(__file__).resolve().parents[2]
INTERIM = REPO / "data" / "interim"
RAW     = REPO / "data" / "raw"

# Inputs
NKC_CSV         = INTERIM / "nkc_translations.csv"
SCKN_CSV        = RAW / "sckn_charts.csv"
BOOKS_GZ        = RAW / "goodreads_books.json.gz"

LU_BY_ISBN      = INTERIM / "goodreads_by_isbn13.json"
LU_BY_AT        = INTERIM / "goodreads_by_author_title.json"
LU_BY_TITLE     = INTERIM / "goodreads_by_title.json"
LU_BY_AUTHOR    = INTERIM / "goodreads_by_author.json"
LU_WORK_BOOKS   = INTERIM / "goodreads_work_to_books.json"
LU_DETAILS      = INTERIM / "goodreads_book_details.json"

# Outputs
OUT_MATCHED  = INTERIM / "matched_dataset.csv"
OUT_TRAINING = INTERIM / "training_dataset.csv"

# Cascade thresholds (pinned in MATCHING_REBUILD_PLAN.md after spot-checks).
TITLE_FUZZY_THRESHOLD  = 88
AUTHOR_FUZZY_THRESHOLD = 85

# Training-set gates.
MIN_PUB_YEAR             = 2003       # SCKN charts start in 2003
MIN_GR_RATINGS_FOR_TRAIN = 10         # filter, not feature — book must exist on GR

# ── Text utils (NFKD + fuzzy helpers) ────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import (              # noqa: E402
    normalize,
    norm_nkc_author,
    norm_isbn,
    fuzzy_author_match,
)

# ── NKC source_lang (3-letter MARC) → Goodreads language_code (2-letter ISO) ──
#
# NKC uses MARC bibliographic codes (eng, ger, fre, rus …). Goodreads stores
# 2-letter ISO 639-1 codes ("en", "en-US", "de", "fr", "ru", …). The mapping
# below covers the languages we observed in NKC's source_lang_counts (>99% of
# the cohort lives in the first ~25 entries).
#
# Books published in a regional dialect ("en-GB", "en-US") still match "eng"
# because we test for prefix below.
SRC_LANG_TO_GR_PREFIX = {
    "eng": "en", "ger": "de", "fre": "fr", "spa": "es", "rus": "ru",
    "ita": "it", "pol": "pl", "por": "pt", "dut": "nl", "nor": "no",
    "nob": "no", "nno": "no", "swe": "sv", "dan": "da", "fin": "fi",
    "jpn": "ja", "chi": "zh", "kor": "ko", "hun": "hu", "tur": "tr",
    "ara": "ar", "heb": "he", "gre": "el", "ukr": "uk", "slo": "sk",
    "cze": "cs", "rum": "ro", "bul": "bg", "vie": "vi", "tha": "th",
    "ind": "id", "may": "ms", "cat": "ca", "ice": "is", "lat": "la",
    "scc": "sr", "scr": "hr",
}


def gr_lang_target(source_lang_field: str) -> str:
    """Return the 2-letter ISO prefix we expect on Goodreads, or ''."""
    if not source_lang_field:
        return ""
    # NKC can list several pipe-separated values ("eng|ger" for compilations).
    # Use the first one; multi-source books are rare and second value rarely
    # changes the right answer.
    first = source_lang_field.split("|", 1)[0].strip().lower()
    return SRC_LANG_TO_GR_PREFIX.get(first, "")


def lang_matches(gr_lang: str, target_prefix: str) -> bool:
    """True if ``gr_lang`` starts with ``target_prefix`` (e.g. "en-US" ~ "en")."""
    if not target_prefix or not gr_lang:
        return False
    return gr_lang.lower().startswith(target_prefix)


# ── Phase 1: load lookups ────────────────────────────────────────────────────

def t() -> float:
    return time.monotonic()


t0 = t()
print("Loading Goodreads lookups …", flush=True)


def load_json(path: Path):
    print(f"  {path.name}", flush=True)
    return json.loads(path.read_text(encoding="utf-8"))


by_isbn13     = load_json(LU_BY_ISBN)
by_at         = load_json(LU_BY_AT)
by_title      = load_json(LU_BY_TITLE)
by_author     = load_json(LU_BY_AUTHOR)
work_to_books = load_json(LU_WORK_BOOKS)
book_details  = load_json(LU_DETAILS)

print(f"  ({t()-t0:.1f}s) "
      f"isbn={len(by_isbn13):,}  at={len(by_at):,}  title={len(by_title):,}  "
      f"author={len(by_author):,}  works={len(work_to_books):,}  details={len(book_details):,}",
      flush=True)


# Inverted index: book_id → list[norm_author]. Needed for Layer 3 (we have
# candidate book_ids from by_title and need their authors to fuzzy-score
# against the NKC author). by_author is ``norm_author → [book_ids]`` so we
# invert it. Memory cost ~ 800K * 2 authors * 30 chars = 250 MB, acceptable.
print("Building book → authors inverted index …", flush=True)
ts = t()
book_authors_inv: dict[str, list[str]] = defaultdict(list)
for norm_a, bids in by_author.items():
    for bid in bids:
        book_authors_inv[bid].append(norm_a)
print(f"  {len(book_authors_inv):,} books indexed  ({t()-ts:.1f}s)", flush=True)


# ── Phase 2: load SCKN labels ────────────────────────────────────────────────

print("\nLoading SCKN charts …", flush=True)
ts = t()

# isbn13 → dict of aggregates. One SCKN row = one weekly top-10 chart entry.
sckn_by_isbn: dict[str, dict] = {}

with open(SCKN_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    sckn_total_rows = 0
    sckn_bad_isbn = 0
    for row in reader:
        sckn_total_rows += 1
        raw_isbn = row.get("isbn", "")
        if not raw_isbn:
            sckn_bad_isbn += 1
            continue
        isbn13 = norm_isbn(raw_isbn)
        if not isbn13 or len(isbn13) != 13:
            sckn_bad_isbn += 1
            continue

        try:
            year = int(row["year"])
            rank = int(row["rank"])
        except (ValueError, TypeError, KeyError):
            continue

        cat = row.get("category", "")
        s = sckn_by_isbn.setdefault(isbn13, {
            "appearances":   0,
            "first_year":    9999,
            "best_rank":     999,
            "categories":    set(),
        })
        s["appearances"] += 1
        if year < s["first_year"]:
            s["first_year"] = year
        if rank < s["best_rank"]:
            s["best_rank"] = rank
        s["categories"].add(cat)

print(f"  {sckn_total_rows:,} SCKN rows  ({sckn_bad_isbn:,} bad isbn)  "
      f"→ {len(sckn_by_isbn):,} unique ISBNs  ({t()-ts:.1f}s)", flush=True)


def sckn_lookup(czech_isbn_field: str) -> tuple[int, int | None, int | None, str]:
    """
    Look up SCKN stats for an NKC record. czech_isbn can hold several
    pipe-separated ISBNs (different bindings of the same Czech edition). Sum
    appearances across all of them, take min(year), min(rank), union(categories).
    Returns (appearances, first_year_or_None, best_rank_or_None, categories_str).
    """
    if not czech_isbn_field:
        return 0, None, None, ""

    appearances = 0
    first_year: int | None = None
    best_rank:  int | None = None
    cats: set[str] = set()

    for raw in czech_isbn_field.split("|"):
        raw = raw.strip()
        if not raw:
            continue
        isbn13 = norm_isbn(raw)
        if not isbn13 or len(isbn13) != 13:
            continue
        s = sckn_by_isbn.get(isbn13)
        if not s:
            continue
        appearances += s["appearances"]
        if first_year is None or s["first_year"] < first_year:
            first_year = s["first_year"]
        if best_rank is None or s["best_rank"] < best_rank:
            best_rank = s["best_rank"]
        cats |= s["categories"]

    return appearances, first_year, best_rank, "|".join(sorted(cats))


# ── Phase 3: cascade ─────────────────────────────────────────────────────────

def tiebreak(candidates: list[str], czech_year: int | None,
             target_lang: str) -> str:
    """
    Pick a single book_id out of multiple cascade matches.

    Step A: group by work_id. For each work, pick the best edition:
            language match → pub_year proximity → ratings_count.
    Step B: among work representatives, pick the best:
            language match → ratings_count → pub_year proximity.
    """
    # Group by work_id. Books without work_id are treated as their own work
    # (they keep their book_id as the work key).
    groups: dict[str, list[str]] = defaultdict(list)
    for bid in candidates:
        d = book_details.get(bid)
        if not d:
            continue
        _, _, _, _, work_id = d
        groups[work_id or f"_solo:{bid}"].append(bid)

    if not groups:
        # All candidates lacked details — fall back to first candidate.
        return candidates[0]

    def edition_key(bid: str):
        _, ratings, pub_year, lang, _ = book_details[bid]
        lm = lang_matches(lang, target_lang)
        if czech_year and pub_year:
            pub_dist = abs(pub_year - czech_year)
        else:
            pub_dist = 10_000
        # Sort ascending: best first. Negate "good" attributes so lower=better.
        return (-int(lm), pub_dist, -ratings)

    # Pick best edition per work.
    reps = [sorted(bids, key=edition_key)[0] for bids in groups.values()]

    def work_key(bid: str):
        _, ratings, pub_year, lang, _ = book_details[bid]
        lm = lang_matches(lang, target_lang)
        if czech_year and pub_year:
            pub_dist = abs(pub_year - czech_year)
        else:
            pub_dist = 10_000
        # Across works: ratings dominate after language match.
        return (-int(lm), -ratings, pub_dist)

    return sorted(reps, key=work_key)[0]


def cascade(norm_a: str, norm_t: str, czech_year: int | None,
            target_lang: str) -> tuple[str | None, str, int]:
    """
    Run the 3-layer cascade. Returns (book_id_or_None, layer_label, fuzzy_score).

    layer_label ∈ {"1", "2", "3", "unmatched"}
    fuzzy_score = 100 for Layer 1 (exact); ≥88 for Layer 2; ≥85 for Layer 3; 0 otherwise.
    """
    # ── Layer 1: author exact + title exact ──────────────────────────────────
    if norm_a and norm_t:
        key = f"{norm_a}||{norm_t}"
        cands = by_at.get(key)
        if cands:
            return tiebreak(cands, czech_year, target_lang), "1", 100

    # ── Layer 2: author exact + title fuzzy ──────────────────────────────────
    if norm_a and norm_t:
        author_pool = by_author.get(norm_a)
        if author_pool:
            # Build parallel list of candidate titles (skip books without details).
            titles_and_ids = [
                (book_details[bid][0], bid)
                for bid in author_pool
                if bid in book_details and book_details[bid][0]
            ]
            if titles_and_ids:
                titles = [t for t, _ in titles_and_ids]
                # rapidfuzz batch scoring with cutoff (C-implemented, fast).
                hits = process.extract(
                    norm_t, titles,
                    scorer=fuzz.token_set_ratio,
                    score_cutoff=TITLE_FUZZY_THRESHOLD,
                    limit=None,
                )
                if hits:
                    # Hits: (matched_title, score, index_in_titles). Map back to bids.
                    best_score = max(h[1] for h in hits)
                    cands = [titles_and_ids[h[2]][1] for h in hits]
                    return tiebreak(cands, czech_year, target_lang), "2", int(best_score)

    # ── Layer 3: author fuzzy + title exact ──────────────────────────────────
    if norm_a and norm_t:
        title_pool = by_title.get(norm_t)
        if title_pool:
            cands_with_score: list[tuple[str, int]] = []
            for bid in title_pool:
                authors = book_authors_inv.get(bid, [])
                if not authors:
                    continue
                best_score = 0
                for a in authors:
                    s = fuzzy_author_match(norm_a, a)
                    if s > best_score:
                        best_score = s
                if best_score >= AUTHOR_FUZZY_THRESHOLD:
                    cands_with_score.append((bid, best_score))
            if cands_with_score:
                top_score = max(s for _, s in cands_with_score)
                cands = [bid for bid, _ in cands_with_score]
                return tiebreak(cands, czech_year, target_lang), "3", int(top_score)

    return None, "unmatched", 0


# ── Phase 4: run cascade over NKC ────────────────────────────────────────────

print("\nRunning cascade over NKC translations …", flush=True)
ts = t()

# Header from NKC CSV.
NKC_CARRY_COLS = [
    "nkc_id", "oclc", "czech_isbn", "czech_title", "original_title",
    "original_isbn", "original_sysnum",
    "author", "secondary_authors", "czech_pub_year", "source_lang", "genres",
]

# Per-record outputs we collect during cascade. GR metadata (ratings, shelves,
# etc.) is filled in Phase 5.
match_records: list[dict] = []

# Stats
layer_counts = {"1": 0, "2": 0, "3": 0, "unmatched": 0}

with open(NKC_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, start=1):
        nkc_author = row.get("author", "")
        nkc_title  = row.get("original_title", "")

        norm_a = norm_nkc_author(nkc_author) if nkc_author else ""
        norm_t = normalize(nkc_title) if nkc_title else ""

        try:
            cz_year: int | None = int(row.get("czech_pub_year") or 0) or None
        except ValueError:
            cz_year = None

        target_lang = gr_lang_target(row.get("source_lang", ""))

        book_id, layer, fscore = cascade(norm_a, norm_t, cz_year, target_lang)
        layer_counts[layer] += 1

        # Resolve work_id for output, if matched.
        gr_work_id = ""
        if book_id and book_id in book_details:
            gr_work_id = book_details[book_id][4] or ""

        rec = {k: row.get(k, "") for k in NKC_CARRY_COLS}
        rec.update({
            "match_layer":     layer,
            "matched_book_id": book_id or "",
            "gr_work_id":      gr_work_id,
            "fuzzy_score":     fscore,
        })
        match_records.append(rec)

        if i % 20_000 == 0:
            print(f"  {i:>7,} NKC records  ({t()-ts:.0f}s)  "
                  f"L1={layer_counts['1']:,}  L2={layer_counts['2']:,}  "
                  f"L3={layer_counts['3']:,}  unmatched={layer_counts['unmatched']:,}",
                  flush=True)

print(f"  done — {len(match_records):,} NKC records in {t()-ts:.0f}s", flush=True)
print(f"  Layer counts: {layer_counts}", flush=True)


# ── Phase 5: enrich matched books with full GR metadata ─────────────────────

# Collect set of book_ids that won the cascade (small subset of 2.4M).
needed_book_ids: set[str] = {
    r["matched_book_id"] for r in match_records if r["matched_book_id"]
}
print(f"\nNeed full GR metadata for {len(needed_book_ids):,} books "
      f"({len(needed_book_ids)/len(book_details):.1%} of dump).",
      flush=True)

# Stream goodreads_books.json.gz once; extract heavy fields only for needed
# book_ids. ~30 min single pass.
heavy: dict[str, dict] = {}

print("Streaming goodreads_books.json.gz to fetch metadata …", flush=True)
ts = t()
seen = found = 0
with gzip.open(BOOKS_GZ, "rt", encoding="utf-8") as fh:
    for line in fh:
        seen += 1
        if seen % 250_000 == 0:
            print(f"  {seen:,} lines  ({found:,}/{len(needed_book_ids):,} found)  "
                  f"({t()-ts:.0f}s)", flush=True)
        # Cheap pre-check: is this book in our target set? Parse only if so.
        # book_id appears early in the JSON, but we have to parse fully for
        # other fields. We can short-circuit using a regex match on book_id
        # for ~3x speedup, but the cleanest correctness path is full parse.
        obj = json.loads(line)
        bid = obj.get("book_id", "")
        if bid not in needed_book_ids:
            continue

        # Goodreads sometimes stores publication_year as a string or empty.
        try:
            pub_year = int(obj.get("publication_year") or 0)
        except (ValueError, TypeError):
            pub_year = 0

        heavy[bid] = {
            "gr_title":              obj.get("title_without_series", "") or obj.get("title", ""),
            "gr_pub_year":           pub_year if pub_year else "",
            "gr_ratings_count":      obj.get("ratings_count", "") or "",
            "gr_average_rating":     obj.get("average_rating", "") or "",
            "gr_text_reviews_count": obj.get("text_reviews_count", "") or "",
            # popular_shelves is a list of {name, count} dicts. Serialize back
            # to JSON string so it round-trips cleanly through CSV (notebook 05
            # does json.loads on this column).
            "gr_popular_shelves":    json.dumps(
                obj.get("popular_shelves", []), ensure_ascii=False
            ),
            "gr_language_code":      obj.get("language_code", "") or "",
            "gr_is_ebook":           "true" if obj.get("is_ebook") in (True, "true") else "false",
        }
        found += 1
        if found >= len(needed_book_ids):
            print(f"  all {found:,} found — stopping early at line {seen:,}", flush=True)
            break

print(f"  metadata collected for {len(heavy):,}/{len(needed_book_ids):,} books "
      f"({t()-ts:.0f}s)", flush=True)

if len(heavy) < len(needed_book_ids):
    print(f"  WARNING: {len(needed_book_ids)-len(heavy)} matched book_ids "
          f"were NOT found in the books dump. They will have empty gr_* fields.",
          flush=True)


# ── Phase 6: attach GR metadata + SCKN labels, write outputs ────────────────

print("\nAttaching GR metadata + SCKN labels …", flush=True)

GR_COLS = ["gr_title", "gr_pub_year", "gr_ratings_count", "gr_average_rating",
           "gr_text_reviews_count", "gr_popular_shelves", "gr_language_code",
           "gr_is_ebook"]
SCKN_COLS = ["sckn_appearances", "sckn_first_year", "sckn_best_rank",
             "sckn_categories", "sckn_bestseller"]

for rec in match_records:
    # GR fields
    bid = rec["matched_book_id"]
    if bid and bid in heavy:
        rec.update(heavy[bid])
    else:
        for c in GR_COLS:
            rec[c] = ""

    # SCKN fields
    apps, fy, br, cats = sckn_lookup(rec["czech_isbn"])
    rec["sckn_appearances"] = apps
    rec["sckn_first_year"]  = fy if fy is not None else ""
    rec["sckn_best_rank"]   = br if br is not None else ""
    rec["sckn_categories"]  = cats
    # String, not int — notebook 05 reads it as text. See module docstring.
    rec["sckn_bestseller"]  = "true" if apps >= 1 else "false"


# Output column order — first the original NKC carry-over, then match info,
# then GR metadata, then SCKN labels.
OUT_COLS = (
    NKC_CARRY_COLS
    + ["match_layer", "matched_book_id", "gr_work_id", "fuzzy_score"]
    + GR_COLS
    + SCKN_COLS
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


print(f"  writing {OUT_MATCHED.name} ({len(match_records):,} rows) …", flush=True)
write_csv(OUT_MATCHED, match_records)


# ── Phase 7: build training_dataset.csv (filter + dedup) ────────────────────

# Filter gates:
#   - match_layer != "unmatched"
#   - czech_pub_year >= 2003
#   - gr_ratings_count >= 10 (filter, not feature; rules out empty GR entries)
print("\nFiltering for training set …", flush=True)


def parse_int_safe(v) -> int:
    try:
        return int(float(v)) if v not in ("", None) else 0
    except (ValueError, TypeError):
        return 0


training_pool: list[dict] = []
for r in match_records:
    if r["match_layer"] == "unmatched":
        continue
    year = parse_int_safe(r["czech_pub_year"])
    if year < MIN_PUB_YEAR:
        continue
    if parse_int_safe(r["gr_ratings_count"]) < MIN_GR_RATINGS_FOR_TRAIN:
        continue
    training_pool.append(r)

print(f"  After filter: {len(training_pool):,} candidates", flush=True)

# Dedup by gr_work_id: multiple Czech editions of the same foreign work become
# one training example. Tie-break:
#   1. Max sckn_appearances (keep the positive if any)
#   2. Earliest czech_pub_year (older edition wins; closer to the original publication)
#
# Records without a gr_work_id (Goodreads dump quirk) keep their own bucket
# keyed by matched_book_id so we don't collapse unrelated books.
buckets: dict[str, list[dict]] = defaultdict(list)
for r in training_pool:
    key = r["gr_work_id"] or f"_solo:{r['matched_book_id']}"
    buckets[key].append(r)


def dedup_pick(group: list[dict]) -> dict:
    def sort_key(r):
        apps = parse_int_safe(r["sckn_appearances"])
        year = parse_int_safe(r["czech_pub_year"]) or 9999
        return (-apps, year)
    return sorted(group, key=sort_key)[0]


training: list[dict] = [dedup_pick(g) for g in buckets.values()]
print(f"  After dedup on gr_work_id: {len(training):,} unique works", flush=True)

print(f"  writing {OUT_TRAINING.name} ({len(training):,} rows) …", flush=True)
write_csv(OUT_TRAINING, training)


# ── Summary ─────────────────────────────────────────────────────────────────

n = len(match_records)
n_pos_full = sum(1 for r in match_records if parse_int_safe(r["sckn_appearances"]) >= 1)

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"NKC records processed         : {n:>10,}")
print(f"  Layer 1 (exact + exact)     : {layer_counts['1']:>10,}  ({layer_counts['1']/n:.1%})")
print(f"  Layer 2 (exact + fuzzy ≥{TITLE_FUZZY_THRESHOLD})    : "
      f"{layer_counts['2']:>10,}  ({layer_counts['2']/n:.1%})")
print(f"  Layer 3 (fuzzy ≥{AUTHOR_FUZZY_THRESHOLD} + exact)   : "
      f"{layer_counts['3']:>10,}  ({layer_counts['3']/n:.1%})")
print(f"  Unmatched                   : {layer_counts['unmatched']:>10,}  "
      f"({layer_counts['unmatched']/n:.1%})")
print(f"Matched total                  : {n - layer_counts['unmatched']:>10,}  "
      f"({(n-layer_counts['unmatched'])/n:.1%})")
print(f"SCKN positives in full set     : {n_pos_full:>10,}  ({n_pos_full/n:.1%})")
print()

n_train = len(training)
n_pos_train = sum(1 for r in training if parse_int_safe(r["sckn_appearances"]) >= 1)
print(f"Training set (≥{MIN_PUB_YEAR}, matched, ≥{MIN_GR_RATINGS_FOR_TRAIN} GR ratings, dedup):")
print(f"  Rows                        : {n_train:>10,}")
print(f"  Positives (sckn ≥ 1)        : {n_pos_train:>10,}  ({n_pos_train/n_train:.1%})")
print()
print(f"Outputs:")
print(f"  {OUT_MATCHED}")
print(f"  {OUT_TRAINING}")
print()
print(f"Total time: {t()-t0:.0f}s")
