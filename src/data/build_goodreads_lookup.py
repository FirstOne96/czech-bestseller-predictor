"""
Stream goodreads_book_authors.json.gz and goodreads_books.json.gz once each
and emit seven lookup files to data/interim/.

v2 changes (2026-05-22)
-----------------------
* All text normalisation now uses the NFKD-aware ``normalize()`` from
  ``text_utils`` — Czech/German/Polish/Scandinavian diacritics fold to ASCII
  before key construction, raising exact-match recall.
* New file ``goodreads_by_author.json`` — index for cascade Layer 2
  (author exact + title fuzzy).
* New file ``goodreads_work_to_books.json`` — for canonical-edition tie-break
  and work-aware deduplication in the cascade.
* New file ``goodreads_book_details.json`` — per-book metadata (norm_title,
  ratings_count, pub_year, language_code, work_id).  All other index files now
  store only ``book_id`` references; tie-break metadata is fetched from this
  map at query time.  Removes the [book_id, ratings_count] inline duplication
  that v1 used.

Outputs
-------
data/interim/
  goodreads_author_names.json      author_id → name (string)
  goodreads_by_isbn13.json         isbn13 → book_id
  goodreads_by_author_title.json   "norm_author||norm_title" → list[book_id]
  goodreads_by_title.json          norm_title → list[book_id]
  goodreads_by_author.json         norm_author → list[book_id]
  goodreads_work_to_books.json     work_id → list[book_id]
  goodreads_book_details.json      book_id → [norm_title, ratings_count,
                                              pub_year, language_code, work_id]

"""
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW  = REPO / "data" / "raw"
OUT  = REPO / "data" / "interim"

AUTHORS_GZ = RAW / "goodreads_book_authors.json.gz"
BOOKS_GZ   = RAW / "goodreads_books.json.gz"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import normalize  # noqa: E402

# Goodreads sometimes stores title_without_series with the series suffix still
# attached, e.g. "Harry Potter and the Deathly Hallows (Harry Potter, #7)".
# Strip "(...#...)" at end-of-string before normalising so the key matches the
# plain title used in NKC's 240$a.
_SERIES_RE = re.compile(r"\s*\([^)]*#[^)]*\)\s*$")


def strip_series(title: str) -> str:
    return _SERIES_RE.sub("", title).strip()


def norm_isbn13(raw: str) -> str:
    """Return digits only if exactly 13; otherwise empty string."""
    digits = "".join(c for c in raw if c.isdigit())
    return digits if len(digits) == 13 else ""


def parse_int(v) -> int:
    """Tolerant int parse — empty / None / non-numeric → 0."""
    try:
        return int(v) if v not in (None, "") else 0
    except (ValueError, TypeError):
        return 0


# ── Pass 1: authors ──────────────────────────────────────────────────────────

print("Pass 1: streaming authors …", flush=True)
author_names: dict[str, str] = {}

with gzip.open(AUTHORS_GZ, "rt", encoding="utf-8") as fh:
    for line in fh:
        obj = json.loads(line)
        aid  = obj.get("author_id", "")
        name = obj.get("name", "").strip()
        if aid and name:
            author_names[aid] = name

print(f"  authors loaded: {len(author_names):,}", flush=True)

(OUT / "goodreads_author_names.json").write_text(
    json.dumps(author_names, ensure_ascii=False), encoding="utf-8"
)
print(f"  → wrote goodreads_author_names.json", flush=True)


# ── Pass 2: books ────────────────────────────────────────────────────────────

print("\nPass 2: streaming books …", flush=True)

by_isbn13:        dict[str, str]        = {}
by_author_title:  dict[str, list[str]]  = defaultdict(list)
by_title:         dict[str, list[str]]  = defaultdict(list)
by_author:        dict[str, list[str]]  = defaultdict(list)
work_to_books:    dict[str, list[str]]  = defaultdict(list)
book_details:     dict[str, list]       = {}

total = isbn13_hits = with_work = with_language = 0

with gzip.open(BOOKS_GZ, "rt", encoding="utf-8") as fh:
    for line in fh:
        obj = json.loads(line)
        book_id = obj.get("book_id", "")
        if not book_id:
            continue

        total += 1
        if total % 200_000 == 0:
            print(f"  {total:,} books processed …", flush=True)

        ratings_count = parse_int(obj.get("ratings_count"))
        pub_year      = parse_int(obj.get("publication_year"))
        language_code = (obj.get("language_code") or "").strip()
        work_id       = str(obj.get("work_id") or "").strip()

        # Title — strip series suffix, then NFKD-normalize.
        raw_title  = obj.get("title_without_series", "") or obj.get("title", "")
        norm_title = normalize(strip_series(raw_title))

        # Single source of truth for per-book metadata.
        # Order chosen for predictable downstream unpacking.
        book_details[book_id] = [
            norm_title, ratings_count, pub_year, language_code, work_id,
        ]

        if language_code:
            with_language += 1

        # ISBN-13 → book_id (last-write-wins on collision; collisions are rare).
        isbn = norm_isbn13(obj.get("isbn13", ""))
        if isbn:
            by_isbn13[isbn] = book_id
            isbn13_hits += 1

        # work_id → [book_id, ...]  — used for tie-break and dedup
        if work_id:
            work_to_books[work_id].append(book_id)
            with_work += 1

        # Title-only index (no author check; used as Layer 3 lookup pool)
        if norm_title:
            by_title[norm_title].append(book_id)

        # Per-author indexes — one entry per author in the book record
        for entry in obj.get("authors", []):
            aid = entry.get("author_id", "")
            name = author_names.get(aid, "")
            if not name:
                continue
            norm_name = normalize(name)
            if not norm_name:
                continue
            by_author[norm_name].append(book_id)
            if norm_title:
                by_author_title[norm_name + "||" + norm_title].append(book_id)

print(f"  {total:,} books processed — done", flush=True)


# ── Save outputs ─────────────────────────────────────────────────────────────

outputs = [
    ("goodreads_by_isbn13.json",        by_isbn13),
    ("goodreads_by_author_title.json",  dict(by_author_title)),
    ("goodreads_by_title.json",         dict(by_title)),
    ("goodreads_by_author.json",        dict(by_author)),
    ("goodreads_work_to_books.json",    dict(work_to_books)),
    ("goodreads_book_details.json",     book_details),
]

print()
for fname, data in outputs:
    path = OUT / fname
    print(f"  writing {fname} …", flush=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

# ── Summary ──────────────────────────────────────────────────────────────────

print()
print("=== Summary ===")
print(f"Total books processed         : {total:>10,}")
print(f"  with isbn13                 : {isbn13_hits:>10,}  ({isbn13_hits/total:.1%})")
print(f"  with work_id                : {with_work:>10,}  ({with_work/total:.1%})")
print(f"  with language_code          : {with_language:>10,}  ({with_language/total:.1%})")
print()
print(f"Unique by_isbn13 keys         : {len(by_isbn13):>10,}")
print(f"Unique by_author_title keys   : {len(by_author_title):>10,}")
print(f"Unique by_title keys          : {len(by_title):>10,}")
print(f"Unique by_author keys         : {len(by_author):>10,}")
print(f"Unique work_to_books keys     : {len(work_to_books):>10,}")
print(f"book_details entries          : {len(book_details):>10,}")
print()
print("v1 reference numbers (for sanity check):")
print("  by_author_title keys (v1)   :  2,679,278")
print("  by_title keys (v1)          :  1,695,226")
print("  authors (v1)                :    829,524")
print()
print("v2 by_author_title / by_title keys should be slightly LOWER than v1 —")
print("NFKD merges diacritic variants (e.g. 'Čapek' and 'Capek' → same key).")
