"""
Stream goodreads_book_authors.json.gz and goodreads_books.json.gz once each
and write four lookup dicts to data/interim/.
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW  = REPO / "data" / "raw"
OUT  = REPO / "data" / "interim"

AUTHORS_GZ = RAW / "goodreads_book_authors.json.gz"
BOOKS_GZ   = RAW / "goodreads_books.json.gz"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import normalize


def norm_isbn13(raw: str) -> str:
    digits = "".join(c for c in raw if c.isdigit())
    return digits if len(digits) == 13 else ""


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

out_path = OUT / "goodreads_author_names.json"
out_path.write_text(json.dumps(author_names, ensure_ascii=False), encoding="utf-8")
print(f"  → wrote {out_path.name}", flush=True)


# ── Pass 2: books ────────────────────────────────────────────────────────────

print("Pass 2: streaming books …", flush=True)
by_isbn13:       dict[str, str]        = {}
by_author_title: dict[str, list[str]]  = defaultdict(list)
by_title:        dict[str, list[str]]  = defaultdict(list)

total = isbn13_hits = 0

with gzip.open(BOOKS_GZ, "rt", encoding="utf-8") as fh:
    for line in fh:
        obj     = json.loads(line)
        book_id = obj.get("book_id", "")
        if not book_id:
            continue

        total += 1
        if total % 200_000 == 0:
            print(f"  {total:,} books processed …", flush=True)

        # isbn13
        isbn = norm_isbn13(obj.get("isbn13", ""))
        if isbn:
            by_isbn13[isbn] = book_id
            isbn13_hits += 1

        # title
        raw_title = obj.get("title_without_series", "") or obj.get("title", "")
        norm_title = normalize(raw_title)
        if not norm_title:
            continue

        by_title[norm_title].append(book_id)

        # author+title and author-only — one entry per author
        for entry in obj.get("authors", []):
            aid = entry.get("author_id", "")
            name = author_names.get(aid, "")
            if not name:
                continue
            norm_name = normalize(name)
            by_author_title[norm_name + "||" + norm_title].append(book_id)

print(f"  {total:,} books processed — done", flush=True)

for fname, data in [
    ("goodreads_by_isbn13.json",       by_isbn13),
    ("goodreads_by_author_title.json", dict(by_author_title)),
    ("goodreads_by_title.json",        dict(by_title)),
]:
    p = OUT / fname
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"  → wrote {fname}", flush=True)

# ── Summary ──────────────────────────────────────────────────────────────────

print()
print("=== Summary ===")
print(f"Total books processed        : {total:>8,}")
print(f"ISBN-13 hits                 : {isbn13_hits:>8,}  ({isbn13_hits/total:.1%})")
print(f"Unique author+title keys     : {len(by_author_title):>8,}")
print(f"Unique title keys            : {len(by_title):>8,}")
