"""
Diagnostic: trace specific SCKN-positive books through the pipeline to find
where the label gets lost.

For each suspect ISBN, prints:
  - Is it in the SCKN charts?
  - Does it appear in nkc_translations.csv? (filter `is_czech_translation` passed)
  - If yes, what NKC fields does that record have? (author / original_title)
  - Does the record exist in matched_dataset.csv? With what match_layer / sckn_appearances?
  - As a cross-check: search matched_dataset by author+czech_title fuzzy to find
    a "should-have-been-matched" row that may exist under a different ISBN.

Usage:
    python src/diagnostics/diagnose_missing_sckn.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "matching"))
from text_utils import norm_isbn, normalize  # noqa: E402

INTERIM = REPO / "data" / "interim"
RAW     = REPO / "data" / "raw"

# Books to investigate. Add more as needed — format: (raw ISBN, label).
SUSPECTS = [
    ("978-80-257-1324-2", "Animal Farm — Argo 2015 (Orwell)"),
    ("80-204-0935-1",     "Dvě věže (Pán prstenů II, ilustr.vyd.) — Mladá fronta 2003 (Tolkien)"),
]

# --------------------------------------------------------------------------- #
# Phase 1: load SCKN, NKC, matched_dataset
# --------------------------------------------------------------------------- #

print("=" * 80)
print("Loading SCKN charts …")
sckn_rows_by_isbn: dict[str, list[dict]] = {}
with open(RAW / "sckn_charts.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        isbn = norm_isbn(row.get("isbn", ""))
        if isbn and len(isbn) == 13:
            sckn_rows_by_isbn.setdefault(isbn, []).append(row)
print(f"  SCKN unique normalized ISBNs: {len(sckn_rows_by_isbn):,}")

print("\nLoading nkc_translations.csv …")
nkc_rows = []
with open(INTERIM / "nkc_translations.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        nkc_rows.append(row)
print(f"  NKC translation records: {len(nkc_rows):,}")

# Build index NKC by normalized czech_isbn
nkc_by_isbn: dict[str, list[dict]] = {}
for r in nkc_rows:
    for raw_isbn in r.get("czech_isbn", "").split("|"):
        normed = norm_isbn(raw_isbn.strip())
        if normed and len(normed) == 13:
            nkc_by_isbn.setdefault(normed, []).append(r)

print("\nLoading matched_dataset.csv …")
matched_rows = []
with open(INTERIM / "matched_dataset.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        matched_rows.append(row)
print(f"  matched_dataset rows: {len(matched_rows):,}")

# Index by nkc_id for quick lookup
matched_by_nkc: dict[str, dict] = {r["nkc_id"]: r for r in matched_rows}


# --------------------------------------------------------------------------- #
# Phase 2: investigate each suspect
# --------------------------------------------------------------------------- #

for raw_isbn, label in SUSPECTS:
    print("\n" + "=" * 80)
    print(f"SUSPECT: {label}")
    print(f"  Raw ISBN: {raw_isbn}")
    isbn13 = norm_isbn(raw_isbn)
    print(f"  Normalized: {isbn13}")
    print("=" * 80)

    # SCKN side
    sckn_entries = sckn_rows_by_isbn.get(isbn13, [])
    print(f"\n[SCKN] Entries with this ISBN: {len(sckn_entries)}")
    for s in sckn_entries[:5]:
        print(f"    {s['year']}/w{s['week']}  rank={s['rank']}  cat={s['category']}  "
              f"author={s['author']!r}  title={s['title']!r}  publisher={s['publisher']!r}")
    if len(sckn_entries) > 5:
        print(f"    ... and {len(sckn_entries)-5} more")

    # NKC side
    nkc_entries = nkc_by_isbn.get(isbn13, [])
    print(f"\n[NKC] Translation records with this ISBN: {len(nkc_entries)}")
    if not nkc_entries:
        print("    ⚠ ISBN NOT FOUND in nkc_translations.csv.")
        print("    Possible reasons:")
        print("      - NKC didn't catalogue this edition as a translation (041 missing $h)")
        print("      - parse_nkc.py stripped the ISBN due to format quirk")
        print("      - Edition is in NKC but not under this exact ISBN")
        # Try fuzzy lookup: search by author+czech_title
        # Pick a related SCKN entry to extract author/title
        if sckn_entries:
            sckn_author = sckn_entries[0]["author"]
            sckn_title  = sckn_entries[0]["title"]
            print(f"    Searching NKC by author+title fuzzy…")
            print(f"      target: author~={sckn_author!r}, czech_title~={sckn_title!r}")
            norm_a = normalize(sckn_author)
            norm_t = normalize(sckn_title)
            candidates = []
            for r in nkc_rows:
                if not r["author"] or not r["czech_title"]:
                    continue
                # NKC author is "Surname, Given" — strip comma for normalize
                nkc_author_norm = normalize(r["author"].replace(",", " "))
                nkc_title_norm  = normalize(r["czech_title"])
                # Check if any normalized author token overlap AND title overlap
                if (any(tok in nkc_author_norm for tok in norm_a.split() if len(tok) > 3)
                        and any(tok in nkc_title_norm for tok in norm_t.split() if len(tok) > 3)):
                    candidates.append(r)
            print(f"      candidates found: {len(candidates)}")
            for c in candidates[:8]:
                print(f"        nkc_id={c['nkc_id']}  author={c['author']!r}  "
                      f"czech_title={c['czech_title']!r}  "
                      f"czech_isbn={c['czech_isbn']!r}  year={c['czech_pub_year']}")
    else:
        for nkc in nkc_entries[:5]:
            print(f"    nkc_id={nkc['nkc_id']}  year={nkc['czech_pub_year']}")
            print(f"      author={nkc['author']!r}")
            print(f"      original_title={nkc['original_title']!r}")
            print(f"      czech_title={nkc['czech_title']!r}")
            print(f"      czech_isbn={nkc['czech_isbn']!r}")
            print(f"      source_lang={nkc['source_lang']!r}")
            # Check matched_dataset
            m = matched_by_nkc.get(nkc["nkc_id"])
            if m:
                print(f"      → matched_dataset: layer={m['match_layer']}  "
                      f"matched_book_id={m['matched_book_id']}  "
                      f"work_id={m['gr_work_id']}  "
                      f"sckn_appearances={m['sckn_appearances']}")
            else:
                print(f"      ⚠ NOT in matched_dataset.csv")

    # Cross-check: search matched_dataset by SCKN author+title
    if sckn_entries:
        sckn_author = sckn_entries[0]["author"]
        sckn_title  = sckn_entries[0]["title"]
        print(f"\n[CROSS-CHECK] Matched_dataset rows that look like this book "
              f"(by czech_title fuzzy ~ {sckn_title!r}):")
        norm_target_title = normalize(sckn_title)
        hits = []
        for r in matched_rows:
            cz_title_norm = normalize(r.get("czech_title", ""))
            if norm_target_title and norm_target_title in cz_title_norm:
                hits.append(r)
        print(f"  found: {len(hits)} rows")
        for h in hits[:6]:
            print(f"    nkc_id={h['nkc_id']}  author={h['author']!r}  "
                  f"czech_title={h['czech_title']!r}  year={h['czech_pub_year']}")
            print(f"      czech_isbn={h['czech_isbn']!r}")
            print(f"      layer={h['match_layer']}  matched_book_id={h['matched_book_id']}  "
                  f"sckn_appearances={h['sckn_appearances']}")

print("\n" + "=" * 80)
print("Diagnostic complete.")
