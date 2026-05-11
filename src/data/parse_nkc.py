"""
Stream the NKC MARC21-XML dump and emit a flat CSV of Czech translations.

The input file ``data/raw/nkc.xml.gz`` is 9.9 GB of plain UTF-8 XML (the
``.gz`` extension is misleading — the file is not actually gzipped). It must
be streamed; loading it fully into memory is impossible on a laptop.

Two outputs are produced in a single pass:

1. ``data/interim/nkc_translations.csv`` — one row per record matching the
   "Czech translation of a foreign work" filter, with the fields we need for
   downstream matching against SCKN (Czech title) and Goodreads (original
   title / OCLC).

2. ``reports/m2_nkc_inventory.json`` — coverage statistics over the filtered
   subset. The critical number is ``240_a`` coverage (the original-language
   uniform title): CLAUDE.md flags it as our best Goodreads-matching key when
   no OCLC is present, but warns that it may be sparse. M2 gives us the real
   number so M4 can plan accordingly.

Usage
-----
    python src/data/parse_nkc.py
    python src/data/parse_nkc.py --input ... --csv ... --report ...
    python src/data/parse_nkc.py --limit 100000   # quick smoke test

Filter
------
A record is kept iff some ``041`` datafield satisfies all of:
  - has a ``$a`` subfield equal to ``"cze"``  (target language is Czech)
  - has at least one ``$h`` subfield                (source language declared)
  - some ``$h`` subfield is not in ``{"cze", "und"}`` (not Cz→Cz or unknown)

The ``$h``-must-be-present clause is stricter than CLAUDE.md's prose:
records with ``041 ind1="0"`` (e.g. bilingual textbooks listing
``$a=cze $a=rus`` with no ``$h``) are otherwise indistinguishable from real
translations whose source happens to be omitted. Erring strict on the
training-set side; we'd rather miss a few translations than poison labels
with non-translations.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from lxml import etree

NS = "http://www.loc.gov/MARC21/slim"
RECORD_TAG = f"{{{NS}}}record"

DEFAULT_INPUT = Path("data/raw/nkc.xml.gz")
DEFAULT_CSV = Path("data/interim/nkc_translations.csv")
DEFAULT_REPORT = Path("reports/m2_nkc_inventory.json")

# Year inside 260$c / 264$c — strings like "1982", "c1982", "[1982]",
# "1982 (TZ 2)". Take the first 4-digit year in 1xxx or 20xx.
YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")

# 035$a holds many identifier flavors; we want only OCLC numbers.
# Format: "(OCoLC)12345678" — sometimes zero-padded, sometimes prefixed
# with "ocm"/"ocn"/"on". Keep just the digits.
OCLC_RE = re.compile(r"\(OCoLC\)(?:ocm|ocn|on)?0*(\d+)")

# MARC subfields are terminated by punctuation that follows ISBD rules
# (": ", " /", ".", ","). We strip these for clean joins downstream.
TRAILING_PUNCT = " :/.,;"


# --------------------------------------------------------------------------- #
# Per-record parsing
# --------------------------------------------------------------------------- #

def parse_record(elem) -> tuple[dict[str, str], list[tuple[str, str, str, dict[str, list[str]]]]]:
    """
    Walk a ``<record>`` element once and return its contents in two parts.

    Returns
    -------
    controls : dict[tag -> text]
        Controlfields (e.g. ``001``, ``008``). One value per tag.
    datas : list of (tag, ind1, ind2, subfields)
        One tuple per ``<datafield>`` instance, in document order. A given
        tag can repeat (e.g. several ``041`` or ``765`` fields), so we keep
        instances rather than merging — pairing of e.g. ``765 $t`` with
        ``765 $w`` would otherwise be lost.
    """
    controls: dict[str, str] = {}
    datas: list[tuple[str, str, str, dict[str, list[str]]]] = []

    for child in elem:
        local = etree.QName(child).localname
        if local == "controlfield":
            tag = child.get("tag", "")
            controls[tag] = (child.text or "").strip()
        elif local == "datafield":
            tag = child.get("tag", "")
            ind1 = child.get("ind1") or " "
            ind2 = child.get("ind2") or " "
            subs: dict[str, list[str]] = {}
            for sub in child:
                if etree.QName(sub).localname != "subfield":
                    continue
                code = sub.get("code", "")
                text = (sub.text or "").strip()
                subs.setdefault(code, []).append(text)
            datas.append((tag, ind1, ind2, subs))
    return controls, datas


def first(datas, tag: str, code: str) -> str | None:
    """First value of (tag, code), or None."""
    for t, _, _, subs in datas:
        if t == tag and code in subs and subs[code]:
            return subs[code][0]
    return None


def all_values(datas, tag: str, code: str) -> list[str]:
    """All values of (tag, code), across every datafield instance."""
    out: list[str] = []
    for t, _, _, subs in datas:
        if t == tag:
            out.extend(subs.get(code, []))
    return out


def instances(datas, tag: str):
    """Iterate over (ind1, ind2, subfields-dict) for every instance of tag."""
    for t, ind1, ind2, subs in datas:
        if t == tag:
            yield ind1, ind2, subs


def is_czech_translation(datas) -> bool:
    """See module docstring "Filter" section for the rule."""
    for _, _, subs in instances(datas, "041"):
        a_vals = subs.get("a", [])
        h_vals = subs.get("h", [])
        if "cze" not in a_vals:
            continue
        if not h_vals:
            continue
        if any(h and h not in ("cze", "und") for h in h_vals):
            return True
    return False


def clean(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.rstrip(TRAILING_PUNCT).strip()
    return s or None


def parse_year(s: str | None) -> int | None:
    if not s:
        return None
    m = YEAR_RE.search(s)
    return int(m.group(1)) if m else None


def parse_oclc(values: list[str]) -> str | None:
    """Return the first OCLC number found across all 035$a values."""
    for v in values:
        m = OCLC_RE.search(v)
        if m:
            return m.group(1)
    return None


def parse_isbn(values: list[str]) -> list[str]:
    """
    Extract clean ISBNs from all 020 $a values, deduplicated.

    NKC annotates bindings inline: "80-7203-410-3 (váz.)" or sometimes just
    "(brož.)" with no number at all. Strip everything from the first "("
    onward, then trailing ISBD punctuation. Discard if nothing remains.
    ISBN-10s are kept as-is alongside any ISBN-13s — older records have only
    ISBN-10 and we want them for matching.
    """
    out: list[str] = []
    for v in values:
        isbn = v.split("(")[0].strip().rstrip(TRAILING_PUNCT).strip()
        if isbn and isbn not in out:
            out.append(isbn)
    return out


def extract_row(controls, datas) -> dict[str, str]:
    """
    Project a translation record to the flat schema used in the CSV.

    Multi-valued fields (source languages, linked-record IDs, genres) are
    pipe-joined. Pipe is safe because MARC subfields don't contain it.
    """
    # 041 $h: source languages (one record may carry multiple, e.g. "eng|ger"
    # for compilations translated from several originals). Drop cze/und so
    # the column is meaningful as a feature.
    source_langs: list[str] = []
    for _, _, subs in instances(datas, "041"):
        for h in subs.get("h", []):
            if h and h not in ("cze", "und") and h not in source_langs:
                source_langs.append(h)

   # 765 $t: název originálu (pokud 240$a chybí, toto je záloha)
    # 765 $z: ISBN originálu — přímý klíč pro Goodreads matching
    # 765 $w: systémové/OCLC číslo originálu — pro budoucí použití
    original_title_765: str = ""
    original_isbn_765: list[str] = []
    original_sysnum_765: list[str] = []

    for _, _, subs in instances(datas, "765"):
        # $t — vezmi jen první neprázdnou hodnotu jako zálohu pro original_title
        if not original_title_765:
            for t in subs.get("t", []):
                t = clean(t)
                if t:
                    original_title_765 = t
                    break

        # $z — ISBN originálu, může být víc (různé edice)
        for z in subs.get("z", []):
            z_clean = z.split("(")[0].strip().rstrip(" :/.,;").strip()
            if z_clean and z_clean not in original_isbn_765:
                original_isbn_765.append(z_clean)
        # $w — systémové číslo, může obsahovat OCLC ve formátu (OCoLC)12345
        for w in subs.get("w", []):
            if w and w not in original_sysnum_765:
                original_sysnum_765.append(w)

    genres: list[str] = []
    for g in all_values(datas, "655", "a"):
        g = clean(g)
        if g and g not in genres:
            genres.append(g)

    isbns = parse_isbn(all_values(datas, "020", "a"))

    # 700 $a: added-entry personal names — editors, translators, co-authors.
    # Used as a fallback author signal when 100 $a is absent (anthologies).
    secondary: list[str] = []
    for a in all_values(datas, "700", "a"):
        a = clean(a)
        if a and a not in secondary:
            secondary.append(a)

    pub_year = parse_year(first(datas, "260", "c")) or parse_year(first(datas, "264", "c"))

    return {
    "nkc_id":            controls.get("001", ""),
    "oclc":              parse_oclc(all_values(datas, "035", "a")) or "",
    "czech_isbn":        "|".join(isbns),
    "czech_title":       clean(first(datas, "245", "a")) or "",
    # Preferuj 240$a jako kanonický original_title, 765$t jako zálohu
    "original_title":    clean(first(datas, "240", "a")) or original_title_765 or "",
    "original_isbn":     "|".join(original_isbn_765),   # nový sloupec — z 765$z
    "original_sysnum":   "|".join(original_sysnum_765), # nový sloupec — z 765$w
    "author":            clean(first(datas, "100", "a")) or "",
    "secondary_authors": "|".join(secondary),
    "czech_pub_year":    str(pub_year) if pub_year is not None else "",
    "source_lang":       "|".join(source_langs),
    "genres":            "|".join(genres),
}


CSV_FIELDS = [
    "nkc_id", "oclc", "czech_isbn", "czech_title", "original_title",
    "original_isbn", "original_sysnum",   # nahrazuje linked_records
    "author", "secondary_authors", "czech_pub_year", "source_lang", "genres",
]


# --------------------------------------------------------------------------- #
# Streaming + inventory
# --------------------------------------------------------------------------- #

@dataclass
class Inventory:
    """Counters maintained over the whole stream."""

    total_records: int = 0
    records_with_041: int = 0
    translation_records: int = 0

    # Field presence on the *filtered* (translation) subset.
    has_oclc: int = 0
    has_245_a: int = 0
    has_original_title: int = 0  # combined 240$a + 765$t fallback coverage
    has_100_a: int = 0
    has_260_c: int = 0
    has_264_c: int = 0
    year_resolved: int = 0
    has_765: int = 0
    has_655_a: int = 0
    has_020: int = 0
    has_original_isbn: int = 0    # přidej do @dataclass
    has_original_sysnum: int = 0  # přidej do @dataclass

    source_lang_counts: Counter = field(default_factory=Counter)
    pub_year_counts: Counter = field(default_factory=Counter)

    def update_translation(self, controls, datas, row) -> None:
        self.translation_records += 1
        if row["oclc"]:
            self.has_oclc += 1
        if row["czech_title"]:
            self.has_245_a += 1
        if row["original_title"]:
            self.has_original_title += 1
        if row["author"]:
            self.has_100_a += 1
        if first(datas, "260", "c"):
            self.has_260_c += 1
        if first(datas, "264", "c"):
            self.has_264_c += 1
        if row["czech_pub_year"]:
            self.year_resolved += 1
            self.pub_year_counts[int(row["czech_pub_year"])] += 1
        if any(t == "765" for t, _, _, _ in datas):
            self.has_765 += 1
        if row["original_isbn"]:
            self.has_original_isbn += 1
        if row["original_sysnum"]:
            self.has_original_sysnum += 1
        if all_values(datas, "655", "a"):
            self.has_655_a += 1
        if row["czech_isbn"]:
            self.has_020 += 1
        for lang in row["source_lang"].split("|"):
            if lang:
                self.source_lang_counts[lang] += 1

    def to_dict(self) -> dict:
        n = self.translation_records or 1  # avoid div-by-zero
        return {
            "total_records": self.total_records,
            "records_with_041": self.records_with_041,
            "translation_records": self.translation_records,
            "filter_rate": round(self.translation_records / max(self.total_records, 1), 4),
            "field_coverage_on_translations": {
                "035_oclc": round(self.has_oclc / n, 4),
                "245_a_czech_title": round(self.has_245_a / n, 4),
                "original_title_combined": round(self.has_original_title / n, 4),  # 240$a + 765$t fallback
                "100_a_author": round(self.has_100_a / n, 4),
                "260_c": round(self.has_260_c / n, 4),
                "264_c": round(self.has_264_c / n, 4),
                "year_resolved": round(self.year_resolved / n, 4),
                "765_link": round(self.has_765 / n, 4),
                "765_z_original_isbn":   round(self.has_original_isbn / n, 4),
                "765_w_original_sysnum": round(self.has_original_sysnum / n, 4),
                "655_a_genre": round(self.has_655_a / n, 4),
                "020_isbn": round(self.has_020 / n, 4),
            },
            "source_language_top20": dict(self.source_lang_counts.most_common(20)),
            "publication_year_distribution": dict(sorted(self.pub_year_counts.items())),
        }


def iter_records(path: Path) -> Iterator:
    """
    Yield each ``<record>`` element from the file, freeing memory as we go.

    The clear-then-delete-preceding-siblings dance is the canonical lxml
    pattern for streaming large XML: ``elem.clear()`` releases the element's
    own children, but the parent still holds a reference to the cleared
    element until we drop it from the parent's child list.
    """
    context = etree.iterparse(str(path), tag=RECORD_TAG, recover=True)
    for _, elem in context:
        yield elem
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context


def process(
    input_path: Path,
    csv_path: Path,
    report_path: Path,
    limit: int | None = None,
    min_year: int | None = None,
    progress_every: int = 100_000,
) -> Inventory:
    inv = Inventory()
    started = time.monotonic()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for elem in iter_records(input_path):
            inv.total_records += 1
            controls, datas = parse_record(elem)

            has_041 = any(t == "041" for t, _, _, _ in datas)
            if has_041:
                inv.records_with_041 += 1

            if is_czech_translation(datas):
                row = extract_row(controls, datas)
                year_val = int(row["czech_pub_year"]) if row["czech_pub_year"] else None
                if (min_year is not None and (year_val is None or year_val < min_year)) or (year_val is not None and year_val < 1800):
                    continue
                writer.writerow(row)
                inv.update_translation(controls, datas, row)

            if inv.total_records % progress_every == 0:
                elapsed = time.monotonic() - started
                rate = inv.total_records / elapsed if elapsed else 0
                print(
                    f"  {inv.total_records:>10,} records  "
                    f"({inv.translation_records:,} kept)  "
                    f"{rate:>6,.0f} rec/s",
                    flush=True,
                )

            if limit is not None and inv.total_records >= limit:
                break

    elapsed = time.monotonic() - started
    report = inv.to_dict()
    report["input_file"] = str(input_path)
    report["csv_file"] = str(csv_path)
    report["limit"] = limit
    report["min_year"] = min_year
    report["elapsed_seconds"] = round(elapsed, 1)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return inv


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="Parse NKC MARC21-XML to a flat translation CSV.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N records (smoke-test mode).")
    p.add_argument("--min-year", type=int, default=None,
                   help="Drop records with czech_pub_year below this value (e.g. 2000).")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming {args.input} → {args.csv}", flush=True)
    inv = process(args.input, args.csv, args.report, limit=args.limit, min_year=args.min_year)

    print(
        f"\nDone. {inv.total_records:,} records seen, "
        f"{inv.translation_records:,} translations written to {args.csv}.\n"
        f"Inventory report: {args.report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
