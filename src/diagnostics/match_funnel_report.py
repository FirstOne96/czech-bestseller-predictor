"""
Compute the SCKN -> NKC -> Goodreads match funnel and write it to
``reports/match_funnel.json``.

The cascade matcher (``src/matching/build_matched_dataset.py``) prints layer
counts to stdout at runtime but never persists them. This script reconstructs
the full funnel from the saved pipeline artifacts so the numbers are
reproducible and citable in the thesis.

The funnel answers: of the books that ever charted on SCKN, how many survive
each join in the pipeline?

    SCKN charts (labels)
      -> exist in NKC translation catalogue?   (SCKN ISBN found in NKC)
        -> NKC record matched to a Goodreads book?  (cascade layer != unmatched)
          -> survive training filters + dedup?      (the modelable set)

Reads (no large dumps — only the saved intermediates)
-----------------------------------------------------
- data/raw/sckn_charts.csv
- data/interim/matched_dataset.csv      (all NKC records + match_layer + sckn_appearances)
- data/interim/training_dataset.csv     (post-filter, post-dedup modelable set)

Writes
------
- reports/match_funnel.json

Usage
-----
    python src/diagnostics/match_funnel_report.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "matching"))
from text_utils import norm_isbn  # noqa: E402

RAW     = REPO / "data" / "raw"
INTERIM = REPO / "data" / "interim"
REPORTS = REPO / "reports"


def _isbn_set(values) -> set[str]:
    """Normalize an iterable of raw ISBN strings to the set of valid ISBN-13s."""
    out: set[str] = set()
    for v in values:
        if v is None:
            continue
        n = norm_isbn(str(v).strip())
        if n and len(n) == 13:
            out.add(n)
    return out


def build_funnel() -> dict:
    # ── SCKN side ────────────────────────────────────────────────────────────
    sckn = pd.read_csv(RAW / "sckn_charts.csv", dtype=str)
    sckn_isbns = _isbn_set(sckn["isbn"].dropna())

    # ── NKC + cascade results ────────────────────────────────────────────────
    m = pd.read_csv(INTERIM / "matched_dataset.csv", dtype=str)
    m["sckn_app"] = pd.to_numeric(m["sckn_appearances"], errors="coerce").fillna(0).astype(int)
    matched = m["match_layer"] != "unmatched"
    pos = m["sckn_app"] >= 1

    # Every ISBN that appears on any NKC translation record (pipe-separated field)
    nkc_isbns = _isbn_set(
        raw for field in m["czech_isbn"].dropna() for raw in str(field).split("|")
    )
    sckn_in_nkc = sckn_isbns & nkc_isbns

    layer_counts = {k: int(v) for k, v in Counter(m["match_layer"]).items()}
    n_pos = int(pos.sum())
    n_pos_matched = int((pos & matched).sum())

    # ── Final modelable set ──────────────────────────────────────────────────
    t = pd.read_csv(INTERIM / "training_dataset.csv", dtype=str)
    t_pos = int((t["sckn_bestseller"].astype(str).str.lower() == "true").sum())

    def pct(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    return {
        "_about": "SCKN -> NKC -> Goodreads match funnel. Regenerate with "
                  "python src/diagnostics/match_funnel_report.py",
        "sckn": {
            "chart_rows": int(len(sckn)),
            "unique_isbn13": len(sckn_isbns),
        },
        "sckn_to_nkc": {
            "sckn_isbns_in_nkc": len(sckn_in_nkc),
            "share_of_sckn": pct(len(sckn_in_nkc), len(sckn_isbns)),
            "sckn_isbns_not_in_nkc": len(sckn_isbns - sckn_in_nkc),
            "note": "Books not in NKC are largely Czech-original titles and "
                    "non-book chart categories — out of scope (translations only).",
        },
        "nkc_to_goodreads": {
            "nkc_records_total": int(len(m)),
            "matched_to_goodreads": int(matched.sum()),
            "match_rate": pct(int(matched.sum()), len(m)),
            "layer_counts": layer_counts,
        },
        "bestsellers": {
            "nkc_records_sckn_positive": n_pos,
            "also_matched_to_goodreads": n_pos_matched,
            "match_rate_among_positives": pct(n_pos_matched, n_pos),
            "positive_but_unmatched": n_pos - n_pos_matched,
        },
        "modelable_set": {
            "rows_unique_foreign_works": int(len(t)),
            "positives": t_pos,
            "positive_rate": pct(t_pos, len(t)),
        },
    }


def main() -> dict:
    funnel = build_funnel()
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "match_funnel.json"
    out.write_text(json.dumps(funnel, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO)}\n")
    print(json.dumps(funnel, ensure_ascii=False, indent=2))
    return funnel


if __name__ == "__main__":
    main()
