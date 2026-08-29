"""
Scrape SCKN (Svaz českých knihkupců a nakladatelů) weekly bestseller charts.

The charts live at https://www.sckn.cz/r-p-b/?year=YEAR&week=WEEK and cover
three categories per week:
  - beletrie              (fiction)
  - naučná literatura     (non-fiction)
  - literatura pro děti a mládež  (children's & YA)

Each chart has up to 10 entries per category (older charts occasionally more).
Non-existent weeks return an empty table, so iterating 1–53 is safe.

Usage
-----
    python src/data/scrape_sckn.py                          # 2003 → current year
    python src/data/scrape_sckn.py --start-year 2010 --end-year 2017
    python src/data/scrape_sckn.py --output data/raw/sckn_charts.csv

Output: CSV with columns year, week, category, rank, isbn, author, title, publisher
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.sckn.cz/r-p-b/"
DEFAULT_OUTPUT = Path("data/raw/sckn_charts.csv")
DEFAULT_DELAY = 0.5  # seconds between requests — be polite to the server


def fix_c1_mojibake(text: str) -> str:
    """
    Repair SCKN's server-side encoding bug.

    SCKN's HTML is served as UTF-8, but the backend encoded some Windows-1250
    bytes (notably Š=0x8A, Ž=0x8E, Ť=0x8D, š=0x9A, ž=0x9E, ť=0x9D) through
    Latin-1 → UTF-8 instead of decoding them properly first. As a result,
    those characters arrive as C1 control codepoints (U+0080–U+009F) inside
    an otherwise valid UTF-8 stream. C1 controls never appear in normal HTML
    text, so we treat every codepoint in that range as a Windows-1250 byte
    and recover the real character.

    Codepoints with no Windows-1250 mapping (the 5 empty slots at 0x81, 0x83,
    0x88, 0x90, 0x98) and stray U+FFFD replacement characters are dropped —
    keeping them produces invisible / square glyphs in the CSV.
    """
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if 0x80 <= cp <= 0x9F:
            try:
                out.append(ch.encode("latin-1").decode("windows-1250"))
                continue
            except UnicodeDecodeError:
                continue  # unmappable C1 control — drop instead of leaving a square
            except UnicodeEncodeError:
                pass
        elif cp == 0xFFFD:
            continue  # replacement char from a prior decoding failure — drop
        out.append(ch)
    return "".join(out)


def decode_sckn_response(content: bytes) -> str:
    """
    Decode an SCKN HTTP response body to clean text.

    The page is mostly UTF-8, but the server has two encoding hazards:

    A. Czech characters double-encoded as `C2 8x/9x` byte pairs that UTF-8
       decodes to C1 control codepoints (U+0080–U+009F). Repaired by
       :func:`fix_c1_mojibake` after decoding.
    B. Occasional raw Windows-1250 bytes that are not valid UTF-8. The
       default decoder would replace these with U+FFFD (renders as a
       square). We avoid that by decoding the bytes incrementally: take the
       longest UTF-8 prefix, fall back to Windows-1250 for any byte that
       breaks the decode, then continue.
    """
    parts: list[str] = []
    i, n = 0, len(content)
    while i < n:
        try:
            parts.append(content[i:].decode("utf-8"))
            break
        except UnicodeDecodeError as e:
            if e.start > 0:
                parts.append(content[i:i + e.start].decode("utf-8"))
            try:
                parts.append(content[i + e.start:i + e.end].decode("windows-1250"))
            except UnicodeDecodeError:
                pass  # unmappable byte — drop
            i += e.end
    return fix_c1_mojibake("".join(parts))


def parse_chart(html: str, year: int, week: int) -> list[dict]:
    """
    Parse one SCKN chart page.

    Returns a list of dicts, one per book entry. Position 1 is rank 1 within
    that category for that week.
    """
    soup = BeautifulSoup(html, "html.parser")

    # The chart is always the first <table class="ranking"> on the page.
    # The second one is a sidebar with archive year links.
    tables = soup.find_all("table", class_="ranking")
    if not tables:
        return []

    chart_table = tables[0]

    # Category names come from <h2> inside <thead>
    category_headers = [h.get_text(strip=True) for h in chart_table.find_all("h2")]
    tbodies = chart_table.find_all("tbody")

    rows = []
    for category, tbody in zip(category_headers, tbodies):
        for rank, tr in enumerate(tbody.find_all("tr"), start=1):
            cells = tr.find_all("td")
            if len(cells) < 4:
                continue  # skip malformed rows
            rows.append({
                "year": year,
                "week": week,
                "category": category,
                "rank": rank,
                "isbn": cells[0].get_text(strip=True),
                "author": cells[1].get_text(strip=True),
                "title": cells[2].get_text(strip=True),
                "publisher": cells[3].get_text(strip=True),
            })

    return rows


def fetch_chart(session: requests.Session, year: int, week: int, max_attempts: int = 3) -> str:
    """
    Fetch one chart page with retries, returning the body as repaired text.

    Encoding repair (UTF-8 + C1 mojibake fix) is delegated to
    :func:`decode_sckn_response`; we operate on raw bytes from ``resp.content``
    rather than ``resp.text`` so that no broken bytes are silently turned into
    U+FFFD before we get a chance to recover them.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = session.get(
                BASE_URL, params={"year": year, "week": week}, timeout=15
            )
            resp.raise_for_status()
            return decode_sckn_response(resp.content)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(2 * (attempt + 1))  # 2s, 4s
    assert last_exc is not None
    raise last_exc


def scrape_all(
    start_year: int = 2003,
    end_year: int | None = None,
    delay: float = DEFAULT_DELAY,
) -> list[dict]:
    """
    Scrape every week of every year in [start_year, end_year].

    Prints one progress line per year to stdout.
    """
    if end_year is None:
        end_year = datetime.now().year

    session = requests.Session()
    # Identify ourselves; some sites block generic Python user-agents
    session.headers["User-Agent"] = (
        "thesis-scraper/1.0 (academic research; contact: student)"
    )

    all_rows: list[dict] = []

    for year in range(start_year, end_year + 1):
        year_rows = 0
        for week in range(1, 54):  # ISO weeks go up to 53
            try:
                html = fetch_chart(session, year, week)
            except requests.RequestException as exc:
                print(
                    f"  WARNING: {year} w{week:02d} dropped after retries — {exc}",
                    file=sys.stderr,
                )
                continue

            rows = parse_chart(html, year, week)
            all_rows.extend(rows)
            year_rows += len(rows)
            time.sleep(delay)

        print(f"{year}: {year_rows} entries", flush=True)

    return all_rows


FIELDNAMES = ["year", "week", "category", "rank", "isbn", "author", "title", "publisher"]


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape SCKN printed-book bestseller charts"
    )
    parser.add_argument(
        "--start-year", type=int, default=2003, help="First year to scrape (default: 2003)"
    )
    parser.add_argument(
        "--end-year", type=int, default=None, help="Last year to scrape (default: current year)"
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Output CSV path"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY, help="Seconds between requests"
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    end = args.end_year or datetime.now().year

    print(
        f"Scraping SCKN charts {args.start_year}–{end} → {out_path}",
        flush=True,
    )

    rows = scrape_all(args.start_year, args.end_year, args.delay)

    if not rows:
        print("No data scraped — check the SCKN website manually.")
        sys.exit(1)

    save_csv(rows, out_path)
    print(f"\nDone. Saved {len(rows):,} rows to {out_path}")


if __name__ == "__main__":
    main()
