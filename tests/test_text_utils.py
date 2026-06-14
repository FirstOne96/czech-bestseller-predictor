"""
Smoke test for text_utils (NFKD diacritic folding + fuzzy match helpers).

Run from the repository root:

    python tests/test_text_utils.py

Output: one line per assertion. All ✓ means normalization/fuzzy-matching is
behaving as expected.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "matching"))

from text_utils import (
    normalize,
    norm_nkc_author,
    norm_isbn,
    fuzzy_title_match,
    fuzzy_author_match,
)

failed = 0


def check(label: str, got, expected) -> None:
    """Exact-equality assertion. Counts failures."""
    global failed
    ok = got == expected
    status = "✓" if ok else "✗"
    if not ok:
        failed += 1
    print(f"  {status} {label:<48} got={got!r:<40} expected={expected!r}")


def check_range(label: str, got: int, lo: int, hi: int) -> None:
    """Range assertion for fuzzy scores."""
    global failed
    ok = lo <= got <= hi
    status = "✓" if ok else "✗"
    if not ok:
        failed += 1
    print(f"  {status} {label:<48} got={got:<40} expected in [{lo}, {hi}]")


# -----------------------------------------------------------------------------
print("=== normalize() — basic behaviour ===")
check("lowercase",
      normalize("Harry Potter"),
      "harry potter")
check("strip leading article 'the'",
      normalize("The Lord of the Rings"),
      "lord of the rings")
check("preserve internal 'the' (only leading dropped)",
      normalize("Lord of the Rings"),
      "lord of the rings")
check("strip punctuation",
      normalize("Hannibal: Rising!"),
      "hannibal rising")
check("collapse whitespace",
      normalize("  spaced   out  text  "),
      "spaced out text")

# -----------------------------------------------------------------------------
print("\n=== normalize() — diacritic folding (the v2 improvement) ===")
check("Czech caron Č",
      normalize("Čapek"),
      "capek")
check("Czech multi-char",
      normalize("Šel přes řeku"),
      "sel pres reku")
check("Russian-Latin transliteration with apostrophe",
      normalize("Vasil'jevič"),
      "vasiljevic")
check("German umlaut ü",
      normalize("Müller"),
      "muller")
check("German Eszett ß",
      normalize("Straße"),
      "strasse")
check("Polish ł (non-decomposing — covered by replacement map)",
      normalize("Łukasz"),
      "lukasz")
check("Danish ø",
      normalize("Jørgen"),
      "jorgen")
check("Norse æ",
      normalize("Æthelred"),
      "aethelred")
check("French é, ï, ç combo",
      normalize("Émile naïve façade"),
      "emile naive facade")
check("Spanish ñ",
      normalize("Niño"),
      "nino")

# -----------------------------------------------------------------------------
print("\n=== norm_nkc_author() — NKC inversion ===")
check("Surname, Given",
      norm_nkc_author("King, Stephen"),
      "stephen king")
check("with diacritics",
      norm_nkc_author("Čapek, Karel"),
      "karel capek")
check("no comma (passthrough to normalize)",
      norm_nkc_author("Stephen King"),
      "stephen king")
check("Russian author from NKC",
      norm_nkc_author("Lunačarskij, Anatolij Vasil'jevič"),
      "anatolij vasiljevic lunacarskij")

# -----------------------------------------------------------------------------
print("\n=== norm_isbn() — unchanged behaviour ===")
check("ISBN-13 passthrough (dashes removed)",
      norm_isbn("978-80-7203-410-3"),
      "9788072034103")
check("ISBN-10 → ISBN-13 (EAN-13 check digit)",
      norm_isbn("80-7203-410-3"),
      "9788072034109")
check("12-digit (drop-9): prepend 9",
      norm_isbn("788072034103"),
      "9788072034103")

# -----------------------------------------------------------------------------
print("\n=== fuzzy_title_match() — token_set_ratio ===")
check("identical → 100",
      fuzzy_title_match("harry potter", "harry potter"),
      100)
check("empty input → 0",
      fuzzy_title_match("", "harry potter"),
      0)
check_range("subtitle subset 'quiet' ⊂ 'quiet power introverts'",
            fuzzy_title_match("quiet", "quiet power introverts"),
            88, 100)
check_range("subtitle drift 'maus' ⊂ 'maus survivors tale'",
            fuzzy_title_match("maus", "maus survivors tale"),
            88, 100)
check_range("unrelated 'hitler' vs 'spain'",
            fuzzy_title_match("hitler", "spain"),
            0, 50)

# -----------------------------------------------------------------------------
print("\n=== fuzzy_author_match() — token_set_ratio ===")
check("identical → 100",
      fuzzy_author_match("stephen king", "stephen king"),
      100)
check_range("middle-initial drift",
            fuzzy_author_match("john smith", "john a smith"),
            85, 100)
check_range("token reorder",
            fuzzy_author_match("king stephen", "stephen king"),
            85, 100)
check_range("patronymic missing",
            fuzzy_author_match("anna karenina", "anna sergeevna karenina"),
            85, 100)

# -----------------------------------------------------------------------------
print()
if failed:
    print(f"FAILED — {failed} assertion(s) did not match expected output.")
    sys.exit(1)
else:
    print("PASSED — all assertions OK. Step 1 done.")
