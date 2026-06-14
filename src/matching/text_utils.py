"""
Shared text normalization helpers.

Imported by build_goodreads_lookup.py, build_matched_dataset.py, and various
notebooks. Any change here affects all downstream matching.

v2 changes (2026-05-22)
-----------------------
- `normalize()` now applies NFKD diacritic folding before punctuation strip.
    Czech  "Čapek"   → "capek"
    German "Müller"  → "muller"
    Polish "Łukasz"  → "lukasz"
- Added `_LATIN_REPLACEMENTS` map for characters NFKD does not decompose
  on its own (ł, ø, æ, ß, đ, þ, ı).
- New functions `fuzzy_title_match()` and `fuzzy_author_match()` using
  `rapidfuzz.fuzz.token_set_ratio`. Required for new matching cascade
  (Layers 2 and 3).
- `norm_nkc_author()` and `norm_isbn()` semantics unchanged.
"""
import re
import string
import unicodedata

from rapidfuzz import fuzz

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLES = {
    "the", "a", "an",                    # English
    "der", "die", "das", "ein", "eine",  # German
    "le", "la", "les", "un", "une",      # French
}

# Latin-script characters that NFKD does NOT decompose into base + diacritic.
# Pre-replaced before the NFKD pass so the final string is pure ASCII for these
# letters.  Coverage targets: Polish, Scandinavian, German Eszett, Croatian,
# Icelandic, Turkish — i.e. languages that show up in NKC's source_lang column.
_LATIN_REPLACEMENTS = {
    "ł": "l", "Ł": "L",       # Polish
    "ø": "o", "Ø": "O",       # Danish, Norwegian
    "æ": "ae", "Æ": "AE",     # Old English, Norse
    "œ": "oe", "Œ": "OE",     # French ligature
    "ß": "ss",                # German Eszett
    "đ": "d", "Đ": "D",       # Croatian, Vietnamese
    "þ": "th", "Þ": "TH",     # Icelandic
    "ı": "i", "İ": "I",       # Turkish dotless I
}


def normalize(s: str) -> str:
    """Canonical form: NFKD fold → lowercase → strip punctuation → drop leading
    article → collapse whitespace.

    Examples
    --------
    >>> normalize("Čapek")
    'capek'
    >>> normalize("Lunačarskij")
    'lunacarskij'
    >>> normalize("The Lord of the Rings")
    'lord of the rings'
    >>> normalize("Müller über die Straße")
    'muller uber die strasse'
    >>> normalize("Łukasz")
    'lukasz'
    """
    # Step 1: replace non-decomposing Latin characters that NFKD won't touch.
    for src, dst in _LATIN_REPLACEMENTS.items():
        if src in s:
            s = s.replace(src, dst)
    # Step 2: NFKD decomposes "ü" → "u" + combining diaeresis, "č" → "c" +
    # combining caron, etc.
    s = unicodedata.normalize("NFKD", s)
    # Step 3: drop the combining marks left over from step 2 (Unicode category
    # "Mn" = Mark, Nonspacing).
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Step 4: lowercase, strip punctuation, split.
    words = s.lower().translate(_PUNCT_TABLE).split()
    # Step 5: drop leading article (e.g. "The Lord of the Rings" → "lord of the rings").
    if words and words[0] in _ARTICLES:
        words = words[1:]
    return " ".join(words)


def norm_nkc_author(author: str) -> str:
    """Invert NKC's 'Surname, Given' format to 'Given Surname', then normalize.

    Examples
    --------
    >>> norm_nkc_author("King, Stephen")
    'stephen king'
    >>> norm_nkc_author("Čapek, Karel")
    'karel capek'
    >>> norm_nkc_author("Lunačarskij, Anatolij")
    'anatolij lunacarskij'
    """
    if "," in author:
        surname, given = author.split(",", 1)
        author = given.strip() + " " + surname.strip()
    return normalize(author)


def fuzzy_title_match(s1: str, s2: str) -> int:
    """Token-set ratio between two pre-normalized titles, 0-100.

    Returns 100 when one token set is a subset of the other — the key property
    for catching subtitle variants:

        "quiet"                vs  "quiet power introverts"          → 100
        "maus"                 vs  "maus survivors tale"             → 100
        "harry potter"         vs  "harry potter philosophers stone" → 100
        "hitler"               vs  "spain"                           → ~18

    Caller MUST normalize both inputs via ``normalize()`` first — this function
    does not re-normalize.
    """
    if not s1 or not s2:
        return 0
    return int(fuzz.token_set_ratio(s1, s2))


def fuzzy_author_match(s1: str, s2: str) -> int:
    """Token-set ratio between two pre-normalized author names, 0-100.

    Robust to:
      - middle-initial drift:    "john smith"   vs "john a smith"   → 100
      - token reordering:        "king stephen" vs "stephen king"   → 100
      - extra/missing patronymic: "anna sergeevna" vs "anna"        → 100

    NOT robust to single-token transliteration variants — e.g.
    "lunacharsky" vs "lunacarskij" scores ~65 because both are single tokens
    with different character sequences. Those cases are expected to match in
    Layer 1 already (NFKD folding produces equal strings) or to remain
    unmatched.

    Caller MUST normalize both inputs via ``normalize()`` or
    ``norm_nkc_author()`` first.
    """
    if not s1 or not s2:
        return 0
    return int(fuzz.token_set_ratio(s1, s2))


def _isbn10_to_isbn13(s: str) -> str:
    """Convert a bare 10-char ISBN-10 string to ISBN-13 (EAN-13 check digit)."""
    base = "978" + s[:9]
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(base))
    check = (10 - total % 10) % 10
    return base + str(check)


def norm_isbn(raw: str) -> str:
    """Normalise an ISBN to a canonical 13-digit string.

    Steps applied in order:
    1. Strip dashes/spaces, uppercase.
    2. 10-char strings (ISBN-10): convert to ISBN-13 via the EAN-13 formula.
    3. 12-digit strings (truncated ISBN-13 with leading 9 dropped): prepend '9'.
    """
    s = re.sub(r"[\s\-]", "", raw).upper().strip()
    if len(s) == 10 and s[:9].isdigit() and (s[9].isdigit() or s[9] == "X"):
        s = _isbn10_to_isbn13(s)
    elif len(s) == 12 and s.isdigit():
        s = "9" + s
    return s
