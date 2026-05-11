"""
Shared text normalization helpers.

Imported by both src/data/build_goodreads_lookup.py, build_matched_dataset.py,
and notebooks/04_matching_spike.ipynb — any change here affects all three.
"""
import re
import string

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLES = {
    "the", "a", "an",                    # English
    "der", "die", "das", "ein", "eine",  # German
    "le", "la", "les", "un", "une",      # French
}


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, drop leading article, collapse whitespace."""
    words = s.lower().translate(_PUNCT_TABLE).split()
    if words and words[0] in _ARTICLES:
        words = words[1:]
    return " ".join(words)


def norm_nkc_author(author: str) -> str:
    """Invert NKC 'Surname, Given' to 'Given Surname', then normalize."""
    if "," in author:
        surname, given = author.split(",", 1)
        author = given.strip() + " " + surname.strip()
    return normalize(author)


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
