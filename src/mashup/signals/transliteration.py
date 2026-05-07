"""Lyric normalization for repetition matching.

Repetition detection benefits from collapsing trivial variations:
- Capitalization, punctuation, whitespace
- Unicode diacritics (Devanagari combining marks, Latin accents)
- Repeated vowels (piyaaa -> piya, looove -> love), common in transliterated
  Indian lyrics where romanization is non-canonical between releases
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_REPEATED_VOWEL = re.compile(r"([aeiouAEIOU])\1+")
_WHITESPACE = re.compile(r"\s+")


def normalize_lyric(line: str) -> str:
    s = unicodedata.normalize("NFKD", line)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _REPEATED_VOWEL.sub(r"\1", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s
