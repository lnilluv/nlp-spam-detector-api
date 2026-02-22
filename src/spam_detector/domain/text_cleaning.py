"""Text normalization rules for spam classification."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SYMBOL_RE = re.compile(r"[^a-z0-9\s']", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def clean_text(message: str) -> str:
    """Normalize incoming text while preserving semantic tokens."""
    without_urls = _URL_RE.sub(" ", message)
    lowered = without_urls.lower()
    without_symbols = _SYMBOL_RE.sub(" ", lowered)
    normalized = _SPACE_RE.sub(" ", without_symbols).strip()
    return normalized
