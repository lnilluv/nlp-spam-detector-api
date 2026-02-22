"""Keyword-based baseline adapter used as safe fallback."""

from __future__ import annotations


class KeywordSpamModelAdapter:
    """Simple deterministic model for local smoke tests and fallback mode."""

    _SPAM_TERMS = {
        "free",
        "win",
        "winner",
        "prize",
        "claim",
        "offer",
        "urgent",
        "call",
        "cash",
        "credit",
        "entry",
        "limited",
    }

    def predict_proba(self, messages: list[str]) -> list[float]:
        scores: list[float] = []
        for message in messages:
            tokens = set(message.split())
            if not tokens:
                scores.append(0.01)
                continue
            matches = len(tokens.intersection(self._SPAM_TERMS))
            score = min(0.99, 0.1 + matches * 0.2)
            scores.append(round(score, 4))
        return scores
