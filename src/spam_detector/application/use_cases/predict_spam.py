"""Use case for spam inference."""

from __future__ import annotations

from spam_detector.application.ports.model_port import ModelPort
from spam_detector.domain.prediction import Prediction
from spam_detector.domain.text_cleaning import clean_text


class PredictSpamUseCase:
    def __init__(self, model_port: ModelPort, threshold: float = 0.5) -> None:
        self._model_port = model_port
        self._threshold = threshold

    def execute(self, message: str) -> Prediction:
        normalized = clean_text(message)
        score = float(self._model_port.predict_proba([normalized])[0])
        is_spam = score >= self._threshold
        label = "spam" if is_spam else "ham"
        return Prediction(label=label, score=round(score, 4), is_spam=is_spam)
