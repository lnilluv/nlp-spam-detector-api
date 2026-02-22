"""Dependency wiring for runtime profiles."""

from __future__ import annotations

import os
from dataclasses import dataclass

from spam_detector.adapters.api.fastapi_app import create_app
from spam_detector.adapters.ml.keyword_model import KeywordSpamModelAdapter
from spam_detector.adapters.ml.tensorflow_model import TensorFlowSpamModelAdapter
from spam_detector.application.use_cases.predict_spam import PredictSpamUseCase


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    runtime: str
    model_path: str
    threshold: float


def load_config() -> RuntimeConfig:
    backend = os.getenv("MODEL_BACKEND", "keyword").strip().lower()
    runtime = os.getenv("MODEL_RUNTIME", "cpu").strip().lower()
    model_path = os.getenv("MODEL_PATH", "artifacts/model.keras")
    threshold = float(os.getenv("SPAM_THRESHOLD", "0.5"))
    return RuntimeConfig(
        backend=backend,
        runtime=runtime,
        model_path=model_path,
        threshold=threshold,
    )


def build_app():
    config = load_config()
    if config.backend == "tensorflow":
        model_adapter = TensorFlowSpamModelAdapter(
            model_path=config.model_path,
            runtime=config.runtime,
        )
        backend_name = f"tensorflow:{config.runtime}"
    else:
        model_adapter = KeywordSpamModelAdapter()
        backend_name = "keyword"

    use_case = PredictSpamUseCase(model_port=model_adapter, threshold=config.threshold)
    return create_app(predict_use_case=use_case, backend_name=backend_name)
