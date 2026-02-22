"""TensorFlow model adapter for CPU, NVIDIA CUDA, and Apple Metal."""

from __future__ import annotations

import os
from pathlib import Path


class TensorFlowSpamModelAdapter:
    def __init__(self, model_path: str, runtime: str = "cpu") -> None:
        self._runtime = runtime
        self._model_path = Path(model_path)
        self._tf = self._load_tensorflow()
        self._configure_runtime()
        self._model = self._load_model()

    def _load_tensorflow(self):
        try:
            import tensorflow as tf
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "TensorFlow is not installed. Use MODEL_BACKEND=keyword or install runtime dependencies."
            ) from exc
        return tf

    def _configure_runtime(self) -> None:
        tf = self._tf
        if self._runtime == "nvidia":
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                raise RuntimeError("NVIDIA runtime selected but no GPU device was detected.")
        if self._runtime == "metal":
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    def _load_model(self):
        if not self._model_path.exists():
            raise RuntimeError(f"Model file not found: {self._model_path}")
        return self._tf.keras.models.load_model(self._model_path)

    def predict_proba(self, messages: list[str]) -> list[float]:
        predictions = self._model.predict(messages, verbose=0)
        return [float(value[0]) for value in predictions]
