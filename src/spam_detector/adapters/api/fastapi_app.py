"""FastAPI adapter exposing use cases."""

from __future__ import annotations

from fastapi import FastAPI

from spam_detector.adapters.api.schemas import HealthResponse, PredictRequest, PredictResponse


def create_app(predict_use_case, backend_name: str = "keyword") -> FastAPI:
    app = FastAPI(title="Spam Detector API", version="1.0.0")

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok", backend=backend_name)

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        prediction = predict_use_case.execute(payload.message)
        return PredictResponse(
            label=prediction.label,
            score=prediction.score,
            is_spam=prediction.is_spam,
        )

    return app
