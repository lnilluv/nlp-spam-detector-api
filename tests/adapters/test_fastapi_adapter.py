from fastapi.testclient import TestClient

from spam_detector.adapters.api.fastapi_app import create_app


class StubUseCase:
    class Result:
        label = "spam"
        score = 0.88
        is_spam = True

    def execute(self, _message):
        return self.Result()


def test_healthz_endpoint_is_ok():
    app = create_app(predict_use_case=StubUseCase())
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_prediction_payload():
    app = create_app(predict_use_case=StubUseCase())
    client = TestClient(app)

    response = client.post("/predict", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "spam"
    assert body["score"] == 0.88
    assert body["is_spam"] is True
