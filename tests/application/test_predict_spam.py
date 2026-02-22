from spam_detector.application.use_cases.predict_spam import PredictSpamUseCase


class StubModel:
    def predict_proba(self, messages):
        assert messages == ["free entry now"]
        return [0.91]


def test_predict_spam_use_case_returns_thresholded_response():
    use_case = PredictSpamUseCase(model_port=StubModel(), threshold=0.5)

    result = use_case.execute("free entry now")

    assert result.label == "spam"
    assert result.is_spam is True
    assert result.score == 0.91
