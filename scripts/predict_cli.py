"""CLI utility for local spam prediction."""

from __future__ import annotations

import argparse

from spam_detector.application.use_cases.predict_spam import PredictSpamUseCase
from spam_detector.adapters.ml.keyword_model import KeywordSpamModelAdapter
from spam_detector.adapters.ml.tensorflow_model import TensorFlowSpamModelAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict if a message is spam")
    parser.add_argument("message")
    parser.add_argument("--backend", default="keyword", choices=["keyword", "tensorflow"])
    parser.add_argument("--runtime", default="cpu", choices=["cpu", "nvidia", "metal"])
    parser.add_argument("--model-path", default="artifacts/model.keras")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.backend == "tensorflow":
        model = TensorFlowSpamModelAdapter(model_path=args.model_path, runtime=args.runtime)
    else:
        model = KeywordSpamModelAdapter()

    use_case = PredictSpamUseCase(model_port=model, threshold=args.threshold)
    prediction = use_case.execute(args.message)
    print(f"label={prediction.label} score={prediction.score} is_spam={prediction.is_spam}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
