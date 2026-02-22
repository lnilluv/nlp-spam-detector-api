"""Simple smoke test against running API service."""

from __future__ import annotations

import argparse

import httpx


def run(base_url: str) -> int:
    timeout = httpx.Timeout(10.0)
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        health = client.get("/healthz")
        if health.status_code != 200:
            print(f"Health check failed: {health.status_code}")
            return 1

        prediction = client.post("/predict", json={"message": "free entry claim cash now"})
        if prediction.status_code != 200:
            print(f"Predict call failed: {prediction.status_code}")
            return 1

        payload = prediction.json()
        if "label" not in payload or "score" not in payload:
            print("Predict payload missing fields")
            return 1
        print("Smoke test passed")
        print(payload)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API smoke checks")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args.base_url))
