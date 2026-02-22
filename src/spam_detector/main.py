"""ASGI entrypoint."""

from spam_detector.composition_root.container import build_app

app = build_app()
