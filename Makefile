PYTHON=.venv313/bin/python

.PHONY: test lint run smoke train audit

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

run:
	PYTHONPATH=src MODEL_BACKEND=keyword MODEL_RUNTIME=cpu $(PYTHON) -m uvicorn spam_detector.main:app --reload

smoke:
	$(PYTHON) scripts/smoke_api.py

train:
	PYTHONPATH=src $(PYTHON) scripts/train_tensorflow.py --epochs 10

audit:
	$(PYTHON) -m pip_audit -r requirements/cpu.txt
