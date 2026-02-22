# Deep Learning NLP Spam Detector

Production-ready SMS spam detection project built from a deep learning notebook and hardened for portfolio demonstrations, local macOS development (Metal), and VPS deployment (CPU now, NVIDIA-ready later).

## What this project does

- Classifies SMS text messages as `ham` or `spam`.
- Provides a deployable API (`/healthz`, `/predict`) via FastAPI.
- Supports runtime profiles:
  - `cpu` (VPS-safe default)
  - `nvidia` (CUDA-ready profile)
  - `metal` (Apple Silicon local profile)
- Keeps a deep learning path (TensorFlow training + exported model artifacts).
- Includes a fallback keyword adapter for deterministic smoke testing and resilient startup.

## Notebook contents (`main.ipynb`)

The notebook is the original deep learning experiment and contains:

1. **Data loading and EDA**
   - Loads the AT&T SMS dataset from public S3.
   - Inspects null values and class distribution.

2. **Text preprocessing**
   - Cleans messages (lowercasing, character filtering).
   - Removes stop words and lemmatizes text.

3. **Tokenization and sequence preparation**
   - Builds a tokenizer and encodes text.
   - Pads sequences to fixed length for neural input.

4. **Deep learning model training**
   - Embedding + global pooling + dense layers.
   - Early stopping and TensorBoard logging.

5. **Evaluation and visualization**
   - Validation loss/accuracy curves.
   - Confusion matrix for ham/spam quality.

This repository keeps that notebook as the experiment artifact and adds production-oriented code around it.

## Architecture (Hexagonal)

- `src/spam_detector/domain/`
  - Pure business logic (text normalization, prediction entity).
- `src/spam_detector/application/`
  - Use cases and ports (model inference contract).
- `src/spam_detector/adapters/`
  - API adapter (FastAPI), ML adapters (TensorFlow and keyword fallback).
- `src/spam_detector/composition_root/`
  - Runtime wiring (`MODEL_BACKEND`, `MODEL_RUNTIME`, `MODEL_PATH`).

Dependency flow: `adapters -> application -> domain`.

## Repository structure

```text
src/spam_detector/
  domain/
  application/
  adapters/
  composition_root/
scripts/
  train_tensorflow.py
  predict_cli.py
  smoke_api.py
requirements/
  base.txt
  cpu.txt
  nvidia.txt
  metal.txt
  dev.txt
tests/
Dockerfile.cpu
Dockerfile.nvidia
docker-compose.yml
```

## Quickstart

### 1) Local dev (Python 3.13)

```bash
python3.13 -m venv .venv313
.venv313/bin/python -m pip install -r requirements/dev.txt
```

### 2) Run tests

```bash
.venv313/bin/python -m pytest
```

### 3) Run API locally (keyword fallback)

```bash
PYTHONPATH=src MODEL_BACKEND=keyword MODEL_RUNTIME=cpu .venv313/bin/python -m uvicorn spam_detector.main:app
```

### 4) Smoke test API

```bash
.venv313/bin/python scripts/smoke_api.py --base-url http://127.0.0.1:8000
```

## Deep learning training path

Train and export TensorFlow model artifacts:

```bash
PYTHONPATH=src .venv313/bin/python scripts/train_tensorflow.py --epochs 10 --output-dir artifacts
```

Then run inference with TensorFlow backend:

```bash
PYTHONPATH=src MODEL_BACKEND=tensorflow MODEL_RUNTIME=cpu MODEL_PATH=artifacts/model.keras .venv313/bin/python -m uvicorn spam_detector.main:app
```

## Docker deployment

### CPU profile (recommended for current VPS)

```bash
docker compose --profile cpu up --build -d
python3 scripts/smoke_api.py --base-url http://127.0.0.1:8000
```

### NVIDIA profile (for future GPU hosts)

```bash
docker compose --profile nvidia up --build -d
python3 scripts/smoke_api.py --base-url http://127.0.0.1:8001
```

### Metal note

Apple Metal acceleration is available in local macOS Python environments (`requirements/metal.txt`) and is not a Linux Docker runtime feature.

## Security posture improvements

- Environment-based configuration via `.env.example`.
- Local-only security notes/checklists kept under ignored `.local/`.
- Generated logs and artifacts ignored by git.
- Non-root Docker users and health checks.
- Dependency audit command included (`pip-audit`).

## Commands

- `make test` - run unit tests.
- `make lint` - run lint checks.
- `make run` - start local API.
- `make smoke` - run API smoke tests.
- `make train` - train and export model.
- `make audit` - run vulnerability audit.

## Portfolio highlights

- End-to-end NLP spam detection project from notebook experiment to deployable service.
- Deep learning model workflow with reproducible training and artifact export.
- Clean hexagonal architecture separation for maintainability.
- Multi-runtime strategy for real-world deployment constraints (CPU now, GPU-ready, Metal-ready).
