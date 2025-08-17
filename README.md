# Laziestant • AI Text Detection API

A simple Flask API + web UI to classify text as AI-generated or human-written using TF-IDF + BernoulliNB.

## Features

- Upload CSV dataset (columns: `text`, `label`)
- Train a Bernoulli Naive Bayes model
- Predict text via web UI or JSON API
- Stores model artifacts in `models/` directory
- Tailwind CSS UI

## Endpoints

- `GET /` – Home + prediction UI
- `GET /api/info` – Project + model info
- `GET|POST /upload-csv` – Upload CSV dataset
- `POST /train` – Train on latest uploaded CSV
- `POST /predict` – Predict label for input text

## Quickstart

1. Create a virtual environment and install deps
2. Run the app
3. Open <http://localhost:5000>

## CSV Format

- Required columns: `text`, `label`
- Labels should include two classes (e.g., `human`, `ai`)

## Notes

- Artifacts saved to `models/`: `model.pkl`, `vectorizer.pkl`, `label_mapping.pkl`
- Uploads are stored under `uploads/`
