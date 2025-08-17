# Laziestant • AI Text Detection API

Flask API + web UI to classify text as AI‑generated or human‑written using TF‑IDF and Logistic Regression. Includes CSV upload, training, prediction, and a model status panel with charts.

## What it does

- Upload a CSV dataset and train a text classifier
- Predict per‑text probabilities (AI vs Human) via web UI or JSON
- Persist model artifacts and metadata to `models/`
- Visualize model accuracy (pie chart) and data split (radar chart) in the UI

## Tech stack

- Backend: Flask, scikit‑learn
- Model: LogisticRegression (liblinear, class_weight="balanced")
- Vectorizer: TfidfVectorizer (norm='l1', 1–2 grams, sublinear_tf, stop_words='english')
- Preprocessing: lowercasing, newline/quote cleanup, contraction expansion, punctuation removal, whitespace collapse
- Frontend: Tailwind CSS, Chart.js

## Endpoints

- GET `/` – Web UI (analyzer + model status)
- GET `/api/info` – Project + model info and metadata
- GET `/model/status` – Detailed model status (features, file sizes)
- GET|POST `/upload-csv` – Upload CSV dataset
- POST `/train` – Train on the latest uploaded CSV
- POST `/predict` – Predict label and probabilities for input text

## Quickstart

1. Install dependencies

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
```

2. Run the app

```bash
py app.py
# then open http://localhost:5000
```

Optional: Windows `run.bat` or Unix `run.sh` are included.

## Dataset format

- Preferred: CSV with columns `text`, `label`
- Supported labels: `human`/`ai`, or `0`/`1` (common synonyms are normalized)
- Headerless CSVs are supported: last column is the label, all prior columns are joined as text

Example rows:

```csv
text,label
This essay discusses...,human
AI systems often...,ai
```

## Train and predict

After uploading a CSV (via UI or `/upload-csv`), trigger a training run:

```bash
curl -X POST http://127.0.0.1:5000/train
```

Make a prediction:

```bash
curl -X POST http://127.0.0.1:5000/predict \
	-H "Content-Type: application/json" \
	-d '{"text":"Write a short paragraph about the solar system."}'
```

Response (abridged):

```json
{
	"prediction": "human",
	"confidence": 0.51,
	"probabilities": { "human": 0.51, "ai": 0.49 }
}
```

## UI and charts

- Model Status shows: Status, Algorithm, Health (High/Medium/Low), Accuracy, Dataset Size, Last Trained
- Charts:
	- Accuracy: pie chart (Correct vs Other)
	- Data Split: radar chart (Total / Train / Test)

## Artifacts and metadata

Saved under `models/`:

- `model.pkl` – trained Logistic Regression
- `vectorizer.pkl` – TF‑IDF vectorizer
- `label_mapping.pkl` – normalized label→class mapping
- `model_meta.json` – metadata: accuracy, dataset size, split counts, vectorizer config, model params, dataset_used
- `latest_dataset.txt` – path to the most recent dataset

Uploads are stored under `uploads/`.

## Configuration

- `MAX_CONTENT_MB` (env): max upload size in MB (default 100)

## Troubleshooting

- 400 “Dataset must have exactly 2 labels”: ensure only two classes exist in `label`
- 413 “File too large”: reduce file size or increase `MAX_CONTENT_MB`
- Headerless CSVs: ensure the last column is the label; others are text parts

## Notes

- The UI “confidence” is per‑text; the Model Status “accuracy” is overall test accuracy.
