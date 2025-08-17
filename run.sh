#!/usr/bin/env bash
# Install dependencies and run the Flask app
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export FLASK_APP=app.py
export FLASK_ENV=development
python app.py
