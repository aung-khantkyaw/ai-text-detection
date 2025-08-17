@echo off
REM Install dependencies and run the Flask app
python -m venv .venv
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
set FLASK_APP=app.py
set FLASK_ENV=development
python app.py
