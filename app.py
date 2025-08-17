"""
AI Text Detection API
A Flask web application for detecting AI-generated vs human-written text using BernoulliNB
Part of Laziestant's Projects
"""

from flask import Flask, request, jsonify, render_template
import json
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import re
import string
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
# Configurable upload size limit (default 100 MB). Override with env MAX_CONTENT_MB
MAX_CONTENT_MB = int(os.getenv('MAX_CONTENT_MB', '100'))
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024

# Global variables for model and vectorizer
model = None
vectorizer = None
label_mapping = None
model_info = {
    'trained': False,
    'accuracy': None,
    'training_date': None,
    'dataset_size': None
}

class TextPreprocessor:
    # Minimal contractions map inspired by notebook (trimmed for size)
    CONTRACTIONS = {
        "ain't": "am not", "aren't": "are not", "can't": "cannot",
        "could've": "could have", "couldn't": "could not", "didn't": "did not",
        "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hasn't": "has not", "haven't": "have not", "he'd": "he would",
        "he'll": "he will", "he's": "he is", "how's": "how is",
        "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
        "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
        "let's": "let us", "might've": "might have", "mightn't": "might not",
        "must've": "must have", "mustn't": "must not", "needn't": "need not",
        "shan't": "shall not", "she'd": "she would", "she'll": "she will",
        "she's": "she is", "should've": "should have", "shouldn't": "should not",
        "that's": "that is", "there's": "there is", "they'd": "they would",
        "they'll": "they will", "they're": "they are", "they've": "they have",
        "wasn't": "was not", "we'd": "we would", "we'll": "we will",
        "we're": "we are", "we've": "we have", "weren't": "were not",
        "what's": "what is", "won't": "will not", "wouldn't": "would not",
        "u": "you", "ur": "your", " n ": " and "
    }

    @staticmethod
    def _remove_tags(text: str) -> str:
        # mimic notebook tag cleanup for \n and single quotes
        for tag in ["\n", "'"]:
            text = text.replace(tag, "")
        return text

    @staticmethod
    def _expand_contractions(text: str) -> str:
        t = text
        for k, v in TextPreprocessor.CONTRACTIONS.items():
            t = re.sub(rf"\b{re.escape(k)}\b", v, t)
        return t

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in string.punctuation)

    @staticmethod
    def preprocess(text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = text.lower()
        t = TextPreprocessor._remove_tags(t)
        t = TextPreprocessor._expand_contractions(t)
        t = TextPreprocessor._remove_punctuation(t)
        # collapse whitespace
        t = ' '.join(t.split())
        return t


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


@app.route('/')
def index():
    return render_template('index.html', model_info=model_info)


@app.route('/model/status')
def model_status():
    """Dedicated endpoint for detailed model status information"""
    global model, vectorizer, label_mapping, model_info
    
    # Check if model is loaded
    model_loaded = model is not None and vectorizer is not None
    
    if not model_loaded:
        # Try to load model files to get info
        model_path = os.path.join(MODELS_DIR, 'model.pkl')
        vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                
                mapping_path = os.path.join(MODELS_DIR, 'label_mapping.pkl')
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'rb') as f:
                        label_mapping = pickle.load(f)
                
                model_loaded = True
                model_info['trained'] = True
                
                # Load metadata
                meta_path = os.path.join(MODELS_DIR, 'model_meta.json')
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        for k in ['accuracy', 'training_date', 'dataset_size', 'label_mapping']:
                            if k in meta and meta[k] is not None:
                                model_info[k] = meta[k]
                    except Exception:
                        pass
            except Exception:
                pass
    
    # Build comprehensive status
    status = {
        'model_loaded': model_loaded,
        'model_available': model_loaded,
        'algorithm': 'Logistic Regression' if model_loaded else 'Not loaded',
    }
    
    # Add model info if available, filtering out null values
    for key, value in model_info.items():
        if value is not None:
            if key == 'accuracy' and isinstance(value, (int, float)):
                status[key] = f"{(value * 100):.1f}%" if value <= 1 else f"{value:.1f}%"
            elif key == 'label_mapping' and isinstance(value, dict):
                status['classes'] = list(value.keys())
                status['class_mapping'] = value
            else:
                status[key] = value
    
    # Add file information
    if model_loaded:
        try:
            model_path = os.path.join(MODELS_DIR, 'model.pkl')
            vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
            
            if hasattr(vectorizer, 'get_feature_names_out'):
                status['feature_count'] = len(vectorizer.get_feature_names_out())
            elif hasattr(vectorizer, 'get_feature_names'):
                status['feature_count'] = len(vectorizer.get_feature_names())
            
            # File sizes
            if os.path.exists(model_path):
                status['model_file_size'] = f"{os.path.getsize(model_path) / 1024:.1f} KB"
            if os.path.exists(vectorizer_path):
                status['vectorizer_file_size'] = f"{os.path.getsize(vectorizer_path) / 1024:.1f} KB"
                
        except Exception as e:
            status['info_error'] = str(e)
    
    return jsonify(status)


@app.route('/api/info')
def api_info():
    # Read persisted metadata if available; otherwise build a best-effort snapshot
    def read_meta_or_fallback():
        meta_path = os.path.join(MODELS_DIR, 'model_meta.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        # Fallback if artifacts exist but no meta file
        model_path = os.path.join(MODELS_DIR, 'model.pkl')
        vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
        mapping_path = os.path.join(MODELS_DIR, 'label_mapping.pkl')
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            label_map = None
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, 'rb') as f:
                        label_map = pickle.load(f)
                except Exception:
                    label_map = None
            try:
                ts = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                ts = None
            return {
                'trained': True,
                'accuracy': None,
                'training_date': ts,
                'dataset_size': None,
                'label_mapping': label_map,
                'labels': list(label_map.keys()) if isinstance(label_map, dict) else None,
                'dataset_used': None
            }
        return None

    meta = read_meta_or_fallback()

    # If meta exists but has null accuracy/dataset_size, try to compute once from latest dataset
    def enrich_meta_if_needed(current_meta):
        if not current_meta:
            return None
        needs_accuracy = current_meta.get('accuracy') is None
        needs_size = current_meta.get('dataset_size') is None
        if not (needs_accuracy or needs_size):
            return current_meta

        # Resolve dataset path
        latest_path_file = os.path.join(MODELS_DIR, 'latest_dataset.txt')
        if not os.path.exists(latest_path_file):
            return current_meta
        with open(latest_path_file, 'r') as f:
            dataset_path_raw = f.read().strip()
        candidate_paths = []
        if os.path.isabs(dataset_path_raw):
            candidate_paths.append(os.path.normpath(dataset_path_raw))
        candidate_paths.append(os.path.normpath(os.path.join(BASE_DIR, dataset_path_raw)))
        dataset_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if dataset_path is None:
            return current_meta

        # Load artifacts
        global model, vectorizer, label_mapping
        try:
            if model is None or vectorizer is None:
                with open(os.path.join(MODELS_DIR, 'model.pkl'), 'rb') as f:
                    model = pickle.load(f)
                with open(os.path.join(MODELS_DIR, 'vectorizer.pkl'), 'rb') as f:
                    vectorizer = pickle.load(f)
            if label_mapping is None:
                mapping_path = os.path.join(MODELS_DIR, 'label_mapping.pkl')
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'rb') as f:
                        label_mapping = pickle.load(f)
        except Exception:
            return current_meta

        # Load dataset and compute metrics
        try:
            required_columns = ['text', 'label']
            df = pd.read_csv(dataset_path, dtype=str)
            if not all(col in df.columns for col in required_columns):
                df2 = pd.read_csv(dataset_path, header=None, dtype=str)
                if df2.shape[1] >= 2:
                    text_part = df2.iloc[:, :-1].apply(lambda r: ','.join([str(x) for x in r.values if str(x) != 'nan']).strip(), axis=1)
                    label_part = df2.iloc[:, -1].astype(str).str.strip()
                    df = pd.DataFrame({'text': text_part, 'label': label_part})
                else:
                    return current_meta
            df['text'] = df['text'].apply(TextPreprocessor.preprocess)
            df = df[df['text'].str.strip() != '']
            if len(df) == 0:
                return current_meta

            # Map labels using saved mapping if available
            if isinstance(label_mapping, dict) and label_mapping:
                labels_norm = df['label'].apply(lambda v: str(v).strip().lower())
                y_true = labels_norm.map(label_mapping)
                if y_true.isna().any():
                    # If mapping fails (mismatched labels), skip accuracy
                    y_true = None
            else:
                y_true = None

            X_vec = vectorizer.transform(df['text'])
            y_pred = model.predict(X_vec)
            if y_true is not None:
                current_meta['accuracy'] = float(accuracy_score(y_true, y_pred))
            current_meta['dataset_size'] = int(len(df))

            # Persist updated meta
            try:
                with open(os.path.join(MODELS_DIR, 'model_meta.json'), 'w', encoding='utf-8') as f:
                    json.dump(current_meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        except Exception:
            return current_meta

        return current_meta

    meta = enrich_meta_if_needed(meta)

    # Update in-memory status view using meta if available
    if meta:
        model_info['trained'] = True
        for k in ['accuracy', 'training_date', 'dataset_size', 'label_mapping']:
            if k in meta and meta[k] is not None:
                model_info[k] = meta[k]

    # Filter out null values from model_status
    filtered_model_status = {k: v for k, v in model_info.items() if v is not None}
    
    # Build base info structure
    info = {
        'project_name': 'AI Text Detection API',
        'description': 'Detect AI-generated vs human-written text',
        'author': 'Laziestant',
        'model': 'Logistic Regression',
        'model_status': filtered_model_status
    }
    
    # Only include model_meta if it exists and has content
    if meta and any(v is not None for v in meta.values()):
        filtered_meta = {k: v for k, v in meta.items() if v is not None}
        if filtered_meta:
            info['model_meta'] = filtered_meta
    
    print(f"API Info: {json.dumps(info, indent=2)}")
    return jsonify(info)


@app.route('/upload-csv', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'GET':
        # Render upload page (no visible nav link on home)
        return render_template('upload.html')

    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            required_columns = ['text', 'label']
            # First attempt: read with header row
            df = pd.read_csv(filepath, dtype=str)
            if not all(col in df.columns for col in required_columns):
                # Second attempt: headerless or extra commas; read without header
                df2 = pd.read_csv(filepath, header=None, dtype=str)
                if df2.shape[1] >= 2:
                    # Join all but last columns as text, last as label
                    text_part = df2.iloc[:, :-1].apply(lambda r: ','.join([str(x) for x in r.values if str(x) != 'nan']).strip(), axis=1)
                    label_part = df2.iloc[:, -1].astype(str).str.strip()
                    df = pd.DataFrame({'text': text_part, 'label': label_part})
                else:
                    os.remove(filepath)
                    return jsonify({
                        'error': f'CSV must contain columns: {required_columns} or be a two-column file (text,label). Found shape: {list(df2.shape)}'
                    }), 400

            # Persist latest dataset path (relative path is OK; train resolves it)
            with open(os.path.join(MODELS_DIR, 'latest_dataset.txt'), 'w') as f:
                f.write(os.path.relpath(filepath, BASE_DIR))

            # Ensure preview has expected columns
            preview = df.head(3)[required_columns].to_dict('records')

            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'rows': int(len(df)),
                'columns': list(df.columns),
                'sample_data': preview
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400

    return jsonify({'error': 'Invalid file type. Only CSV files allowed.'}), 400


@app.route('/train', methods=['POST'])
def train_model():
    global model, vectorizer, label_mapping, model_info

    latest_path_file = os.path.join(MODELS_DIR, 'latest_dataset.txt')
    if not os.path.exists(latest_path_file):
        return jsonify({'error': 'No dataset uploaded. Please upload a CSV file first.'}), 400

    with open(latest_path_file, 'r') as f:
        dataset_path_raw = f.read().strip()

    # Resolve dataset path: support relative paths stored in file, and normalize
    candidate_paths = []
    # If path is absolute, normalize it
    if os.path.isabs(dataset_path_raw):
        candidate_paths.append(os.path.normpath(dataset_path_raw))
    # Try as relative to BASE_DIR
    candidate_paths.append(os.path.normpath(os.path.join(BASE_DIR, dataset_path_raw)))

    dataset_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    # If still missing, search uploads/ for newest CSV
    if dataset_path is None:
        try:
            csvs = [os.path.join(UPLOADS_DIR, f) for f in os.listdir(UPLOADS_DIR) if f.lower().endswith('.csv')]
            if csvs:
                csvs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                dataset_path = csvs[0]
        except Exception:
            pass

    # Fallback to bundled sample_data.csv if present
    if dataset_path is None:
        sample_csv = os.path.join(BASE_DIR, 'sample_data.csv')
        if os.path.exists(sample_csv):
            dataset_path = sample_csv

    if dataset_path is None or not os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset file not found. Please re-upload.', 'detail': dataset_path_raw}), 400

    # Read dataset; support headerless CSV and text with commas
    required_columns = ['text', 'label']
    df = pd.read_csv(dataset_path, dtype=str)
    if not all(col in df.columns for col in required_columns):
        df2 = pd.read_csv(dataset_path, header=None, dtype=str)
        if df2.shape[1] >= 2:
            text_part = df2.iloc[:, :-1].apply(lambda r: ','.join([str(x) for x in r.values if str(x) != 'nan']).strip(), axis=1)
            label_part = df2.iloc[:, -1].astype(str).str.strip()
            df = pd.DataFrame({'text': text_part, 'label': label_part})
        elif df2.shape[1] == 2:
            df2.columns = required_columns
            df = df2
        else:
            return jsonify({'error': 'Dataset must have columns text,label or be a two-column CSV.'}), 400
    df['text'] = df['text'].apply(TextPreprocessor.preprocess)
    df = df[df['text'].str.strip() != '']
    if len(df) == 0:
        return jsonify({'error': 'No valid text data found after preprocessing'}), 400

    X = df['text']
    y = df['label']

    # Create binary mapping
    # Normalize labels, support numeric (0/1) or string ('human'/'ai')
    labels_norm = y.apply(lambda v: str(v).strip().lower())
    classes = sorted(labels_norm.unique())
    if len(classes) != 2:
        return jsonify({'error': f'Dataset must have exactly 2 labels. Found: {classes}'}), 400

    # Prefer explicit semantics when possible
    synonyms_human = {'0', 'human', 'human-written', 'human written', 'real'}
    synonyms_ai = {'1', 'ai', 'ai-generated', 'ai generated', 'machine', 'gpt', 'llm'}

    if set(classes).issubset({'0', '1'}):
        mapping = {'0': 0, '1': 1}
    elif 'human' in classes or any(c in synonyms_human for c in classes):
        mapping = {c: (0 if c in synonyms_human else 1) for c in classes}
    elif 'ai' in classes or any(c in synonyms_ai for c in classes):
        mapping = {c: (1 if c in synonyms_ai else 0) for c in classes}
    else:
        # Fallback: deterministic order
        mapping = {classes[0]: 0, classes[1]: 1}

    label_mapping = mapping
    y_binary = labels_norm.apply(lambda v: mapping.get(v))
    
    # Debug: Print class distribution
    class_distribution = y_binary.value_counts()
    print(f"Class distribution: {dict(class_distribution)}")
    print(f"Label mapping: {mapping}")
    print(f"Sample labels before mapping: {list(labels_norm.unique())}")
    print(f"Sample labels after mapping: {list(y_binary.unique())}")

    # Handle tiny datasets robustly
    class_counts = y_binary.value_counts()
    min_per_class = int(class_counts.min())
    small_dataset = len(df) < 10 or min_per_class < 2

    if small_dataset:
        # Train on all data; relax vectorizer thresholds for small corpora
        vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), stop_words='english', 
            min_df=1, max_df=0.95, sublinear_tf=True, norm='l1'
        )
        X_all_vec = vectorizer.fit_transform(X)
        # Logistic Regression with balanced class weights
        model = LogisticRegression(max_iter=200, class_weight='balanced', solver='liblinear')
        model.fit(X_all_vec, y_binary)
        # Report training accuracy as a proxy for tiny datasets
        accuracy = float(model.score(X_all_vec, y_binary))
        X_train, X_test, y_train, y_test = X, pd.Series([], dtype=X.dtype), y_binary, pd.Series([], dtype=y_binary.dtype)
    else:
        # Normal stratified split
        # Choose a safe test_size for small datasets to satisfy stratify requirements
        counts = y_binary.value_counts()
        min_count = counts.min()
        n_samples = len(y_binary)
        if n_samples < 2:
            return jsonify({'error': 'Dataset too small to train. Need at least 2 samples.'}), 400
        test_size = 0.2
        # Ensure at least 1 sample per class in test split if possible
        if n_samples < 10 or min_count < 3:
            # Keep within sensible bounds
            test_size = max(0.2, min(0.5, 1.0 / max(1, min_count)))
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
            )
        except ValueError:
            # Fallback: no stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=test_size, random_state=42, stratify=None
            )
        vectorizer = TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2), stop_words='english', 
            min_df=2, max_df=0.85, sublinear_tf=True, norm='l1'
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        # Logistic Regression with balanced class weights per notebook alignment
        model = LogisticRegression(max_iter=300, class_weight='balanced', solver='liblinear')
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)

    # Save artifacts under models/
    with open(os.path.join(MODELS_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, 'label_mapping.pkl'), 'wb') as f:
        pickle.dump(label_mapping, f)

    # Persist model metadata
    dataset_rel = os.path.relpath(dataset_path, BASE_DIR)
    model_info.update({
        'trained': True,
        'accuracy': round(accuracy, 4),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_size': len(df),
        'label_mapping': label_mapping
    })

    meta = {
        'trained': True,
        'accuracy': model_info['accuracy'],
        'training_date': model_info['training_date'],
        'dataset_size': model_info['dataset_size'],
        'label_mapping': label_mapping,
        'labels': list(mapping.keys()),
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'dataset_used': dataset_rel,
        'vectorizer': {
            'type': 'TfidfVectorizer',
            'ngram_range': (1, 2),
            'max_features': 8000,
            'norm': 'l1'
        },
        'model': 'LogisticRegression',
        'model_params': {
            'class_weight': 'balanced',
            'solver': 'liblinear'
        }
    }
    try:
        with open(os.path.join(MODELS_DIR, 'model_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return jsonify({
        'message': 'Model trained successfully',
        'accuracy': accuracy,
        'dataset_size': len(df),
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'labels': list(mapping.keys()),
        'dataset_used': os.path.relpath(dataset_path, BASE_DIR)
    })


@app.route('/predict', methods=['POST'])
def predict_text():
    global model, vectorizer, label_mapping

    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    if not text or not str(text).strip():
        return jsonify({'error': 'No text provided'}), 400

    # Load artifacts if not in memory
    if model is None or vectorizer is None or label_mapping is None:
        model_path = os.path.join(MODELS_DIR, 'model.pkl')
        vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
        mapping_path = os.path.join(MODELS_DIR, 'label_mapping.pkl')
        if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
            return jsonify({'error': 'Model not trained. Please train the model first.'}), 400
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                label_mapping = pickle.load(f)
        else:
            label_mapping = {0: 'human', 1: 'ai'}

    cleaned_text = TextPreprocessor.preprocess(text)
    text_vec = vectorizer.transform([cleaned_text])

    pred = model.predict(text_vec)[0]
    # Some classifiers may not have predict_proba; LogisticRegression does
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_vec)[0]
        confidence = float(max(proba))
        prob_human, prob_ai = float(proba[0]), float(proba[1])
    else:
        # Fallback: derive pseudo-proba from decision function if available
        if hasattr(model, 'decision_function'):
            score = float(model.decision_function(text_vec)[0])
            # map score to [0,1] via logistic
            import math
            p_ai = 1 / (1 + math.exp(-score))
            prob_ai = p_ai
            prob_human = 1 - p_ai
            confidence = float(max(prob_ai, prob_human))
        else:
            prob_ai = 1.0 if int(pred) == 1 else 0.0
            prob_human = 1.0 - prob_ai
            confidence = 1.0

    # Build reverse mapping for label
    # Always return human/ai for clarity
    predicted_label = 'ai' if int(pred) == 1 else 'human'

    result = {
        'text': text,
        'prediction': predicted_label,
        'confidence': round(confidence, 4),
        'probabilities': {
            'human': round(float(prob_human), 4),
            'ai': round(float(prob_ai), 4)
        },
        'text_length': len(text),
        'word_count': len(text.split())
    }

    return jsonify(result)


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(413)
def file_too_large(error):
    limit_mb = int(app.config.get('MAX_CONTENT_LENGTH', 0) / (1024 * 1024))
    return jsonify({'error': 'File too large', 'limit_mb': limit_mb}), 413


if __name__ == '__main__':
    # Best-effort load model
    try:
        model_path = os.path.join(MODELS_DIR, 'model.pkl')
        vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            mapping_path = os.path.join(MODELS_DIR, 'label_mapping.pkl')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    label_mapping = pickle.load(f)
            model_info['trained'] = True
            # Load persisted metadata to populate accuracy and dates
            meta_path = os.path.join(MODELS_DIR, 'model_meta.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    for k in ['accuracy', 'training_date', 'dataset_size', 'label_mapping']:
                        if k in meta and meta[k] is not None:
                            model_info[k] = meta[k]
                    print(f'Existing model loaded successfully! Accuracy: {model_info.get("accuracy", "N/A")}, Dataset Size: {model_info.get("dataset_size", "N/A")}')
                except Exception as e:
                    print(f'Model loaded but metadata failed: {e}')
            else:
                print('Existing model loaded successfully! (No metadata file found)')
                # Try to get basic info from file timestamps
                try:
                    ts = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
                    model_info['training_date'] = ts
                except Exception:
                    pass
    except Exception as e:
        print(f'Could not load existing model: {e}')

    print("AI Text Detection API - Part of Laziestant's Projects")
    print("Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
