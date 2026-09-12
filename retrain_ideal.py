from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / 'data.csv'
RISK_MODEL_FILE = ROOT / 'sif_risk_model.pkl'
PRECURSOR_MODEL_FILE = ROOT / 'precursor_model.pkl'
SIM_MODEL_FILE = ROOT / 'similarity_vectorizer.pkl'
META_FILE = ROOT / 'model_metadata.json'

REQUIRED = ['Report','Risk','Precursor','Data Status']

def build_text(df):
    cols = ['Report','Category','Report Type','Location','Department']
    return df[cols].fillna('').astype(str).agg(' | '.join, axis=1).str.lower().str.strip()

def make_pipeline(min_df=1, max_features=60000):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            sublinear_tf=True,
            ngram_range=(1,2),
            min_df=min_df,
            max_df=0.98,
            max_features=max_features,
            token_pattern=r'(?u)\b\w[\w/-]*\b',
        )),
        ('clf', LogisticRegression(
            max_iter=4000,
            class_weight='balanced',
            C=2.0,
            solver='lbfgs',
            random_state=26165,
        )),
    ])

def evaluate(y_true, y_pred):
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision_weighted': round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'recall_weighted': round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'f1_weighted': round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
    }

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    df = df[df['Report'].fillna('').astype(str).str.len() > 10].copy()
    df['Risk'] = df['Risk'].astype(str).str.upper().str.strip()
    df['Precursor'] = df['Precursor'].astype(str).str.strip()
    # Train on all labelled source-grounded rows. The actual-public row remains in the corpus.
    # Production deployment should additionally use an OIL-authorized holdout set.
    X = build_text(df)
    y_risk = df['Risk']
    y_prec = df['Precursor']

    Xtr, Xte, yrtr, yrte = train_test_split(X, y_risk, test_size=0.20, random_state=26165, stratify=y_risk)
    risk_model = make_pipeline()
    risk_model.fit(Xtr, yrtr)
    risk_pred = risk_model.predict(Xte)
    risk_metrics = evaluate(yrte, risk_pred)

    Xtrp, Xtep, yptr, ypte = train_test_split(X, y_prec, test_size=0.20, random_state=26165, stratify=y_prec)
    precursor_model = make_pipeline()
    precursor_model.fit(Xtrp, yptr)
    precursor_pred = precursor_model.predict(Xtep)
    precursor_metrics = evaluate(ypte, precursor_pred)

    # Similarity index is deliberately separate from the classifier.
    sim = TfidfVectorizer(lowercase=True, strip_accents='unicode', sublinear_tf=True, ngram_range=(1,2), min_df=1)
    sim.fit(X)

    joblib.dump(risk_model, RISK_MODEL_FILE)
    joblib.dump(precursor_model, PRECURSOR_MODEL_FILE)
    joblib.dump(sim, SIM_MODEL_FILE)

    meta = {
        'dataset_rows': int(len(df)),
        'precursor_classes': sorted(y_prec.unique().tolist()),
        'risk_classes': sorted(y_risk.unique().tolist()),
        'risk_metrics_holdout': risk_metrics,
        'precursor_metrics_holdout': precursor_metrics,
        'feature_text': ['Report','Category','Report Type','Location','Department'],
        'model': 'TF-IDF (1-2 grams, sublinear TF) + balanced Logistic Regression',
        'seed': 26165,
        'warning': 'Holdout metrics are not a substitute for evaluation on unseen OIL-authorized reports; source-grounded/augmented rows may share templates and terminology.',
    }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print('IDEAL OIL SIF ENGINE RETRAINING COMPLETE')
    print(f'Dataset: {len(df)} rows')
    print(f'Precursor classes: {len(y_prec.unique())}')
    print('Risk:', risk_metrics)
    print('Precursor:', precursor_metrics)
    print(f'Saved: {RISK_MODEL_FILE.name}, {PRECURSOR_MODEL_FILE.name}, {SIM_MODEL_FILE.name}, {META_FILE.name}')
    print('\nPrecursor class distribution:')
    print(y_prec.value_counts().to_string())

if __name__ == '__main__':
    main()
