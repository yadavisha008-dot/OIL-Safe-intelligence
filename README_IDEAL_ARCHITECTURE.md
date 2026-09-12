# OIL SIF Intelligence Engine — Ideal Architecture

## 1. Data layer
- `data.csv` — 5,000 source-grounded records.
- Each row carries `Data Status` and `Source / Provenance`.
- `ACTUAL_PUBLICLY_REPORTED` means directly public OIL incident evidence.
- `DERIVED_FROM_OIL_PUBLIC_DOCUMENT` means curated from OIL public safety documents.
- `SOURCE_GROUNDED_AUGMENTED` means a training instance constructed from OIL-published hazard/control terminology; it is **not** an actual confidential OIL report.

## 2. Offline ML training
Run:

```bash
python retrain_ideal.py
```

It creates:
- `sif_risk_model.pkl`
- `precursor_model.pkl`
- `similarity_vectorizer.pkl`
- `model_metadata.json`

Models:
- TF-IDF word n-grams (1–2), sublinear TF
- balanced Logistic Regression
- risk classifier
- 16-class SIF precursor classifier

## 3. Runtime inference
`app_ideal.py` loads the saved models; it does not retrain every time Streamlit starts.

Pipeline:

Safety Report
→ text normalization / context fields
→ risk model
→ SIF precursor model
→ evidence extraction
→ SIF prioritisation score
→ safety-barrier screening
→ recommended controls
→ similar-report retrieval
→ HSE review queue

## 4. Production data path
When OIL provides authorized historical records, ingest them as the primary supervised dataset and maintain a separate untouched validation/test set. Do not mix confidential actual reports with synthetic/augmented records without provenance labels.

## 5. Run

```bash
cd C:\Users\<your-user>\Downloads\OIL-SIF-ENGINE
python retrain_ideal.py
streamlit run app_ideal.py
```

## 6. SIH/production positioning
Do not claim the current 5,000 rows are 5,000 actual OIL reports. Public OIL disclosures provide aggregate near-miss counts, selected incidents, HSE procedures, audits and safety controls; the complete internal report database is not publicly available.
