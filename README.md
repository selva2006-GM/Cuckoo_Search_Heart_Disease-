<<<<<<< HEAD
# Cuckoo_Search_Heart_Disease-
=======
# Heart Disease ML Pipeline — Live Web Dashboard

A real-time ML pipeline dashboard where **every result streams live from Python to the browser**.

---

## Project Structure

```
heart_ml/
├── app.py                          ← Flask server + SSE streaming
├── preprocessing.py                ← Data loading, cleaning, encoding
├── randomforest.py                 ← RF model wrapper
├── feature_selection.py            ← Top-K feature selector
├── cuckoo_search.py                ← Cuckoo Search + Random Search
├── requirements.txt
├── templates/
│   └── index.html                  ← Full live dashboard frontend
└── dataset/
    └── cleaned_data/
        └── heart_combined.csv      ← ← PUT YOUR DATASET HERE
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your dataset

Place your CSV file at:
```
dataset/cleaned_data/heart_combined.csv
```

The file must have a `target` column (0 = no disease, 1 = disease) and the following feature columns:
`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

### 3. Run the server

```bash
python app.py
```

### 4. Open the dashboard

```
http://localhost:5000
```

Click **▶ Run Pipeline** — all data streams live from Python to the browser.

---

## How Live Streaming Works

The backend uses **Server-Sent Events (SSE)**:

- Browser connects to `/stream` (persistent HTTP connection)
- Python emits named events as each pipeline stage completes
- Frontend JavaScript listens for each event and updates charts/metrics in real time
- No page reload needed

### SSE Events

| Event | Fired when |
|-------|-----------|
| `preprocessing` | Dataset loaded & cleaned |
| `split` | Train/test split done |
| `baseline_rf` | Baseline RF evaluated |
| `feature_selection` | Top-K features selected |
| `rf_after_fs` | RF retrained on selected features |
| `rs_progress` | Each Random Search iteration |
| `random_search_done` | Random Search finished |
| `cs_progress` | Each Cuckoo Search iteration |
| `cuckoo_search_done` | Cuckoo Search finished |
| `final_model` | Final model trained & evaluated |
| `pipeline_complete` | All done |

### Prediction API

```
POST /predict
Content-Type: application/json

{ "cp": 3, "thalach": 130, "ca": 1, "oldpeak": 2.3, "thal": 2, "exang": 1 }
```

Returns:
```json
{ "probability": 78.4, "prediction": 1, "features_used": ["cp","thalach","ca","oldpeak","thal","exang"] }
```

---

## Pipeline Stages

1. **Preprocessing** — Load CSV, null check, encode categoricals, normalize continuous, 80/20 split
2. **Baseline RF** — n_estimators=100, max_depth=None, all 13 features → metrics + ROC + feature importances
3. **Feature Selection** — RF gini importances → select top 6 features (retaining ~82% importance)
4. **RF After FS** — Retrain on 6 features → compare before vs after
5. **Optimization** — Random Search (30 iter) then Cuckoo Search (25 nests, 30 iter, Pa=0.25) — live convergence chart
6. **Final Model** — Best params + selected features → full metrics, confusion matrix, all-stage comparison
7. **Predict** — Live Flask API endpoint, returns probability from the trained model
>>>>>>> 53f180f (initial commit)
