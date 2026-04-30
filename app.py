"""
Heart Disease ML Pipeline — Flask Backend with SSE Live Streaming
Run:  python app.py
Open: http://localhost:5000
"""

import json
import time
import threading
import queue
import os

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request, jsonify
from sklearn.model_selection import train_test_split

from preprocessing import Preprocessing
from randomforest import RandomForestModel
from feature_selection import FeatureSelector
from cuckoo_search import CuckooSearch, RandomSearch

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────
# Global state (one pipeline at a time is fine for a demo)
# ─────────────────────────────────────────────────────────────────
pipeline_state = {
    "running": False,
    "X_train": None, "X_test": None,
    "y_train": None, "y_test": None,
    "rf": None, "fs": None,
    "selected_features": None,
    "best_params": None,
    "scaler": None,
}

# SSE queues: each connected client gets its own queue
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()


def sse_broadcast(event: str, data: dict):
    """Push an SSE event to every connected client."""
    msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


def emit(event, data):
    sse_broadcast(event, data)
    time.sleep(0.05)          # tiny delay so the browser can paint


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    """SSE endpoint — each client connects here."""
    q: queue.Queue = queue.Queue(maxsize=200)
    with _sse_lock:
        _sse_clients.append(q)

    def generate():
        # Send a heartbeat immediately so the connection is confirmed
        yield "event: ping\ndata: {}\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield msg
                except queue.Empty:
                    yield "event: ping\ndata: {}\n\n"
        except GeneratorExit:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/run", methods=["POST"])
def run_pipeline():
    """Start the pipeline in a background thread."""
    if pipeline_state["running"]:
        return jsonify({"error": "Pipeline already running"}), 409

    data = request.get_json(silent=True) or {}
    dataset_path = data.get("dataset", "dataset/cleanedData/heart_combined.csv")
    target_col   = data.get("target", "target")

    t = threading.Thread(target=_run_pipeline, args=(dataset_path, target_col), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/predict", methods=["POST"])
def predict():
    """Single-sample prediction using the trained model."""
    if pipeline_state["rf"] is None or pipeline_state["selected_features"] is None:
        return jsonify({"error": "Model not trained yet"}), 400

    body = request.get_json()
    features = pipeline_state["selected_features"]
    try:
        row = pd.DataFrame([[float(body[f]) for f in features]], columns=features)
        prob = pipeline_state["rf"].predict_proba_single(row)
        pred = int(prob >= 0.5)
        return jsonify({
            "probability": round(prob * 100, 2),
            "prediction": pred,
            "features_used": features,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/status")
def status():
    return jsonify({
        "running": pipeline_state["running"],
        "model_ready": pipeline_state["rf"] is not None,
        "selected_features": pipeline_state["selected_features"],
    })


# ─────────────────────────────────────────────────────────────────
# Pipeline logic (runs in background thread)
# ─────────────────────────────────────────────────────────────────
def _run_pipeline(dataset_path: str, target_col: str):
    pipeline_state["running"] = True

    try:
        # ── STEP 1: PREPROCESSING ──────────────────────────────
        emit("stage", {"stage": "preprocessing", "status": "start"})
        time.sleep(0.3)

        prep = Preprocessing(dataset_path)
        df   = prep.df
        s    = prep.stats

        emit("preprocessing", {
            "raw_rows":     s["raw_rows"],
            "raw_cols":     s["raw_cols"],
            "clean_rows":   s["clean_rows"],
            "null_count":   s["null_count"],
            "dup_count":    s["dup_count"],
            "n_features":   s["n_features"],
            "feature_names": s["feature_names"],
        })
        X = df.drop(columns=[target_col])
        y = df[target_col]
        # Train/test split
        # Step 1: limit dataset to 500 samples
        X_small, _, y_small, _ = train_test_split(X, y,train_size=900,stratify=y,random_state=42)

        # Step 2: split that into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_small, y_small,
            test_size=0.2,
            stratify=y_small,
            random_state=42
        )

        class_counts = y.value_counts().to_dict()
        emit("split", {
            "train_size":  len(X_train),
            "test_size":   len(X_test),
            "class_0":     int(class_counts.get(0, 0)),
            "class_1":     int(class_counts.get(1, 0)),
            "feature_names": list(X.columns),
        })

        pipeline_state["X_train"] = X_train
        pipeline_state["X_test"]  = X_test
        pipeline_state["y_train"] = y_train
        pipeline_state["y_test"]  = y_test
        emit("stage", {"stage": "preprocessing", "status": "done"})
        time.sleep(0.4)

        # ── STEP 2: BASELINE RANDOM FOREST ────────────────────
        emit("stage", {"stage": "baseline_rf", "status": "start"})
        rf_baseline = RandomForestModel(n_estimators=100)
        rf_baseline.train(X_train, y_train)
        metrics_baseline = rf_baseline.evaluate(X_test, y_test)
        fi_baseline = rf_baseline.feature_importances(list(X.columns))

        emit("baseline_rf", {
            **metrics_baseline,
            "feature_importances": fi_baseline,
            "n_estimators": 100,
            "max_depth": "None",
        })
        emit("stage", {"stage": "baseline_rf", "status": "done"})
        time.sleep(0.4)

        # ── STEP 3: FEATURE SELECTION ─────────────────────────
        emit("stage", {"stage": "feature_selection", "status": "start"})
        fs = FeatureSelector()
        selected = fs.select_top_k(rf_baseline.model, X_train, top_k=8)

        emit("feature_selection", {
            "selected_features":    selected,
            "all_features":         [f["feature"] for f in fi_baseline],
            "all_importances":      [f["importance"] for f in fi_baseline],
            "cumulative_importance": fs.cumulative_importance,
            "total_retained":       fs.total_importance_retained,
            "dropped_features":     [f for f in list(X.columns) if f not in selected],
        })
        emit("stage", {"stage": "feature_selection", "status": "done"})
        time.sleep(0.4)

        # ── STEP 4: RF AFTER FEATURE SELECTION ───────────────
        emit("stage", {"stage": "rf_after_fs", "status": "start"})
        X_train_fs = X_train[selected]
        X_test_fs  = X_test[selected]

        rf_fs = RandomForestModel(n_estimators=100)
        rf_fs.train(X_train_fs, y_train)
        metrics_fs = rf_fs.evaluate(X_test_fs, y_test)

        emit("rf_after_fs", {
            **metrics_fs,
            "baseline_accuracy": metrics_baseline["accuracy"],
            "delta_accuracy": round(metrics_fs["accuracy"] - metrics_baseline["accuracy"], 2),
        })
        emit("stage", {"stage": "rf_after_fs", "status": "done"})
        time.sleep(0.4)

        # ── STEP 5: RANDOM SEARCH ────────────────────────────
        emit("stage", {"stage": "random_search", "status": "start"})
        rs_history = []

        def rs_cb(info):
            rs_history.append(info["best_score"])
            emit("rs_progress", {
                "iteration":   info["iteration"],
                "best_score":  info["best_score"],
                "best_params": info["best_params"],
                "history":     info["history"],
            })

        rs = RandomSearch(n_iterations=30)
        rs_params, rs_score, _ = rs.optimize(
            None, X_train_fs, X_test_fs, y_train, y_test, progress_cb=rs_cb
        )

        emit("random_search_done", {
            "best_score":  round(rs_score, 4),
            "best_params": {
                "n_estimators": int(rs_params[0]),
                "max_depth":    int(rs_params[1]),
            },
            "history": rs_history,
        })
        emit("stage", {"stage": "random_search", "status": "done"})
        time.sleep(0.4)

        # ── STEP 6: CUCKOO SEARCH ────────────────────────────
        emit("stage", {"stage": "cuckoo_search", "status": "start"})
        cs_history = []

        def cs_cb(info):
            cs_history.append(info["best_score"])
            emit("cs_progress", {
                "iteration":   info["iteration"],
                "best_score":  info["best_score"],
                "best_params": info["best_params"],
                "history":     info["history"],
                "rs_history":  rs_history,
            })

        cs = CuckooSearch(n_nests=25, n_iterations=30, pa=0.25)
        rf_dummy = RandomForestModel()
        cs_params, cs_score, _ = cs.optimize(
            rf_dummy, X_train_fs, X_test_fs, y_train, y_test, progress_cb=cs_cb
        )

        emit("cuckoo_search_done", {
            "best_score":  round(cs_score, 4),
            "best_params": {
                "n_estimators": int(cs_params[0]),
                "max_depth":    int(cs_params[1]),
            },
            "rs_score":   round(rs_score, 4),
            "cs_history": cs_history,
            "rs_history": rs_history,
        })
        emit("stage", {"stage": "cuckoo_search", "status": "done"})
        time.sleep(0.4)

        # ── STEP 7: FINAL MODEL (best params + FS) ──────────
        emit("stage", {"stage": "final_model", "status": "start"})

        # Pick winner
        if cs_score >= rs_score:
            best_params = cs_params
            winner = "Cuckoo Search"
        else:
            best_params = rs_params
            winner = "Random Search"

        n_est  = max(10, min(300, int(best_params[0])))
        max_d  = max(2,  min(20,  int(best_params[1])))

        rf_final = RandomForestModel(n_estimators=n_est, max_depth=max_d)
        rf_final.train(X_train_fs, y_train)
        metrics_final = rf_final.evaluate(X_test_fs, y_test)

        # Save to global state for prediction
        pipeline_state["rf"]               = rf_final
        pipeline_state["fs"]               = fs
        pipeline_state["selected_features"] = selected
        pipeline_state["best_params"]       = {"n_estimators": n_est, "max_depth": max_d}

        emit("final_model", {
            **metrics_final,
            "winner":            winner,
            "n_estimators":      n_est,
            "max_depth":         max_d,
            "baseline_accuracy": metrics_baseline["accuracy"],
            "fs_accuracy":       metrics_fs["accuracy"],
            "rs_accuracy":       round(rs_score, 2),
            "cs_accuracy":       round(cs_score, 2),
            "final_accuracy":    metrics_final["accuracy"],
            "selected_features": selected,
            "stages": [
                {"label": "Baseline RF",     "accuracy": metrics_baseline["accuracy"]},
                {"label": "RF + Feat Sel",   "accuracy": metrics_fs["accuracy"]},
                {"label": "Random Search",   "accuracy": round(rs_score, 2)},
                {"label": "Cuckoo Search",   "accuracy": round(cs_score, 2)},
                {"label": "Final (CS + FS)", "accuracy": metrics_final["accuracy"]},
            ],
        })
        emit("stage", {"stage": "final_model", "status": "done"})
        emit("pipeline_complete", {"message": "Pipeline finished successfully."})

    except Exception as exc:
        import traceback
        emit("error", {"message": str(exc), "trace": traceback.format_exc()})
    finally:
        pipeline_state["running"] = False


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀  Heart ML Pipeline server starting...")
    print("📂  Place your dataset at:  dataset/cleaned_data/heart_combined.csv")
    print("🌐  Open:  http://localhost:5000\n")
    app.run(debug=True, threaded=True, port=5000)
