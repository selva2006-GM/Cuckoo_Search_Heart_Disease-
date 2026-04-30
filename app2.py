"""
Heart Disease ML Pipeline — CLI Version
Usage:
    python cli_pipeline.py
    python cli_pipeline.py --dataset dataset/cleanedData/heart_combined.csv --target target
    python cli_pipeline.py --predict   # interactive prediction after training
"""

import argparse
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocessing import Preprocessing
from randomforest import RandomForestModel
from feature_selection import FeatureSelector
from cuckoo_search import CuckooSearch, RandomSearch


# ─────────────────────────────────────────────────────────────────
# Simple output helpers
# ─────────────────────────────────────────────────────────────────

def header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def step(label: str):
    print(f"\n> {label}")

def kv(key: str, value, indent: int = 4):
    pad = " " * indent
    print(f"{pad}{key}: {value}")

def ok(msg: str):
    print(f"  [OK] {msg}")

def warn(msg: str):
    print(f"  [WARN] {msg}")

def err(msg: str):
    print(f"  [ERROR] {msg}", file=sys.stderr)

def progress_bar(current: int, total: int, width: int = 30, prefix: str = ""):
    filled = int(width * current / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    pct = f"{100 * current / max(total, 1):5.1f}%"
    print(f"\r  {prefix}[{bar}] {pct}  {current}/{total}", end="", flush=True)

def metrics_table(metrics: dict, extras: dict | None = None):
    rows = {
        "Accuracy":  f"{metrics.get('accuracy', 0):.4f}",
        "Precision": f"{metrics.get('precision', 0):.4f}",
        "Recall":    f"{metrics.get('recall', 0):.4f}",
        "F1 Score":  f"{metrics.get('f1', 0):.4f}",
        "AUC-ROC":   f"{metrics.get('roc_auc', metrics.get('auc', 0)):.4f}",
    }
    if extras:
        rows.update(extras)
    col = max(len(k) for k in rows) + 2
    for k, v in rows.items():
        print(f"    {k:<{col}} {v}")

def section_done():
    print(f"  {'-' * 56}")


# ─────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────

def run_pipeline(dataset_path: str, target_col: str) -> dict:
    state = {}

    # ── STEP 1: PREPROCESSING ────────────────────────────────────
    header("STEP 1 - DATA PREPROCESSING")
    step("Loading and cleaning dataset ...")

    prep = Preprocessing(dataset_path)
    df   = prep.df
    s    = prep.stats

    kv("Raw shape",   f"{s['raw_rows']} rows x {s['raw_cols']} cols")
    kv("Clean rows",  s["clean_rows"])
    kv("Nulls found", s["null_count"])
    kv("Duplicates",  s["dup_count"])
    kv("Features",    s["n_features"])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X, _, y, _ = train_test_split(
        X, y, train_size=500, stratify=y, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class_counts = y.value_counts().to_dict()
    kv("Train size",  len(X_train))
    kv("Test size",   len(X_test))
    kv("Class 0",     int(class_counts.get(0, 0)))
    kv("Class 1",     int(class_counts.get(1, 0)))
    ok("Preprocessing complete")
    section_done()

    state.update(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=list(X.columns),
    )

    # ── STEP 2: BASELINE RANDOM FOREST ───────────────────────────
    header("STEP 2 - BASELINE RANDOM FOREST")
    step("Training baseline RF (550 estimators) ...")

    rf_baseline = RandomForestModel(n_estimators=550, max_depth=None)
    rf_baseline.train(X_train, y_train)
    metrics_b = rf_baseline.evaluate(X_test, y_test)
    fi = rf_baseline.feature_importances(list(X.columns))


    metrics_table(metrics_b)
    print()
    step("Top feature importances:")
    for f in fi[:8]:
        print(f"    {f['feature']:<22} {f['importance']:.4f}")

    ok("Baseline model trained")
    section_done()

    # ── STEP 3: FEATURE SELECTION ─────────────────────────────────
    header("STEP 3 - FEATURE SELECTION")
    step("Selecting top-11    features by importance ...")

    fs = FeatureSelector()
    selected = fs.select_top_k(rf_baseline.model, X_train, top_k=11)
    dropped  = [f for f in list(X.columns) if f not in selected]

    kv("Selected features",   ", ".join(selected))
    kv("Dropped features",    ", ".join(dropped) if dropped else "none")
    kv("Importance retained", f"{fs.total_importance_retained:.2%}")
    ok("Feature selection complete")
    section_done()

    state["selected_features"] = selected
    X_train_fs = X_train[selected]
    X_test_fs  = X_test[selected]

    # ── STEP 4: RF AFTER FEATURE SELECTION ────────────────────────
    header("STEP 4 - RF WITH SELECTED FEATURES")
    step("Retraining RF on selected features ...")

    rf_fs = RandomForestModel(n_estimators=550)
    rf_fs.train(X_train_fs, y_train)
    metrics_fs = rf_fs.evaluate(X_test_fs, y_test)

    delta = metrics_fs["accuracy"] - metrics_b["accuracy"]
    sign  = "+" if delta >= 0 else ""
    metrics_table(metrics_fs, extras={"Delta vs Baseline": f"{sign}{delta:.4f}"})
    ok("RF + feature selection evaluated")
    section_done()

    # ── STEP 5: RANDOM SEARCH ─────────────────────────────────────
    header("STEP 5 - RANDOM SEARCH HYPERPARAMETER TUNING")
    step("Running 50 random-search iterations (5-fold CV) ...")

    rs_history: list[float] = []

    def rs_cb(info):
        rs_history.append(info["best_score"])
        progress_bar(info["iteration"], 50, prefix="  RS ")

    rs = RandomSearch(n_iterations=50, cv_folds=5)
    rs_params, rs_score, _ = rs.optimize(
        None, X_train_fs, X_test_fs, y_train, y_test, progress_cb=rs_cb
    )
    print()

    kv("Best score (test acc)", f"{rs_score:.4f}")
    kv("n_estimators",          int(rs_params[0]))
    kv("max_depth",             int(rs_params[1]))
    kv("min_samples_split",     int(rs_params[2]))
    kv("min_samples_leaf",      int(rs_params[3]))
    ok("Random search done")
    section_done()

    # ── STEP 6: CUCKOO SEARCH ─────────────────────────────────────
    header("STEP 6 - CUCKOO SEARCH HYPERPARAMETER TUNING")
    step("Running Cuckoo Search (25 nests, 5 iterations, 3-fold CV) ...")

    cs_history: list[float] = []

    def cs_cb(info):
        cs_history.append(info["best_score"])
        progress_bar(info["iteration"], 10, prefix="  CS ")

    cs = CuckooSearch(n_nests=25, n_iterations=10, pa=0.15, cv_folds=3)
    rf_dummy = RandomForestModel()
    cs_params, cs_score, _ = cs.optimize(
        rf_dummy, X_train_fs, X_test_fs, y_train, y_test, progress_cb=cs_cb
    )
    print()

    kv("Best score (test acc)", f"{cs_score:.4f}")
    kv("n_estimators",          int(round(cs_params[0])))
    kv("max_depth",             int(round(cs_params[1])))
    kv("min_samples_split",     int(round(cs_params[2])))
    kv("min_samples_leaf",      int(round(cs_params[3])))
    ok("Cuckoo search done")
    section_done()

    # ── STEP 7: FINAL MODEL ───────────────────────────────────────
    header("STEP 7 - FINAL MODEL")

    if cs_score >= rs_score:
        best_params = cs_params
        winner = "Cuckoo Search"
    else:
        best_params = rs_params
        winner = "Random Search"

    n_est     = max(10,  min(600, int(round(best_params[0]))))  
    max_d     = max(3,   min(30,  int(round(best_params[1]))))
    min_split = max(2,   min(20,  int(round(best_params[2]))))
    min_leaf  = max(1,   min(10,  int(round(best_params[3]))))

    step(f"Winner: {winner}")
    kv("n_estimators",      n_est)
    kv("max_depth",         max_d)
    kv("min_samples_split", min_split)
    kv("min_samples_leaf",  min_leaf)

    final_model = RandomForestClassifier(
        n_estimators     = n_est,
        max_depth        = max_d,
        min_samples_split= min_split,
        min_samples_leaf = min_leaf,
        random_state     = 42,
        n_jobs           = -1,
    )
    final_model.fit(X_train_fs, y_train)

    # Wrap in RandomForestModel shell to reuse evaluate()
    rf_final = RandomForestModel(n_estimators=n_est, max_depth=max_d)
    rf_final.model = final_model
    metrics_final = rf_final.evaluate(X_test_fs, y_test)

    print()
    metrics_table(metrics_final)
    print()

    step("Accuracy progression:")
    stages = [
        ("Baseline RF",       metrics_b["accuracy"]),
        ("RF + Feat Sel",     metrics_fs["accuracy"]),
        ("Random Search",     rs_score),
        ("Cuckoo Search",     cs_score),
        ("Final (best + FS)", metrics_final["accuracy"]),
    ]
    for label, acc in stages:
        print(f"    {label:<22} {acc:.4f}")

    ok("Pipeline complete!")
    section_done()

    state.update(
        rf=rf_final, fs=fs,
        best_params={
            "n_estimators"     : n_est,
            "max_depth"        : max_d,
            "min_samples_split": min_split,
            "min_samples_leaf" : min_leaf,
        },
        metrics_final=metrics_final,
    )
    return state


# ─────────────────────────────────────────────────────────────────
# Interactive prediction
# ─────────────────────────────────────────────────────────────────

def interactive_predict(state: dict):
    rf       = state["rf"]
    features = state["selected_features"]

    header("PREDICTION - Interactive Mode")
    print(f"  Enter values for: {', '.join(features)}")
    print(f"  (type q to quit)\n")

    while True:
        row_vals = []
        for feat in features:
            while True:
                raw = input(f"  {feat}: ").strip()
                if raw.lower() == "q":
                    print("\n  Exiting prediction mode.")
                    return
                try:
                    row_vals.append(float(raw))
                    break
                except ValueError:
                    warn("Please enter a numeric value.")

        row = pd.DataFrame([row_vals], columns=features)
        try:
            prob = rf.predict_proba_single(row)
            pred = int(prob >= 0.5)
            label = "HEART DISEASE DETECTED" if pred == 1 else "No heart disease"
            print(f"\n  Result:      {label}")
            print(f"  Probability: {prob * 100:.2f}%\n")
        except Exception as e:
            err(f"Prediction failed: {e}")

        again = input("  Run another prediction? [Y/n]: ").strip().lower()
        if again == "n":
            break


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Heart Disease ML Pipeline - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", default="dataset/cleanedData/heart_combined.csv",
        help="Path to the CSV dataset"
    )
    parser.add_argument(
        "--target", default="target",
        help="Name of the target column (default: target)"
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="After training, enter interactive prediction mode"
    )
    args = parser.parse_args()

    print("\nHeart Disease Prediction - ML Pipeline\n")

    try:
        state = run_pipeline(args.dataset, args.target)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as exc:
        err(f"Pipeline failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    if args.predict:
        try:
            interactive_predict(state)
        except KeyboardInterrupt:
            print("\n\nBye!")


if __name__ == "__main__":
    main()
