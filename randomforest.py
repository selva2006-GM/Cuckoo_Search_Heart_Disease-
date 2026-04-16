import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
from scipy.special import gamma

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        return {
            "accuracy":  round(float(accuracy_score(y_test, y_pred))  * 100, 2),
            "precision": round(float(precision_score(y_test, y_pred)) * 100, 2),
            "recall":    round(float(recall_score(y_test, y_pred))    * 100, 2),
            "f1":        round(float(f1_score(y_test, y_pred))        * 100, 2),
            "auc":       round(float(roc_auc_score(y_test, y_prob)),  3),
            "confusion_matrix": cm.tolist(),
            "roc_fpr":   [round(x, 4) for x in fpr.tolist()],
            "roc_tpr":   [round(x, 4) for x in tpr.tolist()],
        }

    def feature_importances(self, feature_names):
        fi = self.model.feature_importances_
        pairs = sorted(zip(feature_names, fi.tolist()), key=lambda x: -x[1])
        return [{"feature": f, "importance": round(v, 4)} for f, v in pairs]

    def predict_proba_single(self, X):
        return float(self.model.predict_proba(X)[:, 1][0])
