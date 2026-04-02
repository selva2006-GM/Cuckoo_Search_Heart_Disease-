import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.model_selection import learning_curve


class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        self.y_pred = self.model.predict(X_test)
        self.y_prob = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, self.y_pred)
        print("Accuracy:", acc)
        return acc

    # ======================
    # DASHBOARD (ALL GRAPHS)
    # ======================
    def plot_all(self, X_train, y_train, y_test, feature_names, acc_before=None, acc_after=None):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # Confusion Matrix
        cm = confusion_matrix(y_test, self.y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=axes[0])
        axes[0].set_title("Confusion Matrix")

        # ROC
        fpr, tpr, _ = roc_curve(y_test, self.y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        axes[1].plot([0, 1], [0, 1], linestyle='--')
        axes[1].legend()
        axes[1].set_title("ROC Curve")

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, self.y_prob)
        axes[2].plot(recall, precision)
        axes[2].set_title("Precision-Recall")

        # Feature Importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        axes[3].bar(range(len(importances)), importances[indices])
        axes[3].set_xticks(range(len(importances)))
        axes[3].set_xticklabels(feature_names[indices], rotation=90)
        axes[3].set_title("Feature Importance")

        # Learning Curve
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_train, y_train, cv=5
        )

        axes[4].plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
        axes[4].plot(train_sizes, np.mean(test_scores, axis=1), label="Validation")
        axes[4].legend()
        axes[4].set_title("Learning Curve")

        # Comparison (Before vs After FS)
        if acc_before is not None and acc_after is not None:
            labels = ["Before FS", "After FS"]
            values = [acc_before, acc_after]

            axes[5].bar(labels, values)
            axes[5].set_title("Feature Selection Impact")

            for i, v in enumerate(values):
                axes[5].text(i, v + 0.01, f"{v:.2f}", ha='center')
        else:
            fig.delaxes(axes[5])

        plt.tight_layout()
        plt.show()