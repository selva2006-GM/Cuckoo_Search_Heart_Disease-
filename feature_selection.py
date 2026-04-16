import numpy as np
from scipy.special import gamma

class FeatureSelector:
    def select_top_k(self, model, X_train, top_k=6):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]
        selected = list(X_train.columns[indices])

        cumulative = 0.0
        cumulative_data = []
        for idx in indices:
            cumulative += importances[idx]
            cumulative_data.append(round(float(cumulative) * 100, 2))

        self.selected_features = selected
        self.cumulative_importance = cumulative_data
        self.total_importance_retained = round(float(cumulative) * 100, 2)
        return selected
