import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.special import gamma

class Preprocessing:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = MinMaxScaler()
        self.df = None
        self.stats = {}
        self._load()

    def _load(self):
        self.df = pd.read_csv(self.filepath)

        raw_shape = self.df.shape
        null_count = int(self.df.isnull().sum().sum())
        dup_count = int(self.df.duplicated().sum())

        # Drop duplicates
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Encode categoricals
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = pd.factorize(self.df[col])[0]

        # Normalise continuous cols
        continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        existing = [c for c in continuous_cols if c in self.df.columns]
        self.df[existing] = self.scaler.fit_transform(self.df[existing])

        # Class distribution
        target_col = 'target' if 'target' in self.df.columns else self.df.columns[-1]
        class_counts = self.df[target_col].value_counts().to_dict()

        self.stats = {
            "raw_rows": raw_shape[0],
            "raw_cols": raw_shape[1],
            "clean_rows": len(self.df),
            "null_count": null_count,
            "dup_count": dup_count,
            "n_features": len(self.df.columns) - 1,
            "class_0": int(class_counts.get(0, 0)),
            "class_1": int(class_counts.get(1, 0)),
            "feature_names": [c for c in self.df.columns if c != target_col],
        }
