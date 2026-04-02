import pandas as pd

class FeatureSelector:
    def __init__(self):
        pass

    def select_top_k(self, model, X, top_k=8):
        importances = model.feature_importances_
        feature_names = X.columns

        feature_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })

        feature_df = feature_df.sort_values(by="importance", ascending=False)

        print("\n📊 Feature Importance Ranking:")
        print(feature_df)

        selected_features = feature_df.head(top_k)["feature"]

        print(f"\n✅ Top {top_k} Selected Features:")
        print(selected_features.tolist())

        return selected_features.tolist()