from preprocessing import load_data
from model_training import train_model
from cuckoo_search import cuckoo_search
from feature_selection import get_feature_names

# Load data
X, y = load_data("dataset/cleaned data/heart_combined.csv")

# Baseline
baseline_acc = train_model(X, y)
print("Baseline Accuracy:", baseline_acc)

