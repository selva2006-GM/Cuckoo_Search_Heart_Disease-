import pandas as pd
import os

# Column names
columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

# Get current script directory
base_path = os.path.dirname(__file__)

# Find all .data files in the same folder
files = [f for f in os.listdir(base_path) if f.endswith(".data")]

print("📂 Found files:", files)

all_data = []

for file in files:
    full_path = os.path.join(base_path, file)
    print(f"Processing {file}...")

    df = pd.read_csv(full_path, names=columns, encoding="latin1")

    df = df.replace("?", pd.NA)
    df = df.dropna()
    df = df.apply(pd.to_numeric)

    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    all_data.append(df)

# Merge everything
if all_data:
    final_data = pd.concat(all_data, ignore_index=True)
    output_path = os.path.join(base_path, "heart_combined.csv")
    final_data.to_csv(output_path, index=False)

    print("🔥 Combined dataset created at:", output_path)
else:
    print("❌ No .data files found. Put them in the same folder as this script.")