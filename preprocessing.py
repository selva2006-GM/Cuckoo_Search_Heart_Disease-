import pandas as pd
import matplotlib.pyplot as plt

class Preprocessing:
    def __init__(self, file):
        self.df = pd.read_csv(file)

    def get_data(self):
        return self.df

    def plot_numeric_columns(self):
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) == 0:
            print("No numeric columns found.")
            return
        
        self.df[numeric_cols].hist(figsize=(12, 8))
        plt.tight_layout()
        plt.show()
