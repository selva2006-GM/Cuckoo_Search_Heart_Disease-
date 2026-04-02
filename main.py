from preprocessing import Preprocessing
from randomforest import RandomForestModel
from feature_selection import FeatureSelector
from sklearn.model_selection import train_test_split
from cuckoo_search import CuckooSearch
from cuckoo_search import CuckooSearch
from Randomsearch import RandomSearch

class Model:
    def __init__(self):
        self.preprocessing = None
        self.df = None
        self.rf = RandomForestModel()
        self.fs = FeatureSelector()

    def load_data(self, file, target_column):
        self.preprocessing = Preprocessing(file)
        self.df = self.preprocessing.df

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def run_random_forest(self):
        print("\n🚀 Running Cuckoo Search Optimization...")

        cs = CuckooSearch()
        best_params, best_score = cs.optimize(
            self.rf,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        )

        print("\n🔥 Best Parameters Found:", best_params)

        # Apply best params
        self.rf.model.set_params(
            n_estimators=int(best_params[0]),
            max_depth=int(best_params[1])
        )

        print("\n🌲 Training Final Model with Optimized Parameters...")
        self.rf.train(self.X_train, self.y_train)
        acc_before = self.rf.evaluate(self.X_test, self.y_test)

        # Feature Selection (same as before)
        selected_features = self.fs.select_top_k(
            self.rf.model, self.X_train, top_k=8
        )

        X_train_fs = self.X_train[selected_features]
        X_test_fs = self.X_test[selected_features]

        print("\n🔁 Retraining after Feature Selection...")
        self.rf.train(X_train_fs, self.y_train)
        acc_after = self.rf.evaluate(X_test_fs, self.y_test)

        # Final dashboard
        self.rf.plot_all(
            X_train_fs,
            self.y_train,
            self.y_test,
            X_train_fs.columns,
            acc_before,
            acc_after
        )
        
    def plot_optimization_comparison(self, rs_history, cs_history):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))

        plt.plot(rs_history, label="Random Search", marker='o')
        plt.plot(cs_history, label="Cuckoo Search", marker='o')

        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Random vs Cuckoo Search")
        plt.legend()
        plt.grid()

        plt.show()    
    
    
    def run_optimization_comparison(self):
        print("\n🎲 Running Random Search...")
        rs = RandomSearch()
        rs_params, rs_score, rs_history = rs.optimize(
            self.rf, self.X_train, self.X_test, self.y_train, self.y_test
        )

        print("\n🐦 Running Cuckoo Search...")
        cs = CuckooSearch()
        cs_params, cs_score, cs_history = cs.optimize(
            self.rf, self.X_train, self.X_test, self.y_train, self.y_test
        )

        print("\n📊 FINAL COMPARISON")
        print(f"Random Search Score : {rs_score:.4f}")
        print(f"Cuckoo Search Score: {cs_score:.4f}")

        # 👉 Choose best algorithm
        if cs_score > rs_score:
            best_params = cs_params
            print("🔥 Using Cuckoo Search Best Params")
        else:
            best_params = rs_params
            print("🎲 Using Random Search Best Params")

        # Apply best params
        self.rf.model.set_params(
            n_estimators=int(best_params[0]),
            max_depth=int(best_params[1])
        )

        # Train with best params
        self.rf.train(self.X_train, self.y_train)
        acc_before = self.rf.evaluate(self.X_test, self.y_test)

        # Feature Selection
        selected_features = self.fs.select_top_k(
            self.rf.model, self.X_train, top_k=8
        )

        X_train_fs = self.X_train[selected_features]
        X_test_fs = self.X_test[selected_features]

        print("\n🔁 Retraining AFTER Feature Selection...")
        self.rf.train(X_train_fs, self.y_train)
        acc_after = self.rf.evaluate(X_test_fs, self.y_test)

        # 🔥 FINAL DASHBOARD (everything)
        self.rf.plot_all(
            X_train_fs,
            self.y_train,
            self.y_test,
            X_train_fs.columns,
            acc_before,
            acc_after
        )

        # 🔥 Optimization comparison graph
        self.plot_optimization_comparison(rs_history, cs_history)
        
        
        
        
        
if __name__ == "__main__":
    model = Model()

    print("Preprocessing the data............")
    model.load_data('dataset/cleaned data/heart_combined.csv', 'target')
    print("preprocessing done!")

    print("\n🚀 Running FULL PIPELINE...")
    model.run_optimization_comparison()

    print("\n✅ Done!")