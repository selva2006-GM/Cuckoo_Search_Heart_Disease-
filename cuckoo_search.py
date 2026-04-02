import numpy as np
import random


class CuckooSearch:
    def __init__(self, n_nests=10, n_iterations=20, pa=0.25):
        self.n_nests = n_nests
        self.n_iterations = n_iterations
        self.pa = pa  # probability of abandoning nest

    def fitness(self, solution, model, X_train, X_test, y_train, y_test):
        n_estimators = int(solution[0])
        max_depth = int(solution[1])

        # Update model params
        model.model.set_params(
            n_estimators=n_estimators,
            max_depth=max_depth
        )

        model.train(X_train, y_train)
        acc = model.evaluate(X_test, y_test)

        return acc

    def generate_solution(self):
        return [
            random.randint(50, 200),   # n_estimators
            random.randint(3, 15)      # max_depth
        ]
    
    def plot_convergence(self, history):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(history, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Best Accuracy")
        plt.title("Cuckoo Search Convergence")
        plt.grid()
        plt.show()
    
    def optimize(self, model, X_train, X_test, y_train, y_test):
        import matplotlib.pyplot as plt

        nests = [self.generate_solution() for _ in range(self.n_nests)]
        fitness_values = [
            self.fitness(n, model, X_train, X_test, y_train, y_test)
            for n in nests
        ]

        best_history = []

        for iteration in range(self.n_iterations):
            for i in range(self.n_nests):
                new_solution = self.generate_solution()
                new_score = self.fitness(new_solution, model, X_train, X_test, y_train, y_test)

                if new_score > fitness_values[i]:
                    nests[i] = new_solution
                    fitness_values[i] = new_score

            best_score = max(fitness_values)
            best_history.append(best_score)

            print(f"Iteration {iteration+1}: Best Score = {best_score:.4f}")

        best_idx = fitness_values.index(max(fitness_values))
        best_nest = nests[best_idx]

        # 🔥 Plot convergence
        self.plot_convergence(best_history)

        return best_nest, best_score, best_history



