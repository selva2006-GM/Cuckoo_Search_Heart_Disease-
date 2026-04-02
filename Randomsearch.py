import random

class RandomSearch:
    def __init__(self, n_trials=20):
        self.n_trials = n_trials

    def optimize(self, model, X_train, X_test, y_train, y_test):
        best_score = 0
        best_params = None

        history = []

        for i in range(self.n_trials):
            n_estimators = random.randint(50, 200)
            max_depth = random.randint(3, 15)

            model.model.set_params(
                n_estimators=n_estimators,
                max_depth=max_depth
            )

            model.train(X_train, y_train)
            acc = model.evaluate(X_test, y_test)

            history.append(acc)

            if acc > best_score:
                best_score = acc
                best_params = (n_estimators, max_depth)

        print("\n🎲 Random Search Best:", best_params, "Score:", best_score)
        return best_params, best_score, history