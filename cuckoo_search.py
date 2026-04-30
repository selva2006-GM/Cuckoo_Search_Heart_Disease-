import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.special import gamma


class CuckooSearch:
    def __init__(self, n_nests=40, n_iterations=30, pa=0.15, cv_folds=5):
        """
        n_nests     : number of host nests (population size)
        n_iterations: number of generations
        pa          : nest abandonment probability (lower = keep good nests longer)
        cv_folds    : stratified k-fold folds for fitness evaluation
        """
        self.n_nests     = n_nests
        self.n_iterations = n_iterations
        self.pa          = pa
        self.cv_folds    = cv_folds

    # ── Levy flight ───────────────────────────────────────────────
    def _levy_flight(self, dim):
        beta  = 1.5
        sigma = (
            gamma(1 + beta) * np.sin(np.pi * beta / 2) /
            (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u    = np.random.randn(dim) * sigma
        v    = np.random.randn(dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    # ── Fitness: cross-validated F1 (more reliable than single split) ──
    def _evaluate(self, params, X_train, y_train):
        n_est       = max(10,  min(300, int(round(params[0]))))
        max_d       = max(3,   min(20,  int(round(params[1]))))
        min_split   = max(2,   min(20,  int(round(params[2]))))
        min_leaf    = max(1,   min(10,  int(round(params[3]))))

        model = RandomForestClassifier(
            n_estimators     = n_est,
            max_depth        = max_d,
            min_samples_split= min_split,
            min_samples_leaf = min_leaf,
            random_state     = 42,
            n_jobs           = -1,
        )
        cv  = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        # Use F1 so we don't overfit to majority class
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        return round(float(scores.mean()) * 100, 4)

    # ── Main optimizer ────────────────────────────────────────────
    def optimize(self, rf, X_train, X_test, y_train, y_test, progress_cb=None):
        # Search space: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
        # Bounds informed by earlier runs
        lb = np.array([ 50,  3,  2, 1])
        ub = np.array([600, 30, 20, 10])
        dim = len(lb)

        # ── Initialise population ─────────────────────────────────
        nests   = lb + np.random.rand(self.n_nests, dim) * (ub - lb)
        fitness = np.array([
            self._evaluate(n, X_train, y_train) for n in nests
        ])

        best_idx   = np.argmax(fitness)
        best_nest  = nests[best_idx].copy()
        best_score = fitness[best_idx]
        history    = []

        for iteration in range(self.n_iterations):
            alpha = 0.05  # step-size scaling; small = fine-grained local search

            # ── Levy flights for each nest ─────────────────────────
            for i in range(self.n_nests):
                levy  = self._levy_flight(dim)
                # Move toward best nest, scaled by search-space width
                step  = alpha * levy * (ub - lb)
                new_nest = best_nest + step         # global-best guided
                new_nest = np.clip(new_nest, lb, ub)

                new_score = self._evaluate(new_nest, X_train, y_train)

                # Replace a random nest if new one is better (not the elite)
                j = np.random.randint(self.n_nests)
                while j == best_idx:               # protect elite slot
                    j = np.random.randint(self.n_nests)

                if new_score > fitness[j]:
                    nests[j]   = new_nest
                    fitness[j] = new_score

            # ── Abandon worst nests (pa fraction), keep elite ─────
            n_abandon   = max(1, int(self.pa * self.n_nests))
            sorted_idxs = np.argsort(fitness)       # ascending
            abandon_idxs = [
                idx for idx in sorted_idxs if idx != best_idx
            ][:n_abandon]

            for idx in abandon_idxs:
                nests[idx]   = lb + np.random.rand(dim) * (ub - lb)
                fitness[idx] = self._evaluate(nests[idx], X_train, y_train)

            # ── Update global best ────────────────────────────────
            cur_best_idx = np.argmax(fitness)
            if fitness[cur_best_idx] > best_score:
                best_score = fitness[cur_best_idx]
                best_nest  = nests[cur_best_idx].copy()
                best_idx   = cur_best_idx

            history.append(round(best_score, 4))

            if progress_cb:
                progress_cb({
                    "iteration"  : iteration + 1,
                    "best_score" : round(best_score, 4),
                    "best_params": {
                        "n_estimators"     : int(round(best_nest[0])),
                        "max_depth"        : int(round(best_nest[1])),
                        "min_samples_split": int(round(best_nest[2])),
                        "min_samples_leaf" : int(round(best_nest[3])),
                    },
                    "history": history[:],
                })

        # ── Final evaluation on actual test set ───────────────────
        # (fit on full X_train with best params, score on X_test)
        n_est     = max(10,  min(300, int(round(best_nest[0]))))
        max_d     = max(3,   min(20,  int(round(best_nest[1]))))
        min_split = max(2,   min(20,  int(round(best_nest[2]))))
        min_leaf  = max(1,   min(10,  int(round(best_nest[3]))))

        final_model = RandomForestClassifier(
            n_estimators     = n_est,
            max_depth        = max_d,
            min_samples_split= min_split,
            min_samples_leaf = min_leaf,
            random_state     = 42,
            n_jobs           = -1,
        )
        final_model.fit(X_train, y_train)
        pred        = final_model.predict(X_test)
        test_score  = round(float(accuracy_score(y_test, pred)) * 100, 4)

        # Return test accuracy as final score so pipeline comparison is consistent
        return best_nest, test_score, history


# ─────────────────────────────────────────────────────────────────
# Random Search (also upgraded to CV for fair comparison)
# ─────────────────────────────────────────────────────────────────

class RandomSearch:
    def __init__(self, n_iterations=30, cv_folds=5):
        self.n_iterations = n_iterations
        self.cv_folds     = cv_folds

    def _evaluate(self, params, X_train, y_train):
        n_est     = max(10,  min(600, int(params[0])))
        max_d     = max(3,   min(30,  int(params[1])))
        min_split = max(2,   min(20,  int(params[2])))
        min_leaf  = max(1,   min(10,  int(params[3])))

        model = RandomForestClassifier(
            n_estimators     = n_est,
            max_depth        = max_d,
            min_samples_split= min_split,
            min_samples_leaf = min_leaf,
            random_state     = 42,
            n_jobs           = 1,
        )
        cv     = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        return round(float(scores.mean()))

    def optimize(self, rf, X_train, X_test, y_train, y_test, progress_cb=None):
        best_score  = 0
        best_params = None
        history     = []

        for i in range(self.n_iterations):
            params = np.array([
                np.random.randint(50, 600),
                np.random.randint(3,  30),
                np.random.randint(2,  20),
                np.random.randint(1,  10),
            ])
            score = self._evaluate(params, X_train, y_train)
            if score > best_score:
                best_score  = score
                best_params = params.copy()
            history.append(round(best_score, 4))

            if progress_cb:
                progress_cb({
                    "iteration"  : i + 1,
                    "best_score" : round(best_score, 4),
                    "best_params": {
                        "n_estimators"     : int(best_params[0]),
                        "max_depth"        : int(best_params[1]),
                        "min_samples_split": int(best_params[2]),
                        "min_samples_leaf" : int(best_params[3]),
                    },
                    "history": history[:],
                })

        # Final test-set score with best params found
        from sklearn.metrics import accuracy_score as _acc
        n_est     = max(10,  min(300, int(best_params[0])))
        max_d     = max(3,   min(20,  int(best_params[1])))
        min_split = max(2,   min(20,  int(best_params[2])))
        min_leaf  = max(1,   min(10,  int(best_params[3])))

        final_model = RandomForestClassifier(
            n_estimators     = n_est,
            max_depth        = max_d,
            min_samples_split= min_split,
            min_samples_leaf = min_leaf,
            random_state     = 42,
            n_jobs            = -1,
        )
        final_model.fit(X_train, y_train)
        pred       = final_model.predict(X_test)
        test_score = round(float(_acc(y_test, pred)) * 100, 4)

        return best_params, test_score, history
