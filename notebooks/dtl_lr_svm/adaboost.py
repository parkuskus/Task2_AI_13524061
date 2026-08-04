"""AdaBoost with CART as weak learner (numpy only).

Implements AdaBoost-SAMME with early stopping on validation F1.
Optimization algorithm for imbalanced classification (bonus).
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dtl_cart import CARTDecisionTree, macro_f1_score


class AdaBoostCART:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=4,
                 min_samples_leaf=15, random_seed=42, filter_defaults=False):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_seed = random_seed
        self.filter_defaults = filter_defaults
        self.estimators = []
        self.alphas = []
        self.threshold = 0.5
        self._filter_mask = None

    def fit(self, X, y, X_val=None, y_val=None):
        n = X.shape[0]

        if self.filter_defaults:
            self._filter_mask = X[:, 10] <= 0
            X_fit, y_fit = X[self._filter_mask], y[self._filter_mask]
        else:
            self._filter_mask = None
            X_fit, y_fit = X, y

        n_fit = X_fit.shape[0]
        weights = np.ones(n_fit) / n_fit
        y_svm = 2 * y_fit - 1

        best_val_f1 = -1
        patience = 10
        no_improve = 0

        for t in range(self.n_estimators):
            tree = CARTDecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_seed=self.random_seed + t,
            )
            tree.fit(X_fit, y_fit, sample_weights=weights)
            preds = tree.predict(X_fit)
            preds_svm = 2 * preds - 1

            wrong = (preds != y_fit).astype(np.float64)
            err = np.sum(weights * wrong) / np.sum(weights)

            if err >= 0.5 or err == 0:
                if err == 0:
                    alpha = self.lr * 10
                else:
                    break
            else:
                alpha = self.lr * 0.5 * np.log((1 - err) / err)

            self.estimators.append(tree)
            self.alphas.append(alpha)

            weights *= np.exp(-alpha * y_svm * preds_svm)
            weights /= weights.sum()

            if X_val is not None and y_val is not None:
                val_f1 = self._eval_f1(X_val, y_val)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break

        return self

    def _raw_score(self, X):
        if self._filter_mask is not None:
            n = X.shape[0]
            score = np.zeros(n)
            def_mask = X[:, 10] > 0
            no_def_idx = np.where(~def_mask)[0]
            if len(no_def_idx) > 0:
                X_sub = X[no_def_idx]
                for tree, alpha in zip(self.estimators, self.alphas):
                    score[no_def_idx] += alpha * (2 * tree.predict(X_sub) - 1)
            return score
        score = np.zeros(X.shape[0])
        for tree, alpha in zip(self.estimators, self.alphas):
            score += alpha * (2 * tree.predict(X) - 1)
        return score

    def predict_proba(self, X):
        score = self._raw_score(X)
        prob = 1.0 / (1.0 + np.exp(-np.clip(score, -500, 500)))
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(np.int64)

    def _eval_f1(self, X, y):
        return macro_f1_score(y, self.predict(X))

    def optimize_threshold(self, X, y, search_range=None):
        if search_range is None:
            search_range = np.linspace(0.1, 0.9, 100)
        best_f1 = -1.0
        best_t = 0.5
        probas = self.predict_proba(X)
        for t in search_range:
            preds = (probas[:, 1] >= t).astype(np.int64)
            f1 = macro_f1_score(y, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        self.threshold = best_t
        return self.threshold, best_f1

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    from dataset_loader import load_dataset

    base = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    data = load_dataset(base)

    X, y = data["X_train"], data["y_train"]
    rng = np.random.RandomState(42)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    sp = int(0.85 * len(y))
    X_tr, y_tr = X[idx[:sp]], y[idx[:sp]]
    X_val, y_val = X[idx[sp:]], y[idx[sp:]]

    configs = [
        ("AdaBoost (T=50, depth=3, leaf=20, lr=0.1)", AdaBoostCART(
            n_estimators=50, max_depth=3, min_samples_leaf=20, learning_rate=0.1)),
        ("AdaBoost (T=100, depth=3, leaf=15, lr=0.1)", AdaBoostCART(
            n_estimators=100, max_depth=3, min_samples_leaf=15, learning_rate=0.1)),
        ("AdaBoost (T=100, depth=4, leaf=10, lr=0.1)", AdaBoostCART(
            n_estimators=100, max_depth=4, min_samples_leaf=10, learning_rate=0.1)),
        ("AdaBoost+filter (T=100, d=4, leaf=10, lr=0.1)", AdaBoostCART(
            n_estimators=100, max_depth=4, min_samples_leaf=10, learning_rate=0.1,
            filter_defaults=True)),
        ("AdaBoost+filter (T=100, d=5, leaf=5, lr=0.1)", AdaBoostCART(
            n_estimators=100, max_depth=5, min_samples_leaf=5, learning_rate=0.1,
            filter_defaults=True)),
        ("AdaBoost+filter (T=200, d=5, leaf=5, lr=0.05)", AdaBoostCART(
            n_estimators=200, max_depth=5, min_samples_leaf=5, learning_rate=0.05,
            filter_defaults=True)),
    ]

    best_f1 = -1
    best_name = ""
    for name, model in configs:
        model.fit(X_tr, y_tr, X_val, y_val)
        opt_t, val_f1 = model.optimize_threshold(X_val, y_val)
        train_pred = model.predict(X_tr)
        train_f1 = macro_f1_score(y_tr, train_pred)
        print(f"  {name:<50}: val_F1={val_f1:.4f}, opt_t={opt_t:.3f}, train_F1={train_f1:.4f}, trees={len(model.estimators)}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_name = name

    print(f"\nBest: {best_name} (val_F1={best_f1:.4f})")
