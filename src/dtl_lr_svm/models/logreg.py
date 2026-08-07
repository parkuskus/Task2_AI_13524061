"""Logistic Regression from scratch (numpy only).

Binary classification with sigmoid + class-weighted BCE + L2 regularization + Adam.
Threshold optimization for imbalanced Macro F1.
"""

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-6,
                 lambda_l2=0.0, class_weight=None, optimizer="adam",
                 beta1=0.9, beta2=0.999, eps=1e-8, verbose=False):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.tol = tol
        self.lambda_l2 = lambda_l2
        self.class_weight = class_weight
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.threshold = 0.5

    def _compute_sample_weights(self, y):
        if self.class_weight == "balanced":
            n = len(y)
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            w = np.ones(n)
            w[y == 0] = n / (2.0 * max(n_neg, 1))
            w[y == 1] = n / (2.0 * max(n_pos, 1))
            return w
        elif isinstance(self.class_weight, dict):
            w = np.ones(len(y))
            for cls, weight in self.class_weight.items():
                w[y == cls] = weight
            return w
        return None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        sw = self._compute_sample_weights(y)

        if self.optimizer == "adam":
            m_w, v_w = np.zeros(n_features), np.zeros(n_features)
            m_b, v_b = 0.0, 0.0
            t = 0

        prev_loss = float("inf")

        for i in range(self.n_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)

            eps_val = 1e-15
            y_pred_c = np.clip(y_pred, eps_val, 1 - eps_val)
            if sw is not None:
                bce = -np.mean(sw * (y * np.log(y_pred_c) + (1 - y) * np.log(1 - y_pred_c)))
            else:
                bce = -np.mean(y * np.log(y_pred_c) + (1 - y) * np.log(1 - y_pred_c))
            reg = 0.5 * self.lambda_l2 * np.dot(self.weights, self.weights)
            loss = bce + reg
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Early stop at iter {i}, loss={loss:.6f}")
                break
            prev_loss = loss

            errors = y_pred - y
            if sw is not None:
                errors = errors * sw
            dw = (1 / n_samples) * np.dot(X.T, errors) + self.lambda_l2 * self.weights
            db = (1 / n_samples) * np.sum(errors)

            if self.optimizer == "adam":
                t += 1
                m_w = self.beta1 * m_w + (1 - self.beta1) * dw
                m_b = self.beta1 * m_b + (1 - self.beta1) * db
                v_w = self.beta2 * v_w + (1 - self.beta2) * (dw ** 2)
                v_b = self.beta2 * v_b + (1 - self.beta2) * (db ** 2)
                m_w_hat = m_w / (1 - self.beta1 ** t)
                m_b_hat = m_b / (1 - self.beta1 ** t)
                v_w_hat = v_w / (1 - self.beta2 ** t)
                v_b_hat = v_b / (1 - self.beta2 ** t)
                self.weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
                self.bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
            else:
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            if self.verbose and i % 200 == 0:
                print(f"Iter {i}: loss = {loss:.6f}")

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(np.int64)

    def optimize_threshold(self, X, y, search_range=None):
        if search_range is None:
            search_range = np.linspace(0.1, 0.9, 100)
        best_f1 = -1.0
        best_t = 0.5
        probas = self.predict_proba(X)
        for t in search_range:
            preds = (probas >= t).astype(np.int64)
            f1 = self._macro_f1(y, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        self.threshold = best_t
        return self.threshold, best_f1

    @staticmethod
    def _macro_f1(y_true, y_pred):
        f1s = []
        for cls in [0, 1]:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        return sum(f1s) / len(f1s)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.loader import load_dataset

    base = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset")
    data = load_dataset(base)

    X, y = data["X_train"], data["y_train"]
    rng = np.random.RandomState(42)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    sp = int(0.85 * len(y))
    X_tr, y_tr = X[idx[:sp]], y[idx[:sp]]
    X_val, y_val = X[idx[sp:]], y[idx[sp:]]

    configs = [
        ("Baseline (no weight, SGD)", LogisticRegressionScratch(
            learning_rate=0.1, n_iterations=5000, optimizer="sgd")),
        ("L2=0.01 + balanced + Adam", LogisticRegressionScratch(
            learning_rate=0.01, n_iterations=5000, lambda_l2=0.01,
            class_weight="balanced", optimizer="adam")),
        ("L2=0.1 + balanced + Adam", LogisticRegressionScratch(
            learning_rate=0.01, n_iterations=5000, lambda_l2=0.1,
            class_weight="balanced", optimizer="adam")),
        ("L2=1.0 + balanced + Adam", LogisticRegressionScratch(
            learning_rate=0.01, n_iterations=5000, lambda_l2=1.0,
            class_weight="balanced", optimizer="adam")),
    ]

    best_f1 = -1
    best_name = ""
    for name, model in configs:
        model.fit(X_tr, y_tr)
        opt_t, val_f1 = model.optimize_threshold(X_val, y_val)
        train_pred = model.predict(X_tr)
        train_f1 = model._macro_f1(y_tr, train_pred)
        print(f"  {name:<40}: val_F1={val_f1:.4f}, opt_t={opt_t:.3f}, train_F1={train_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_name = name

    print(f"\nBest: {best_name} (val_F1={best_f1:.4f})")
