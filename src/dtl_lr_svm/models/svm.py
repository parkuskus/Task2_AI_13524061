"""Linear SVM from scratch (numpy only).

Binary classification with class-weighted hinge loss + L2 regularization + Adam.
Threshold optimization for imbalanced Macro F1.
"""

import numpy as np


class LinearSVMScratch:
    def __init__(self, C=1.0, learning_rate=0.01, n_iterations=1000, tol=1e-6,
                 class_weight=None, optimizer="adam", beta1=0.9, beta2=0.999,
                 eps=1e-8, verbose=False):
        self.C = C
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.tol = tol
        self.class_weight = class_weight
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.threshold = 0.0

    def _compute_sample_weights(self, y):
        if self.class_weight == "balanced":
            n_samples = len(y)
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            w = np.ones(n_samples)
            w[y == 0] = n_samples / (2.0 * max(n_neg, 1))
            w[y == 1] = n_samples / (2.0 * max(n_pos, 1))
            return w
        elif isinstance(self.class_weight, dict):
            w = np.ones(len(y))
            for cls, weight in self.class_weight.items():
                w[y == cls] = weight
            return w
        return None

    def _hinge_loss_and_grad(self, w, b, X, y, sample_weights):
        y_svm = np.where(y == 0, -1, 1)
        margins = y_svm * (np.dot(X, w) + b)
        violated = margins < 1

        if sample_weights is not None:
            losses = np.maximum(0, 1 - margins) * sample_weights
            data_loss = losses.mean()
            dw_data = -np.dot((X[violated].T * sample_weights[violated]), y_svm[violated]) / X.shape[0]
            db_data = -np.sum(y_svm[violated] * sample_weights[violated]) / X.shape[0]
        else:
            data_loss = np.maximum(0, 1 - margins).mean()
            dw_data = -np.dot(X[violated].T, y_svm[violated]) / X.shape[0]
            db_data = -np.sum(y_svm[violated]) / X.shape[0]

        reg_loss = 0.5 * np.dot(w, w)
        total_loss = self.C * data_loss + reg_loss

        dw = self.C * dw_data + w
        db = self.C * db_data
        return dw, db, total_loss

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        sample_weights = self._compute_sample_weights(y)

        if self.optimizer == "adam":
            m_w, v_w = np.zeros(n_features), np.zeros(n_features)
            m_b, v_b = 0.0, 0.0
            t = 0

        prev_loss = float("inf")

        for i in range(self.n_iter):
            dw, db, loss = self._hinge_loss_and_grad(
                self.weights, self.bias, X, y, sample_weights)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Early stop at iter {i}, loss={loss:.6f}")
                break
            prev_loss = loss

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

    def decision_function(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict_proba(self, X):
        scores = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))

    def predict(self, X):
        return (self.decision_function(X) >= self.threshold).astype(np.int64)

    def optimize_threshold(self, X, y, search_range=None):
        if search_range is None:
            search_range = np.linspace(-0.5, 0.5, 100)
        best_f1 = -1.0
        best_t = 0.0
        for t in search_range:
            preds = (self.decision_function(X) >= t).astype(np.int64)
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
        ("Baseline (no weight, no reg, SGD)", LinearSVMScratch(
            C=1.0, learning_rate=0.1, n_iterations=5000, optimizer="sgd")),
        ("Class-balanced + L2 (C=1)", LinearSVMScratch(
            C=1.0, learning_rate=0.01, n_iterations=5000, class_weight="balanced",
            optimizer="adam", verbose=False)),
        ("Class-balanced + L2 (C=0.5)", LinearSVMScratch(
            C=0.5, learning_rate=0.01, n_iterations=5000, class_weight="balanced",
            optimizer="adam")),
        ("Class-balanced + L2 (C=2)", LinearSVMScratch(
            C=2.0, learning_rate=0.01, n_iterations=5000, class_weight="balanced",
            optimizer="adam")),
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
