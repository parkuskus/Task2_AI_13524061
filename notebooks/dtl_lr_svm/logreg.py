"""Logistic Regression from scratch (numpy only).

Binary classification with sigmoid + gradient descent. No regularization.
"""

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-6, verbose=False):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.tol = tol
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        prev_loss = float("inf")

        for i in range(self.n_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)

            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Early stop at iter {i}, loss={loss:.6f}")
                break
            prev_loss = loss

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.verbose and i % 200 == 0:
                print(f"Iter {i}: loss = {loss:.6f}")

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset_loader import load_dataset

    base = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    data = load_dataset(base)

    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=5000, verbose=True)
    model.fit(data["X_train"], data["y_train"])

    train_acc = model.score(data["X_train"], data["y_train"])
    print(f"Train accuracy: {train_acc:.4f}")
