"""Linear SVM from scratch (numpy only).

Binary classification with hinge loss + subgradient descent. No kernel, no regularization.
"""

import numpy as np


def hinge_loss_gradient(w, b, X, y):
    """Subgradient of hinge loss: max(0, 1 - y*(w·x + b))."""
    y = np.where(y == 0, -1, 1)  # convert {0,1} to {-1,+1}
    margins = y * (np.dot(X, w) + b)
    mask = margins < 1

    dw = -np.dot(X[mask].T, y[mask]) / X.shape[0]
    db = -np.sum(y[mask]) / X.shape[0]

    loss = np.maximum(0, 1 - margins).mean()
    return dw, db, loss


class LinearSVMScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-6, verbose=False):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.tol = tol
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        prev_loss = float("inf")

        for i in range(self.n_iter):
            dw, db, loss = hinge_loss_gradient(self.weights, self.bias, X, y)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Early stop at iter {i}, loss={loss:.6f}")
                break
            prev_loss = loss

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if self.verbose and i % 200 == 0:
                print(f"Iter {i}: loss = {loss:.6f}")

        return self

    def decision_function(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.int64)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset_loader import load_dataset

    base = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    data = load_dataset(base)

    svm = LinearSVMScratch(learning_rate=0.1, n_iterations=5000, verbose=True)
    svm.fit(data["X_train"], data["y_train"])

    train_acc = svm.score(data["X_train"], data["y_train"])
    print(f"Train accuracy: {train_acc:.4f}")
