import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.loader import load_csv, build_feature_matrix
from models.logreg import LogisticRegressionScratch


def compute_loss(X, y, w, b, lambda_l2, sw):
    z = np.dot(X, w) + b
    yp = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    yp = np.clip(yp, 1e-15, 1 - 1e-15)
    if sw is not None:
        bce = -np.mean(sw * (y * np.log(yp) + (1 - y) * np.log(1 - yp)))
    else:
        bce = -np.mean(y * np.log(yp) + (1 - y) * np.log(1 - yp))
    return bce + 0.5 * lambda_l2 * np.dot(w, w)


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset")
    cat_cols = ["person_gender", "person_home_ownership", "previous_loan_defaults_on_file"]

    th, tr = load_csv(os.path.join(base_dir, "train.csv"))
    for row in tr:
        row[th.index("person_age")] = str(np.clip(float(row[th.index("person_age")]), 0, 100))

    X, y, fn, cms = build_feature_matrix(th, tr, cat_cols, "loan_status")
    m = X.mean(0); s = X.std(0); s[s == 0] = 1.0
    X = (X - m) / s

    rng = np.random.RandomState(42)
    idx = np.arange(len(y)); rng.shuffle(idx)
    sp = int(0.85 * len(y))
    X_tr, y_tr = X[idx[:sp]], y[idx[:sp]]

    lr_model = LogisticRegressionScratch(
        learning_rate=0.01, n_iterations=5000, lambda_l2=0.01,
        class_weight="balanced", optimizer="adam", verbose=False
    )

    n_features = X_tr.shape[1]
    w = np.zeros(n_features); b = 0.0
    sw = lr_model._compute_sample_weights(y_tr)
    m_w, v_w = np.zeros(n_features), np.zeros(n_features)
    m_b, v_b = 0.0, 0.0; t = 0

    loss_hist = []
    weight_hist = []

    for i in range(lr_model.n_iter):
        z = np.dot(X_tr, w) + b
        yp = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        yp = np.clip(yp, 1e-15, 1 - 1e-15)
        loss = compute_loss(X_tr, y_tr, w, b, 0.01, sw)
        loss_hist.append(loss)
        weight_hist.append(w.copy())

        errors = yp - y_tr
        if sw is not None: errors *= sw
        dw = (1/len(y_tr)) * np.dot(X_tr.T, errors) + 0.01 * w
        db = (1/len(y_tr)) * np.sum(errors)

        t += 1
        m_w = 0.9*m_w + 0.1*dw; m_b = 0.9*m_b + 0.1*db
        v_w = 0.999*v_w + 0.001*(dw**2); v_b = 0.999*v_b + 0.001*(db**2)
        mw_hat = m_w/(1-0.9**t); mb_hat = m_b/(1-0.9**t)
        vw_hat = v_w/(1-0.999**t); vb_hat = v_b/(1-0.999**t)
        w -= 0.01 * mw_hat / (np.sqrt(vw_hat) + 1e-8)
        b -= 0.01 * mb_hat / (np.sqrt(vb_hat) + 1e-8)

        if len(loss_hist) > 1 and abs(loss_hist[-2] - loss_hist[-1]) < 1e-6:
            break

    n_iters = len(loss_hist)
    print(f"Trained {n_iters} iterations, final loss={loss_hist[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(range(n_iters), loss_hist, 'b-', lw=1.2, alpha=0.8)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss (BCE + L2)", fontsize=11)
    ax.set_title("Training Loss Curve", fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    w_final = weight_hist[-1]
    top2 = np.argsort(np.abs(w_final))[-2:]

    ax = axes[1]
    traj = np.array(weight_hist)
    w1_min, w1_max = traj[:, top2[0]].min() - 0.5, traj[:, top2[0]].max() + 0.5
    w2_min, w2_max = traj[:, top2[1]].min() - 0.5, traj[:, top2[1]].max() + 0.5

    w1_range = np.linspace(w1_min, w1_max, 100)
    w2_range = np.linspace(w2_min, w2_max, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = np.zeros_like(W1)

    w_base = w_final.copy()
    for i in range(len(w1_range)):
        for j in range(len(w2_range)):
            w_base[top2[0]] = W1[j, i]
            w_base[top2[1]] = W2[j, i]
            Z[j, i] = compute_loss(X_tr, y_tr, w_base, b, 0.01, sw)

    Z = np.log(Z - Z.min() + 1e-6)
    cf = ax.contourf(W1, W2, Z, levels=30, cmap='YlOrRd', alpha=0.9, extend='both')
    ax.contour(W1, W2, Z, levels=15, colors='white', linewidths=0.4, alpha=0.5)

    ax.plot(traj[:, top2[0]], traj[:, top2[1]], 'b-', lw=1.5, alpha=0.8)
    ax.plot(traj[0, top2[0]], traj[0, top2[1]], 'go', ms=8, label='Start', markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(traj[-1, top2[0]], traj[-1, top2[1]], 'r*', ms=12, label='Converged', markeredgecolor='white', markeredgewidth=1.5)

    feats_short = [fn[i].replace("_", "\n") for i in top2]
    ax.set_xlabel(feats_short[0], fontsize=11)
    ax.set_ylabel(feats_short[1], fontsize=11)
    ax.set_title("Loss Contour & Parameter Trajectory", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    cbar = plt.colorbar(cf, ax=ax, label='log(loss)', shrink=0.85)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    viz_dir = os.path.join(os.path.dirname(__file__), "")
    os.makedirs(viz_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = os.path.join(viz_dir, f"lr_training.{ext}")
        plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        print(f"Saved: {out}")
    plt.close()
