# Task 2 - AI Lab Selection

<p align="center">
  <img src="docs/Project Spesification/public/decision-tree-design-vector.jpg" alt="Cover" width="600"/>
</p>

Repository for **Task #2** of the Artificial Intelligence Laboratory selection process at Institut Teknologi Bandung. Contains two independent projects:

| Project          | Domain                            | Algorithms                                            |
| ---------------- | --------------------------------- | ----------------------------------------------------- |
| **Local Search** | Macroeconomic policy optimisation | Hill-Climbing, Simulated Annealing, Genetic Algorithm |
| **DTL/LR/SVM**   | Loan acceptance classification    | CART, Logistic Regression, Linear SVM                 |

---

## Overview

### Local Search - BI-Rate Optimisation

Determines an 8-quarter trajectory of the BI benchmark interest rate to maximise social welfare while satisfying five macroeconomic constraints (C1–C5). The economy is simulated via a New Keynesian open-economy model that propagates rate decisions through output, inflation, and exchange-rate channels. State space: 21⁸ ≈ 3.78 × 10¹⁰ - explored by local search.

### DTL/LR/SVM - Loan Acceptance Prediction

Kaggle classification task predicting `loan_status` (0 = rejected, 1 = approved) from demographic, financial, and credit-history features. All three models are implemented **from scratch** using only NumPy. Hyperparameter tuning is performed via grid search and Bayesian optimisation with a Gaussian Process surrogate.

---

## Features

### Local Search

- Three search algorithms: **Hill-Climbing** (4 variants: steepest-ascent, sideways, stochastic, random-restart), **Simulated Annealing**, and **Genetic Algorithm**
- Full New Keynesian macroeconomic simulation (IS curve, Phillips curve, UIP) with stochastic shocks
- Five constraints: four hard (C1–C4), one soft/penalised (C5)
- Quadratic loss function with configurable weights, discount factor, and constraint penalties
- RMSD evaluation metric independent of objective weights
- **Bonus** Interactive GUI with real-time convergence plots, constraint status, and hyperparameter sliders (`gui_app.py`)
- **Bonus** Search animation export to GIF (`plot_animation.py`)

### DTL/LR/SVM

- **CART Decision Tree** - Gini & Twoing criterion, weighted splitting, per-class min-leaf, F1-based cost-complexity pruning
- **Logistic Regression** - class-weighted BCE, L2 regularisation, Adam/SGD optimiser, threshold tuning
- **Linear SVM** - class-weighted hinge loss, L2 regularisation, Adam/SGD, sigmoid calibration
- Hyperparameter exploration via grid comparison and Bayesian optimisation (GP + Expected Improvement)
- Side-by-side comparison against scikit-learn baselines (`utils/compare.py`)
- Tree visualisation (depth-limited rendering) and LR loss-contour plots

---

## Dependencies & Prerequisites

- Python 3.10+
- Common packages:

```bash
pip install numpy matplotlib scikit-learn
```

For the GUI (`gui_app.py`), Tkinter is required (bundled with standard Python on most platforms).

---

## Project Structure

```
Task2_AI_13524061/
├── dataset/                          # train/test CSV
├── docs/
│   ├── Project Spesification/
│   │   ├── public/                   # images, cover art
│   │   ├── sections/                 # LaTeX spec chapters
│   │   └── main.tex
│   └── Write-Up/
│       └── Kaggle_Writeup.tex
├── notebooks/
│   ├── dtl_lr_svm/experiments.ipynb
│   └── local_search/local_search_exploration.ipynb
├── submissions/                      # generated Kaggle CSVs
├── src/
│   ├── local_search/
│   │   ├── economy/                  # shocks, simulation
│   │   ├── evaluation/               # constraints, objective, metrics
│   │   ├── search/                   # HC, SA, GA, neighbours, utils
│   │   ├── visualization/            # output GIFs
│   │   ├── config.py                 # model parameters
│   │   ├── main.py                   # CLI entry point
│   │   ├── gui_app.py                # interactive GUI (bonus)
│   │   └── plot_animation.py         # GIF animation (bonus)
│   └── dtl_lr_svm/
│       ├── models/                   # cart, logreg, svm
│       ├── scripts/                  # bayesian_opt, best_cart
│       ├── utils/                    # loader, eda, compare
│       ├── visualization/            # tree & LR contour plots
│       └── main.py                   # CLI entry point
├── extra/                            # supplementary docs & data
├── LICENSE
└── README.md
```

---

## How to Run

### Local Search

```bash
cd src/local_search

# Default run (steepest-ascent + SA + GA)
python main.py

# Specify Hill-Climbing variant
python main.py --hc stochastic

# Generate search animation (GIF)
python plot_animation.py --algo SA --max-iter 200

# Launch interactive GUI
python gui_app.py
```

### DTL/LR/SVM

```bash
cd src/dtl_lr_svm

# Train all three models and generate submissions
python main.py

# Bayesian hyperparameter tuning for CART
python scripts/bayesian_opt.py

# Generate submission with best CART config
python scripts/best_cart.py

# Visualise trained tree
python visualization/visualize_tree.py

# Visualise LR training contour
python visualization/visualize_lr.py

# Compare from-scratch vs scikit-learn
python utils/compare.py
```

**Notebooks** are located under `notebooks/local_search/` and `notebooks/dtl_lr_svm/` with experiment logs and analysis.

---

## License

See [LICENSE](LICENSE) for details.
