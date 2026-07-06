"""
LightGBM acquisition-ranking model — the model that must beat the baselines.
(Trains on the SCKN-chart success label; used to rank translation candidates.)

Pipeline
--------
1. **Hyperparameter selection** via *time-aware* CV on the training years only
   (expanding window: fit <=Y, score Y+1, for Y in 2011..2013). A random k-fold
   would leak future books; this mirrors deployment. Selection metric = PR-AUC.
2. **Imbalance** handled with `scale_pos_weight = n_neg / n_pos`, recomputed for
   whatever rows a given fit sees (no SMOTE).
3. **Final fit** on the full train split with the chosen params and an
   n_estimators taken from the CV early-stopping iterations (so val is never
   used for stopping).
4. **Calibration** (Platt/sigmoid) fit on the val split via a frozen estimator.
   Ranking metrics (PR-AUC, precision@K) are invariant to this monotonic map, so
   the val ranking numbers are unaffected; calibration only fixes the probability
   scale (reported via Brier score) for downstream decision thresholds.

Features: `feature_columns` (czech_pub_year excluded — see split.py).
Reporting: val only here (vs the baselines). Test stays sealed for `evaluate.py`
step / the final notebook.

Outputs
-------
- data/models/lgbm.joblib            — fitted LightGBM (uncalibrated)
- data/models/lgbm_calibrated.joblib — sigmoid-calibrated wrapper
- data/models/lgbm_params.json       — chosen hyperparameters + CV score
- reports/model_comparison_val.csv   — popularity / logreg / lgbm on val

Usage
-----
    python src/models/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.split import load_splits, feature_columns, YEAR_COL  # noqa: E402
from src.models.evaluate import evaluate, results_table               # noqa: E402
from src.models.baselines import popularity_scores, fit_logreg        # noqa: E402

REPO    = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports"
MODELS  = REPO / "data" / "models"

RANDOM_STATE = 0

# Small, deliberately conservative grid — only ~835 train positives, so we keep
# the trees shallow and leaves well-populated to avoid overfitting.
PARAM_GRID = [
    {"num_leaves": 15, "min_child_samples": 30},
    {"num_leaves": 15, "min_child_samples": 60},
    {"num_leaves": 31, "min_child_samples": 30},
    {"num_leaves": 31, "min_child_samples": 60},
]

FIXED_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    n_estimators=1000,          # upper bound; early stopping picks the real count
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

# Expanding-window CV: fit on years <= cut, validate on cut+1.
CV_VAL_YEARS = (2012, 2013, 2014)
EARLY_STOPPING_ROUNDS = 50


def _scale_pos_weight(y) -> float:
    pos = int(np.sum(y))
    neg = len(y) - pos
    return neg / pos if pos else 1.0


def _fit_one(params, X_tr, y_tr, X_va, y_va, features):
    """Fit a LightGBM with early stopping on (X_va, y_va); return (model, best_iter, pr_auc)."""
    model = lgb.LGBMClassifier(
        **FIXED_PARAMS, **params, scale_pos_weight=_scale_pos_weight(y_tr),
    )
    model.fit(
        X_tr[features], y_tr,
        eval_set=[(X_va[features], y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(0)],
    )
    best_iter = model.best_iteration_ or FIXED_PARAMS["n_estimators"]
    scores = model.predict_proba(X_va[features])[:, 1]
    return model, best_iter, average_precision_score(y_va, scores)


def time_aware_cv(params, X, y, features, val_years=CV_VAL_YEARS):
    """Mean PR-AUC over expanding-window folds, plus the per-fold best iterations."""
    pr_aucs, best_iters = [], []
    for vy in val_years:
        tr = X[YEAR_COL] < vy
        va = X[YEAR_COL] == vy
        if va.sum() == 0 or y[tr].sum() == 0:
            continue
        _, best_iter, pr = _fit_one(params, X[tr], y[tr], X[va], y[va], features)
        pr_aucs.append(pr)
        best_iters.append(best_iter)
    return float(np.mean(pr_aucs)), best_iters


def tune(X, y, features):
    """Grid-search PARAM_GRID by time-aware CV PR-AUC. Returns (best_params, n_estimators, cv_score)."""
    results = []
    for params in PARAM_GRID:
        cv_pr, best_iters = time_aware_cv(params, X, y, features)
        n_est = max(int(np.median(best_iters)), 50) if best_iters else 200
        results.append((cv_pr, params, n_est))
        print(f"  cv PR-AUC={cv_pr:.4f}  n_est~{n_est:<4}  {params}")
    cv_pr, best_params, n_est = max(results, key=lambda r: r[0])
    print(f"  -> best: PR-AUC={cv_pr:.4f}  n_est={n_est}  {best_params}")
    return best_params, n_est, cv_pr


def fit_final(X_tr, y_tr, features, params, n_estimators):
    """Refit on the full train split with the chosen params (no early stopping)."""
    final_params = {**FIXED_PARAMS, **params, "n_estimators": n_estimators,
                    "scale_pos_weight": _scale_pos_weight(y_tr)}
    model = lgb.LGBMClassifier(**final_params)
    model.fit(X_tr[features], y_tr)
    return model


def main() -> pd.DataFrame:
    s = load_splits()
    features = feature_columns(s.X_train)

    print(f"Tuning LightGBM via time-aware CV on train years "
          f"(val folds {CV_VAL_YEARS})…")
    best_params, n_estimators, cv_pr = tune(s.X_train, s.y_train, features)

    print("\nFitting final model on the full train split…")
    model = fit_final(s.X_train, s.y_train, features, best_params, n_estimators)

    # Calibrate on val (sigmoid; robust for the small positive count).
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(s.X_val[features], s.y_val)

    # ── Evaluate on val, alongside the baselines ─────────────────────────────
    logreg = fit_logreg(s.X_train, s.y_train, features)
    rows = [
        evaluate(s.y_val, popularity_scores(s.X_val), name="popularity_only"),
        evaluate(s.y_val, logreg.predict_proba(s.X_val[features])[:, 1],
                 name="logreg_balanced"),
        evaluate(s.y_val, model.predict_proba(s.X_val[features])[:, 1],
                 name="lightgbm"),
    ]
    table = results_table(rows)

    # Calibration quality (ranking metrics unchanged, so only Brier is reported).
    uncal_p = model.predict_proba(s.X_val[features])[:, 1]
    cal_p   = calibrated.predict_proba(s.X_val[features])[:, 1]
    brier_uncal = brier_score_loss(s.y_val, uncal_p)
    brier_cal   = brier_score_loss(s.y_val, cal_p)

    print(f"\nModel comparison on VALIDATION (n={len(s.y_val):,}, "
          f"base_rate={s.y_val.mean():.1%}):\n")
    print(table.to_string())
    print(f"\nLightGBM Brier score  uncalibrated={brier_uncal:.4f}  "
          f"calibrated={brier_cal:.4f}")

    # ── Persist artifacts + metrics ──────────────────────────────────────────
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "lgbm.joblib")
    joblib.dump(calibrated, MODELS / "lgbm_calibrated.joblib")
    (MODELS / "lgbm_params.json").write_text(json.dumps({
        "fixed_params": FIXED_PARAMS,
        "tuned_params": best_params,
        "n_estimators": n_estimators,
        "scale_pos_weight": _scale_pos_weight(s.y_train),
        "cv_pr_auc": round(cv_pr, 4),
        "features": features,
        "brier_val_uncalibrated": round(float(brier_uncal), 4),
        "brier_val_calibrated": round(float(brier_cal), 4),
    }, indent=2), encoding="utf-8")

    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "model_comparison_val.csv")
    print(f"\nSaved model -> data/models/lgbm.joblib (+ calibrated, + params)")
    print(f"Saved metrics -> reports/model_comparison_val.csv")
    return table


if __name__ == "__main__":
    main()
