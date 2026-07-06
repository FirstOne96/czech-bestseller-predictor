"""
Baseline models — the floor that the LightGBM model must beat.

Two baselines, both evaluated on the **validation** split (test stays sealed
until the final step):

1. **Popularity-only** — rank books by `pre_cutoff_ratings_count`, nothing
   learned. This *is* the thesis's null hypothesis: "a publisher could just sort
   foreign titles by how popular they already are and skim the top." If the real
   model can't beat this, the multivariate approach isn't justified.

2. **Logistic regression** — `StandardScaler` + `LogisticRegression`
   (`class_weight="balanced"` for the ~16:1 imbalance) on the standard feature
   set (`feature_columns`, i.e. czech_pub_year excluded — see split.py).

Both report through `src/models/evaluate.py` (PR-AUC + precision/recall/lift@K).

Outputs
-------
- prints the comparison table
- writes `reports/baselines_val.csv`

Usage
-----
    python src/models/baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.split import load_splits, feature_columns  # noqa: E402
from src.models.evaluate import evaluate, results_table     # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports"

POPULARITY_FEATURE = "pre_cutoff_ratings_count"


def popularity_scores(X: pd.DataFrame) -> pd.Series:
    """Score = pre-cutoff ratings count. Pure ranking, no fitting."""
    return X[POPULARITY_FEATURE].astype(float)


def fit_logreg(X_train: pd.DataFrame, y_train, features: list[str]) -> object:
    """Fit a class-balanced logistic regression on the standard feature set."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
        ),
    )
    model.fit(X_train[features], y_train)
    return model


def main() -> pd.DataFrame:
    s = load_splits()
    features = feature_columns(s.X_train)  # czech_pub_year excluded

    rows = []

    # 1. Popularity-only ranker (no training).
    rows.append(evaluate(
        s.y_val, popularity_scores(s.X_val), name="popularity_only",
    ))

    # 2. Logistic regression.
    logreg = fit_logreg(s.X_train, s.y_train, features)
    logreg_val_scores = logreg.predict_proba(s.X_val[features])[:, 1]
    rows.append(evaluate(s.y_val, logreg_val_scores, name="logreg_balanced"))

    table = results_table(rows)
    print(f"Baselines on the VALIDATION split "
          f"(n={len(s.y_val):,}, positives={int(s.y_val.sum())}, "
          f"base_rate={s.y_val.mean():.1%}):\n")
    print(table.to_string())
    print(f"\nFeatures used by logreg ({len(features)}): {features}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "baselines_val.csv"
    table.to_csv(out)
    print(f"\nWrote {out.relative_to(REPO)}")
    return table


if __name__ == "__main__":
    main()
