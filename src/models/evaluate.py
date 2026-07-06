"""
Evaluation harness for the acquisition-ranking model — shared by baselines and models.

Why these metrics
-----------------
The positive class is ~6% (in-cohort), so **accuracy and ROC-AUC are misleading**
(a model predicting "never a bestseller" scores 94% accuracy and learns nothing).
We report:

- **PR-AUC** (average precision) — the headline number; area under the
  precision-recall curve, robust to imbalance.
- **precision@K / recall@K** — the acquisition-shortlist framing: "if we
  shortlist the K highest-scored foreign titles, what fraction turned out to be
  successful acquisitions (precision), and what fraction of all successes did
  we catch (recall)?"
- **lift@K** = precision@K / base_rate — how much better than picking at random.
- **ROC-AUC** is kept as a secondary, comparable-to-literature reference only.

Everything operates on a *score* (higher = more likely positive); it does not
matter whether the score is a calibrated probability or a raw ranking, since all
metrics here are rank- or threshold-based.

Usage
-----
    from src.models.evaluate import evaluate, results_table

    row = evaluate(y_true, y_score, name="popularity")   # -> dict of metrics
    print(results_table([row_a, row_b, row_c]))          # -> comparison DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Shortlist sizes for precision/recall@K. Chosen around plausible annual
# acquisition-shortlist sizes for a publisher.
K_VALUES = (50, 100, 250)


def precision_recall_at_k(y_true, y_score, k: int) -> tuple[float, float]:
    """precision@k and recall@k for the top-k highest-scored items.

    Ties are broken by the stable descending sort order. ``k`` is clipped to the
    number of available items.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    k = min(k, n)
    if k == 0:
        return 0.0, 0.0

    # Top-k by score (descending). argsort is ascending, so negate.
    top_idx = np.argsort(-y_score, kind="stable")[:k]
    hits = int(y_true[top_idx].sum())
    total_pos = int(y_true.sum())

    precision = hits / k
    recall = hits / total_pos if total_pos else 0.0
    return precision, recall


def evaluate(y_true, y_score, name: str = "model", ks=K_VALUES) -> dict:
    """Compute the full metric bundle for one set of predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    base_rate = float(y_true.mean())

    row: dict[str, float | str] = {
        "model": name,
        "n": len(y_true),
        "positives": int(y_true.sum()),
        "base_rate": round(base_rate, 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
    }
    for k in ks:
        p, r = precision_recall_at_k(y_true, y_score, k)
        row[f"prec@{k}"] = round(p, 4)
        row[f"rec@{k}"] = round(r, 4)
        row[f"lift@{k}"] = round(p / base_rate, 2) if base_rate else 0.0
    return row


def results_table(rows: list[dict]) -> pd.DataFrame:
    """Stack several ``evaluate`` rows into one comparison table."""
    return pd.DataFrame(rows).set_index("model")


def pr_curve_points(y_true, y_score):
    """Return (recall, precision, thresholds) for plotting a PR curve."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return recall, precision, thresholds


# ── Self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Sanity-check the metrics on a synthetic 6%-positive ranking problem."""
    rng = np.random.default_rng(0)
    n, base = 3000, 0.06
    y = (rng.random(n) < base).astype(int)

    # A "useful" score: positives get a noisy boost over negatives.
    score_good = rng.normal(0, 1, n) + 1.5 * y
    # A useless score: pure noise.
    score_rand = rng.normal(0, 1, n)

    good = evaluate(y, score_good, name="good_ranker")
    rand = evaluate(y, score_rand, name="random")

    print(results_table([good, rand]).to_string())

    assert good["pr_auc"] > rand["pr_auc"], "good ranker should beat random on PR-AUC"
    assert good["lift@50"] > 1.0, "good ranker should have lift > 1 in its top-50"
    assert abs(rand["roc_auc"] - 0.5) < 0.05, "random ROC-AUC should be ~0.5"
    # Perfect ranking -> precision@k == 1 until positives run out.
    perfect = evaluate(y, y + rng.random(n) * 1e-6, name="perfect")
    assert perfect["prec@50"] == 1.0, "perfect ranking should have precision@50 == 1"
    print("\nself-test passed.")


if __name__ == "__main__":
    _selftest()
