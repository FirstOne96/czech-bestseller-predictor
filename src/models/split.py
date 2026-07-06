"""
Temporal train / validation / test split for the acquisition-ranking model.

Why temporal (not random)
-------------------------
The deployment task is "given a foreign book a publisher is *considering today*,
will its Czech edition chart?" — i.e. predict the future from the past. A random
split would leak future information (and future books of the same author/series)
into training, inflating scores. We split strictly by ``czech_pub_year``.

Cohort: 2003–2017
-----------------
SCKN charts start in 2003. We cap the cohort at 2017 because books published
later suffer two artifacts that depress the positive rate (6.4% in 2016 ->
~3% in 2019):
  - **Feature drift**: the Goodreads dump is a 2017 snapshot, so pre-cutoff
    popularity is understated for later books.
  - **Label censoring**: recent SCKN labels are less complete.
Including 2018+ would make the test set look artificially harder for reasons
that have nothing to do with the model. See notebook 07 / the year table in
the thesis EDA.

Split
-----
  train : czech_pub_year <= 2014
  val   : czech_pub_year == 2015      (tuning, calibration, threshold choice)
  test  : czech_pub_year in {2016, 2017}   (reported once, at the very end)

Inputs (data/processed/)
------------------------
- X_train.csv  — 16 feature columns incl. czech_pub_year (the split key)
- y_train.csv  — sckn_bestseller label

Outputs (data/processed/splits/)
--------------------------------
- {X,y}_{train,val,test}.csv — the six split files, for reproducibility.

Usage
-----
    python src/models/split.py                 # write data/processed/splits/*.csv

    from src.models.split import load_splits
    s = load_splits()                          # -> Split bundle (reads from disk,
                                               #    rebuilds if missing)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
SPLITS = PROC / "splits"

# Split boundaries — single source of truth.
COHORT_MIN_YEAR = 2003
COHORT_MAX_YEAR = 2017
TRAIN_MAX_YEAR  = 2014
VAL_YEAR        = 2015
TEST_YEARS      = (2016, 2017)

YEAR_COL  = "czech_pub_year"
LABEL_COL = "sckn_bestseller"


@dataclass
class Split:
    """Container for the six split frames + a convenience summary."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, X, y in [
            ("train", self.X_train, self.y_train),
            ("val",   self.X_val,   self.y_val),
            ("test",  self.X_test,  self.y_test),
        ]:
            yr = X[YEAR_COL]
            rows.append({
                "split": name,
                "years": f"{int(yr.min())}–{int(yr.max())}",
                "n": len(X),
                "positives": int(y.sum()),
                "pos_rate": round(y.mean(), 4),
            })
        return pd.DataFrame(rows)


def feature_columns(X: pd.DataFrame, drop_year: bool = True) -> list[str]:
    """The columns a model should train on.

    ``czech_pub_year`` is excluded by default: it is the temporal-split key, and
    under that split every val/test row has a year strictly greater than any
    training year, so the model can only extrapolate on it (a linear model reads
    a spurious trend; a tree dumps all test rows into one edge leaf). Keeping the
    split honest means not letting the model lean on the calendar year. Pass
    ``drop_year=False`` only for diagnostics.
    """
    return [c for c in X.columns if not (drop_year and c == YEAR_COL)]


def load_Xy(proc: Path = PROC) -> tuple[pd.DataFrame, pd.Series]:
    """Load the full feature matrix + label, with the year column as int."""
    X = pd.read_csv(proc / "X_train.csv")
    y = pd.read_csv(proc / "y_train.csv")[LABEL_COL].astype(int)
    X[YEAR_COL] = pd.to_numeric(X[YEAR_COL], errors="coerce").astype("Int64")
    assert len(X) == len(y), f"X/y length mismatch: {len(X)} != {len(y)}"
    return X, y


def temporal_split(X: pd.DataFrame, y: pd.Series) -> Split:
    """Apply the cohort filter and split by ``czech_pub_year``."""
    in_cohort = X[YEAR_COL].between(COHORT_MIN_YEAR, COHORT_MAX_YEAR)
    X, y = X[in_cohort], y[in_cohort]

    train_m = X[YEAR_COL] <= TRAIN_MAX_YEAR
    val_m   = X[YEAR_COL] == VAL_YEAR
    test_m  = X[YEAR_COL].isin(TEST_YEARS)

    # Sanity: masks partition the cohort with no overlap and no leftovers.
    assert (train_m | val_m | test_m).all(), "cohort rows fell outside all splits"
    assert (train_m.astype(int) + val_m.astype(int) + test_m.astype(int)).max() == 1, \
        "split masks overlap"

    return Split(
        X_train=X[train_m].reset_index(drop=True), y_train=y[train_m].reset_index(drop=True),
        X_val=X[val_m].reset_index(drop=True),     y_val=y[val_m].reset_index(drop=True),
        X_test=X[test_m].reset_index(drop=True),   y_test=y[test_m].reset_index(drop=True),
    )


def save_splits(split: Split, splits_dir: Path = SPLITS) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    split.X_train.to_csv(splits_dir / "X_train.csv", index=False)
    split.y_train.to_frame(LABEL_COL).to_csv(splits_dir / "y_train.csv", index=False)
    split.X_val.to_csv(splits_dir / "X_val.csv", index=False)
    split.y_val.to_frame(LABEL_COL).to_csv(splits_dir / "y_val.csv", index=False)
    split.X_test.to_csv(splits_dir / "X_test.csv", index=False)
    split.y_test.to_frame(LABEL_COL).to_csv(splits_dir / "y_test.csv", index=False)


def load_splits(splits_dir: Path = SPLITS) -> Split:
    """Load splits from disk; rebuild them from the feature matrix if missing."""
    needed = [f"{m}_{s}.csv" for m in ("X", "y") for s in ("train", "val", "test")]
    if not all((splits_dir / f).exists() for f in needed):
        X, y = load_Xy()
        split = temporal_split(X, y)
        save_splits(split, splits_dir)
        return split

    def _y(name: str) -> pd.Series:
        return pd.read_csv(splits_dir / f"y_{name}.csv")[LABEL_COL].astype(int)

    return Split(
        X_train=pd.read_csv(splits_dir / "X_train.csv"), y_train=_y("train"),
        X_val=pd.read_csv(splits_dir / "X_val.csv"),     y_val=_y("val"),
        X_test=pd.read_csv(splits_dir / "X_test.csv"),   y_test=_y("test"),
    )


def main() -> Split:
    X, y = load_Xy()
    split = temporal_split(X, y)
    save_splits(split)
    print("Temporal split (cohort 2003–2017) written to data/processed/splits/\n")
    print(split.summary().to_string(index=False))
    return split


if __name__ == "__main__":
    main()
