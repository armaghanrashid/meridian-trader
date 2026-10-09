"""Leakage diagnostics: a shuffled-label test and a planted future-feature canary.

Two complementary tools:
* the *canary* is a feature deliberately built from the future; `validation.find_lookahead_columns`
  must flag it, proving the timestamp check is not vacuous;
* the *shuffled-label test* destroys any real relationship between features and labels while
  keeping the pipeline intact. An honest pipeline must then score at chance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import models

CANARY_COLUMN = "canary_future"
IMPLAUSIBLE_BALANCED_ACCURACY = 0.75


def with_future_canary(
    build: Callable[[pd.DataFrame], pd.DataFrame], horizon: int = 21
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Wrap a feature builder so it also emits the realised forward return (a pure leak)."""

    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        out = build(df).copy()
        logp = np.log(df["SP500"])
        out[CANARY_COLUMN] = logp.shift(-horizon) - logp
        return out

    return leaky


def is_implausible(balanced_accuracy: float) -> bool:
    """Direction forecasts on daily data do not reach this level; treat it as a leak."""
    if not 0.0 <= balanced_accuracy <= 1.0:
        raise ValueError("balanced accuracy must lie in [0, 1]")
    return balanced_accuracy >= IMPLAUSIBLE_BALANCED_ACCURACY


def balanced_accuracy_null_ci(n_pos: int, n_neg: int, level: float = 0.95) -> tuple[float, float]:
    """Interval around 0.5 for the balanced accuracy of a label-independent predictor.

    BA = (TPR + TNR) / 2 with TPR ~ Bin(n_pos, .5)/n_pos and TNR ~ Bin(n_neg, .5)/n_neg, so
    sd(BA) = 0.25 * sqrt(1/n_pos + 1/n_neg). (Raw accuracy is *not* centred on 50% when the
    classes are imbalanced, which is why this project tests balanced accuracy.)
    """
    z = sps.norm.ppf(0.5 + level / 2.0)
    half = z * 0.25 * np.sqrt(1.0 / n_pos + 1.0 / n_neg)
    return 0.5 - half, 0.5 + half


@dataclass(frozen=True)
class ShuffleResult:
    balanced_accuracies: list[float]
    mean_balanced_accuracy: float
    ci_low: float
    ci_high: float
    passed: bool


def shuffled_label_test(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list,
    spec: models.ModelSpec,
    n_shuffles: int = 10,
    seed: int = 0,
    level: float = 0.95,
) -> ShuffleResult:
    """Re-run the whole walk-forward pipeline on permuted labels.

    Passes when the mean balanced accuracy lies inside the null interval around 50% (the
    interval for a *single* run, a deliberately conservative yardstick for the mean).
    """
    rng = np.random.default_rng(seed)
    scores, n_pos, n_neg = [], 0, 0
    for _ in range(n_shuffles):
        y_shuf = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        preds = models.walk_forward(X, y_shuf, folds, spec)
        scores.append(models.balanced_accuracy(preds["y"], preds["pred"]))
        n_pos, n_neg = int((preds["y"] == 1).sum()), int((preds["y"] == 0).sum())
    lo, hi = balanced_accuracy_null_ci(n_pos, n_neg, level)
    mean = float(np.mean(scores))
    return ShuffleResult(scores, mean, lo, hi, bool(lo <= mean <= hi))
