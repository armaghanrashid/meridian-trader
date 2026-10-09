"""Next-21-day direction models, baselines and the walk-forward loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features, regimes, stats


class MajorityClass(ClassifierMixin, BaseEstimator):
    """Always predicts the class that was more frequent in the training window."""

    def fit(self, X, y):
        self.p1_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.tile([1.0 - self.p1_, self.p1_], (len(X), 1))

    def predict(self, X):
        return np.full(len(X), int(self.p1_ >= 0.5))


class AlwaysLong(ClassifierMixin, BaseEstimator):
    """Buy-and-hold in classifier clothing: always predicts 'up'."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.tile([0.0, 1.0], (len(X), 1))

    def predict(self, X):
        return np.ones(len(X), dtype=int)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], BaseEstimator]
    regimes: bool = False  # append causal HMM regime probabilities fitted on the train window


def _logreg() -> BaseEstimator:
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def default_specs(seed: int = 0) -> list[ModelSpec]:
    """Two baselines and four candidate models.

    Hyper-parameters are fixed a priori, not tuned: tuning on the validation folds would add
    selection bias that the deflated Sharpe would then have to pay for."""

    def gbm() -> BaseEstimator:
        return GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed
        )

    return [
        ModelSpec("majority", MajorityClass),
        ModelSpec("always_long", AlwaysLong),
        ModelSpec("logreg", _logreg),
        ModelSpec("gbm", gbm),
        ModelSpec("logreg+regime", _logreg, regimes=True),
        ModelSpec("gbm+regime", gbm, regimes=True),
    ]


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: ModelSpec,
    regime_cols: list[str] | None = None,
    k: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit on each train window, predict its test block. Everything learned (scaler, model,
    HMM) sees training rows only. Returns one row per out-of-sample date."""
    regime_cols = regime_cols or features.REGIME_COLUMNS
    out = []
    for i, (tr, te) in enumerate(folds):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr = y.iloc[tr].to_numpy()
        A, B = Xtr.to_numpy(), Xte.to_numpy()
        if spec.regimes:
            ptr, pte = regimes.regime_probabilities(Xtr[regime_cols], Xte[regime_cols], k, seed)
            A, B = np.hstack([A, ptr]), np.hstack([B, pte])
        model = spec.factory().fit(A, ytr)
        proba = model.predict_proba(B)[:, 1]
        out.append(
            pd.DataFrame(
                {
                    "proba": proba,
                    "pred": model.predict(B).astype(int),
                    "y": y.iloc[te].to_numpy(),
                    "fold": i,
                },
                index=X.index[te],
            )
        )
    return pd.concat(out)


def balanced_accuracy(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def summarize(preds: pd.DataFrame, horizon: int = 21, seed: int = 0, n_boot: int = 1000) -> dict:
    """Accuracy and balanced accuracy with moving-block-bootstrap 95% intervals.

    Blocks are `horizon` long: labels overlap for `horizon` days, so a shorter block would
    understate uncertainty."""
    arr = preds[["y", "pred"]].to_numpy(dtype=float)

    def acc(m):
        return float((m[:, 0] == m[:, 1]).mean())

    def bal(m):
        return balanced_accuracy(m[:, 0], m[:, 1])

    a, alo, ahi = stats.block_bootstrap_ci(arr, acc, horizon, n_boot, seed=seed)
    b, blo, bhi = stats.block_bootstrap_ci(arr, bal, horizon, n_boot, seed=seed)
    return {
        "n": len(preds),
        "accuracy": a,
        "accuracy_ci": (alo, ahi),
        "balanced_accuracy": b,
        "balanced_accuracy_ci": (blo, bhi),
        "long_rate": float(preds["pred"].mean()),
    }
