"""Market regimes: a Gaussian HMM, a k-means baseline, and volatility-based label alignment.

Two ways of reading an HMM matter for leakage:
* `decode` (Viterbi) uses the whole sequence, so a label at t depends on the future. It is fine
  for *describing* history (the shaded price chart) and is never used as a model input.
* `filter` is the forward algorithm: P(state_t | x_1..x_t). It is causal, and is what the
  walk-forward models consume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_VOL_COLUMN = "rv21"

# hmmlearn logs a harmless "not converging" notice whenever EM stalls on a 1e-4 likelihood wobble.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def _vol_index(X: pd.DataFrame | np.ndarray, vol_col: str | int) -> int:
    if isinstance(vol_col, int):
        return vol_col
    cols = list(X.columns) if isinstance(X, pd.DataFrame) else []
    return cols.index(vol_col) if vol_col in cols else 0


@dataclass
class RegimeModel:
    """A fitted HMM with states re-numbered 0..k-1 in increasing order of volatility."""

    hmm: GaussianHMM
    scaler: StandardScaler
    order: np.ndarray  # order[new_label] = original hmm state
    columns: list[str] | None

    @property
    def k(self) -> int:
        return len(self.order)

    def _z(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.asarray(X, dtype=float))

    def decode(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Viterbi path (uses the full sequence, so NOT causal)."""
        raw = self.hmm.predict(self._z(X))
        inverse = np.empty(self.k, dtype=int)
        inverse[self.order] = np.arange(self.k)
        return inverse[raw]

    def filter(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Causal filtered state probabilities, shape (n, k), columns in aligned order."""
        z = self._z(X)
        h = self.hmm
        log_b = np.column_stack(
            [
                multivariate_normal.logpdf(
                    z, mean=h.means_[s], cov=h.covars_[s], allow_singular=True
                )
                for s in range(h.n_components)
            ]
        )
        log_a = np.log(np.clip(h.transmat_, 1e-300, None))
        alpha = np.log(np.clip(h.startprob_, 1e-300, None)) + log_b[0]
        out = np.empty_like(log_b)
        out[0] = alpha - logsumexp(alpha)
        for t in range(1, len(z)):
            alpha = logsumexp(out[t - 1][:, None] + log_a, axis=0) + log_b[t]
            out[t] = alpha - logsumexp(alpha)
        return np.exp(out)[:, self.order]


def fit_hmm(
    X: pd.DataFrame | np.ndarray,
    k: int = 3,
    seed: int = 0,
    n_init: int = 6,
    vol_col: str | int = DEFAULT_VOL_COLUMN,
) -> tuple[pd.Series | np.ndarray, RegimeModel]:
    """Fit a full-covariance Gaussian HMM (best of `n_init` restarts by log-likelihood).

    Returns Viterbi labels aligned so label 0 is the calmest regime and k-1 the most volatile,
    and the fitted `RegimeModel`.
    """
    scaler = StandardScaler().fit(np.asarray(X, dtype=float))
    z = scaler.transform(np.asarray(X, dtype=float))
    best, best_ll = None, -np.inf
    for i in range(n_init):
        hmm = GaussianHMM(
            n_components=k,
            covariance_type="full",
            n_iter=200,
            tol=1e-3,
            min_covar=1e-3,
            random_state=seed * 1000 + i,
            init_params="mc",
            params="stmc",
        )
        hmm.startprob_ = np.full(k, 1.0 / k)
        hmm.transmat_ = np.full((k, k), 0.02 / max(k - 1, 1)) + np.eye(k) * (
            0.98 - 0.02 / max(k - 1, 1)
        )
        try:
            hmm.fit(z)
            ll = hmm.score(z)
        except (ValueError, FloatingPointError):  # degenerate restart
            continue
        if ll > best_ll:
            best, best_ll = hmm, ll
    if best is None:
        raise RuntimeError("HMM failed to converge on every restart")
    order = np.argsort(best.means_[:, _vol_index(X, vol_col)])
    cols = list(X.columns) if isinstance(X, pd.DataFrame) else None
    model = RegimeModel(hmm=best, scaler=scaler, order=order, columns=cols)
    labels = model.decode(X)
    if isinstance(X, pd.DataFrame):
        labels = pd.Series(labels, index=X.index, name="regime")
    return labels, model


@dataclass
class KMeansRegimes:
    km: KMeans
    scaler: StandardScaler
    order: np.ndarray

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = self.km.predict(self.scaler.transform(np.asarray(X, dtype=float)))
        inverse = np.empty(len(self.order), dtype=int)
        inverse[self.order] = np.arange(len(self.order))
        return inverse[raw]


def fit_kmeans(
    X: pd.DataFrame | np.ndarray,
    k: int = 3,
    seed: int = 0,
    vol_col: str | int = DEFAULT_VOL_COLUMN,
) -> tuple[pd.Series | np.ndarray, KMeansRegimes]:
    """Memoryless baseline: cluster standardised features, ignoring time order entirely."""
    scaler = StandardScaler().fit(np.asarray(X, dtype=float))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(
        scaler.transform(np.asarray(X, dtype=float))
    )
    order = np.argsort(km.cluster_centers_[:, _vol_index(X, vol_col)])
    model = KMeansRegimes(km=km, scaler=scaler, order=order)
    labels = model.predict(X)
    if isinstance(X, pd.DataFrame):
        labels = pd.Series(labels, index=X.index, name="regime")
    return labels, model


def regime_probabilities(
    X_train: pd.DataFrame, X_test: pd.DataFrame, k: int = 3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on the training window only; return causal filtered probabilities for both blocks.

    The test block is filtered *continuing* from the end of training so the recursion is warm
    at the first test row. Only k-1 columns are returned (they sum to one).
    """
    _, model = fit_hmm(X_train, k=k, seed=seed)
    joined = model.filter(pd.concat([X_train, X_test]))
    n = len(X_train)
    return joined[:n, : k - 1], joined[n:, : k - 1]
