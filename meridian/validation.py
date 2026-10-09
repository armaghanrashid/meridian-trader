"""Leakage-safe validation: purged walk-forward splits and a timestamp causality check."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    """Raised when a split or a feature violates temporal causality."""


Fold = tuple[np.ndarray, np.ndarray]


def purged_walk_forward(
    n: int,
    train: int,
    test: int,
    embargo: int,
    horizon: int = 0,
    step: int | None = None,
    expanding: bool = False,
) -> list[Fold]:
    """Walk-forward folds over positions 0..n-1.

    Layout of one fold:  [ train window | embargo gap | test block ]
    * `embargo` positions separate the end of the train window from the test block.
    * Because the label of row t is built from data through t+horizon, any train row with
      t + horizon >= test_start is *purged*: its label would overlap the test period.
    Test blocks tile the series (step defaults to `test`), so concatenated out-of-sample
    predictions form one contiguous, non-repeating track.
    """
    if min(train, test) <= 0 or embargo < 0 or horizon < 0:
        raise ValueError("train/test must be positive; embargo/horizon non-negative")
    step = step or test
    if train + embargo + test > n:
        raise ValueError(f"series of length {n} too short for train+embargo+test")
    folds: list[Fold] = []
    start = 0
    while start + train + embargo + test <= n:
        test_start = start + train + embargo
        origin = 0 if expanding else start
        tr = np.arange(origin, start + train)
        tr = tr[tr + horizon < test_start]  # purge
        te = np.arange(test_start, test_start + test)
        folds.append((tr, te))
        start += step
    return folds


def assert_no_overlap(train_idx: np.ndarray, test_idx: np.ndarray, horizon: int, embargo: int):
    """Raise LeakageError unless the fold is purged and embargoed."""
    if len(np.intersect1d(train_idx, test_idx)):
        raise LeakageError("train and test share rows")
    if train_idx.max() >= test_idx.min():
        raise LeakageError("train extends into the test period")
    if train_idx.max() + horizon >= test_idx.min():
        raise LeakageError("a train label overlaps the test window (not purged)")
    if test_idx.min() - train_idx.max() - 1 < embargo:
        raise LeakageError("gap between train and test is smaller than the embargo")


def find_lookahead_columns(
    build: Callable[[pd.DataFrame], pd.DataFrame],
    df: pd.DataFrame,
    n_probes: int = 12,
    warmup: int = 100,
    seed: int = 0,
    rtol: float = 1e-9,
) -> list[str]:
    """Feature/target timestamp check by perturbation.

    A causal feature at timestamp t depends only on data stamped <= t. For random probe times t
    we corrupt every observation after t and rebuild the features: any column whose values at
    or before t move has consumed information from the future and is returned.
    """
    rng = np.random.default_rng(seed)
    base = build(df)
    probes = rng.integers(warmup, len(df) - 2, size=n_probes)
    num = df.select_dtypes("float").columns
    flagged: set[str] = set()
    for t in probes:
        corrupted = df.copy()
        shape = corrupted.iloc[t + 1 :][num].shape
        corrupted.loc[corrupted.index[t + 1 :], num] = corrupted.iloc[t + 1 :][
            num
        ].to_numpy() * rng.uniform(0.5, 1.5, size=shape)
        rebuilt = build(corrupted)
        for col in base.columns:
            a, b = base[col].iloc[: t + 1], rebuilt[col].iloc[: t + 1]
            same_nan = a.isna().equals(b.isna())
            if not same_nan or not np.allclose(a.dropna(), b.dropna(), rtol=rtol, atol=1e-12):
                flagged.add(col)
    return [c for c in base.columns if c in flagged]
