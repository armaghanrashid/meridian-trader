"""Feature and target construction.

Every feature at row t is a function of rows <= t only. That property is enforced by tests
(truncation invariance) and by `validation.find_lookahead_columns`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
FEATURE_COLUMNS = [
    "ret1",
    "rv21",
    "rv63",
    "mom21",
    "mom63",
    "drawdown",
    "curve",
    "vix",
    "vix_chg5",
]
# Columns the regime model sees: realised vol, implied vol (stress) and trend.
REGIME_COLUMNS = ["rv21", "vix", "mom21"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Strictly backward-looking features from price, VIX and the Treasury curve."""
    logp = np.log(df["SP500"])
    ret1 = logp.diff()
    out = pd.DataFrame(index=df.index)
    out["ret1"] = ret1
    out["rv21"] = ret1.rolling(21).std() * np.sqrt(TRADING_DAYS)
    out["rv63"] = ret1.rolling(63).std() * np.sqrt(TRADING_DAYS)
    out["mom21"] = logp.diff(21)
    out["mom63"] = logp.diff(63)
    out["drawdown"] = df["SP500"] / df["SP500"].cummax() - 1.0
    out["curve"] = df["DGS10"] - df["DGS2"]
    out["vix"] = df["VIXCLS"]
    out["vix_chg5"] = df["VIXCLS"].diff(5)
    return out[FEATURE_COLUMNS]


def targets(df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Forward-looking quantities. Only ever used as labels / realised P&L, never as inputs.

    fwd_ret  log return from t to t+horizon;  y  1 if fwd_ret > 0 (NaN where unobservable);
    next_ret log return from t to t+1, the P&L earned by a position chosen at the close of t.
    """
    logp = np.log(df["SP500"])
    fwd = logp.shift(-horizon) - logp
    y = (fwd > 0).astype(float).where(fwd.notna())
    return pd.DataFrame({"fwd_ret": fwd, "y": y, "next_ret": logp.shift(-1) - logp})


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    fwd_ret: pd.Series
    next_ret: pd.Series
    horizon: int


def dataset(df: pd.DataFrame, horizon: int = 21) -> Dataset:
    """Aligned, NaN-free design matrix and labels. Rows drop only at the warm-up head and the
    unlabelled tail, so the remaining index is contiguous in trading days."""
    X = build(df)
    tg = targets(df, horizon)
    ok = X.notna().all(axis=1) & tg.notna().all(axis=1)
    return Dataset(
        X=X[ok],
        y=tg.loc[ok, "y"].astype(int),
        fwd_ret=tg.loc[ok, "fwd_ret"],
        next_ret=tg.loc[ok, "next_ret"],
        horizon=horizon,
    )
