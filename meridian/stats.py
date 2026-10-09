"""Inference utilities: block bootstrap, Sharpe, deflated Sharpe, turnover and cost sensitivity.

References: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection
Bias, Backtest Overfitting and Non-Normality"; Kunsch (1989) for the moving-block bootstrap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.diagnostic import acorr_ljungbox

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def sharpe(returns: np.ndarray | pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio of per-period returns (risk-free rate taken as zero)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0.0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def max_drawdown(log_returns: np.ndarray | pd.Series) -> float:
    """Worst peak-to-trough fall of the equity curve built from log returns (a negative number)."""
    equity = np.exp(np.cumsum(np.asarray(log_returns, dtype=float)))
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return float((equity / peak - 1.0).min())


def block_bootstrap_ci(
    data: np.ndarray | pd.Series,
    stat: Callable[[np.ndarray], float],
    block_len: int,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Moving-block bootstrap percentile interval -> (point, low, high).

    Overlapping 21-day labels and volatility clustering make observations dependent, so an
    i.i.d. bootstrap would be too narrow. `data` may be 1-D or (n, d) for paired statistics;
    rows are resampled together.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    block_len = max(1, min(int(block_len), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)).reshape(n_boot, -1)[:, :n]
    reps = np.array([stat(x[i]) for i in idx])
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [alpha, 1.0 - alpha])
    return float(stat(x)), float(lo), float(hi)


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n: int, skew: float, kurt: float
) -> float:
    """P(true SR > benchmark) given an observed per-period SR over n observations.

    `kurt` is the plain (non-excess) kurtosis, 3 for a normal distribution.
    """
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    return float(sps.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum per-period SR of n_trials skill-less strategies (False Strategy Theorem)."""
    if n_trials <= 1:
        return 0.0
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    sr: float, sr_variance: float, n_trials: int, n: int, skew: float, kurt: float
) -> float:
    """Probability the strategy's SR is genuinely positive after correcting for having picked it
    from `n_trials` candidates (`sr_variance` is the variance of per-period SR across them)."""
    benchmark = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(sr, benchmark, n, skew, kurt)


def turnover(positions: np.ndarray | pd.Series) -> float:
    """Mean absolute daily position change, starting flat (so the initial entry is counted)."""
    p = np.asarray(positions, dtype=float)
    return float(np.abs(np.diff(np.concatenate([[0.0], p]))).mean())


def net_returns(
    positions: np.ndarray | pd.Series, next_ret: np.ndarray | pd.Series, cost_bps: float
) -> np.ndarray:
    """Per-day P&L (log-return units) of a position chosen at the close, less trading costs
    of `cost_bps` per unit of position change."""
    p = np.asarray(positions, dtype=float)
    traded = np.abs(np.diff(np.concatenate([[0.0], p])))
    return p * np.asarray(next_ret, dtype=float) - traded * cost_bps / 1e4


def cost_sensitivity(
    positions: np.ndarray | pd.Series,
    next_ret: np.ndarray | pd.Series,
    bps_grid: Sequence[float],
) -> pd.DataFrame:
    """Sharpe and annualised return as the per-trade cost rises. Indexed by bps."""
    rows = {}
    for bps in bps_grid:
        r = net_returns(positions, next_ret, bps)
        rows[bps] = {"sharpe": sharpe(r), "ann_return": float(r.mean() * TRADING_DAYS)}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "cost_bps"
    return out


def ljung_box_pvalue(x: np.ndarray | pd.Series, lags: int = 10) -> float:
    """p-value of the Ljung-Box test for serial correlation (small = dependence is present)."""
    res = acorr_ljungbox(np.asarray(x, dtype=float), lags=[lags], return_df=True)
    return float(res["lb_pvalue"].iloc[0])
