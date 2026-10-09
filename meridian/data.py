"""Data access: FRED public CSVs (no API key) with an on-disk cache, plus a synthetic world.

The synthetic generator is a three-state Markov regime-switching geometric Brownian motion.
Its regimes are *planted*, so tests can check that the regime machinery recovers them and that
the walk-forward pipeline finds an edge when one genuinely exists (a positive control).
"""

from __future__ import annotations

import io
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DEFAULT_SERIES = ["SP500", "VIXCLS", "DGS10", "DGS2"]
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("MERIDIAN_CACHE_DIR", ROOT / "data" / "cache"))
# Short publication gaps in rate/vol series (holidays differ across series) are carried forward;
# a gap longer than this many business days is left missing rather than invented.
MAX_FFILL = 5


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse one FRED `fredgraph.csv` body. FRED marks missing observations with '.'."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col = df.columns[0]
    s = pd.Series(
        pd.to_numeric(df[series_id], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(df[date_col]), name="date"),
        name=series_id,
    )
    return s.sort_index()


def fetch_fred(series_id: str, retries: int = 3, timeout: float = 30.0) -> str:
    """Download one series. One id per request: multi-id requests are returned as a zip."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(
                FRED_URL, params={"id": series_id, "cosd": "1990-01-01"}, timeout=timeout
            )
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"could not download FRED series {series_id}") from last


def load(
    series: Sequence[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    cache_dir: Path | str | None = None,
    fetch: Callable[[str], str] = fetch_fred,
) -> pd.DataFrame:
    """Load FRED series into one frame indexed by trading day.

    Each series is cached as `<cache_dir>/<ID>.csv`. Days without an `SP500` print are dropped
    (they are not trading days); other series are forward-filled for at most MAX_FFILL days.
    """
    cache = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    cols = []
    for sid in series:
        path = cache / f"{sid}.csv"
        if not path.exists():
            path.write_text(fetch(sid))
        cols.append(parse_fred_csv(path.read_text(), sid))
    df = pd.concat(cols, axis=1).sort_index()
    if "SP500" in df.columns:
        df = df[df["SP500"].notna()]
    df = df.ffill(limit=MAX_FFILL)
    if start is not None:
        df = df.loc[pd.Timestamp(start) :]
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]
    return df.dropna(how="all")


# Planted regimes, ordered by volatility: calm bull, choppy neutral, stressed bear.
_MU = np.array([0.80, 0.10, -2.00])  # annualised drift (deliberately strong: a positive control)
_SIGMA = np.array([0.08, 0.18, 0.40])  # annualised volatility
_TRANS = np.array(
    [
        [0.992, 0.008, 0.000],
        [0.006, 0.988, 0.006],
        [0.000, 0.020, 0.980],
    ]
)


def synthetic(seed: int = 0, n: int = 2000) -> pd.DataFrame:
    """Regime-switching GBM with the same columns as `load`, plus the planted `true_regime`.

    VIX is the regime's volatility (x100) with multiplicative noise; rates are mean-reverting.
    """
    rng = np.random.default_rng(seed)
    state = np.empty(n, dtype=int)
    state[0] = 0
    u = rng.random(n)
    for t in range(1, n):
        state[t] = np.searchsorted(np.cumsum(_TRANS[state[t - 1]]), u[t])
    state = np.minimum(state, 2)

    dt = 1.0 / 252.0
    mu, sig = _MU[state], _SIGMA[state]
    log_ret = (mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * rng.standard_normal(n)
    price = 2000.0 * np.exp(np.cumsum(log_ret))
    vix = 100.0 * sig * np.exp(0.12 * rng.standard_normal(n))

    def ou(level: float, vol: float) -> np.ndarray:
        x = np.empty(n)
        x[0] = level
        for t in range(1, n):
            x[t] = x[t - 1] + 0.01 * (level - x[t - 1]) + vol * rng.standard_normal()
        return x

    dgs2 = np.clip(ou(2.0, 0.03), 0.05, None)
    dgs10 = np.clip(dgs2 + ou(1.0, 0.02), 0.1, None)
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame(
        {"SP500": price, "VIXCLS": vix, "DGS10": dgs10, "DGS2": dgs2, "true_regime": state},
        index=idx,
    )
