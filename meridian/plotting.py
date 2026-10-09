"""Figure style and plotting helpers. Dark panels read well on GitHub light and dark themes."""

from __future__ import annotations

from itertools import groupby
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

BG, PANEL, FG, MUTED, GRID = "#0d1117", "#11161d", "#e6edf3", "#8b949e", "#262c36"
REGIME_COLORS = ["#3fb950", "#d29922", "#f85149"]  # calm, neutral, stressed
REGIME_NAMES = ["Calm", "Neutral", "Stressed"]
SERIES_COLORS = {
    "logreg": "#58a6ff",
    "gbm": "#bc8cff",
    "logreg+regime": "#39c5cf",
    "gbm+regime": "#ff7b72",
    "always_long": "#e6edf3",
    "majority": "#8b949e",
}
WIDTH_PX, DPI = 1600, 160


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": FG,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": FG,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": FG,
            "lines.linewidth": 1.6,
            "figure.constrained_layout.use": True,
        }
    )


def new_figure(height_in: float, **kw):
    apply_style()
    return plt.subplots(figsize=(WIDTH_PX / DPI, height_in), dpi=DPI, **kw)


def save(fig, path: Path) -> Path:
    """Save at exactly 1600 px wide, then re-save through PIL to drop all metadata chunks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    with Image.open(path) as im:
        assert im.width == WIDTH_PX, f"{path.name} is {im.width}px wide"
        Image.fromarray(np.asarray(im)).save(path, optimize=True)
    return path


def regime_runs(labels: pd.Series):
    """Yield (start, end, label) for each maximal run of one regime."""
    pos = 0
    idx = labels.index
    for lab, grp in groupby(labels.to_numpy()):
        n = len(list(grp))
        yield idx[pos], idx[min(pos + n, len(idx) - 1)], int(lab)
        pos += n


def plot_regimes(price: pd.Series, labels: pd.Series, vol: pd.Series, title: str):
    fig, (ax, ax2) = new_figure(5.6, nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    k = int(labels.max()) + 1
    for a in (ax, ax2):
        for start, end, lab in regime_runs(labels):
            a.axvspan(start, end, color=REGIME_COLORS[lab % 3], alpha=0.20, lw=0)
    ax.plot(price.index, price.to_numpy(), color=FG, lw=1.3)
    ax.margins(x=0)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("S&P 500 (log scale)")
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_yticks([2500, 5000, 7500])
    ax2.plot(vol.index, vol.to_numpy() * 100, color="#58a6ff", lw=1.1)
    ax2.set_ylabel("21d realised vol (%)")
    share = labels.value_counts(normalize=True)
    handles = [
        Patch(
            color=REGIME_COLORS[i],
            alpha=0.5,
            label=f"{REGIME_NAMES[i]} ({share.get(i, 0):.0%} of days)",
        )
        for i in range(k)
    ]
    ax.legend(handles=handles, loc="upper left", ncols=k)
    return fig


def plot_accuracy(rows: pd.DataFrame, always_long_acc: float):
    """rows: index=model, columns acc, acc_lo, acc_hi, bal, bal_lo, bal_hi."""
    fig, (a1, a2) = new_figure(4.8, ncols=2, sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (v, lo, hi), ref, xl, ttl in (
        (
            a1,
            ("acc", "acc_lo", "acc_hi"),
            always_long_acc,
            "Accuracy",
            "Accuracy (95% block-bootstrap CI)",
        ),
        (
            a2,
            ("bal", "bal_lo", "bal_hi"),
            0.5,
            "Balanced accuracy",
            "Balanced accuracy (chance = 0.50)",
        ),
    ):
        for yi, (name, r) in zip(y, rows.iterrows(), strict=True):
            c = SERIES_COLORS.get(name, FG)
            ax.errorbar(
                r[v],
                yi,
                xerr=[[r[v] - r[lo]], [r[hi] - r[v]]],
                fmt="o",
                color=c,
                ecolor=c,
                capsize=3,
                ms=6,
            )
        ax.axvline(ref, color=MUTED, ls="--", lw=1)
        ax.set_xlabel(xl)
        ax.set_title(ttl, fontsize=11)
        ax.grid(axis="y", visible=False)
    a1.set_yticks(y, rows.index)
    return fig


def plot_costs(sharpes: dict[str, pd.Series], title: str):
    fig, ax = new_figure(4.8)
    for name, s in sharpes.items():
        c = SERIES_COLORS.get(name, FG)
        ax.plot(
            s.index,
            s.to_numpy(),
            color=c,
            lw=2.2 if name == "always_long" else 1.6,
            ls="--" if name == "always_long" else "-",
            marker="o",
            ms=3.5,
            label=name,
        )
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Cost per unit of position change (bps)")
    ax.set_ylabel("Net Sharpe (annualised)")
    ax.set_title(title)
    ax.legend(ncols=3, loc="lower left")
    return fig


def plot_equity(curves: dict[str, pd.Series], title: str):
    fig, ax = new_figure(4.8)
    for name, r in curves.items():
        c = SERIES_COLORS.get(name, FG)
        ax.plot(
            r.index,
            np.exp(r.cumsum()).to_numpy(),
            color=c,
            lw=2.2 if name == "always_long" else 1.4,
            ls="--" if name == "always_long" else "-",
            label=name,
        )
    ax.set_ylabel("Growth of 1.0 (net of costs)")
    ax.set_title(title)
    ax.legend(ncols=3, loc="upper left")
    return fig
