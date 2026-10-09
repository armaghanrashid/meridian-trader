"""Build, execute and clean notebooks/01_explore.ipynb.

Run from the repo root: `make notebook`. Only the final figures keep their outputs; every
other output (tables, logs, paths) is cleared so the committed notebook is small and clean.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

CELLS = [
    (
        "md",
        "# Exploring regimes in the S&P 500\n\nA short, honest walk through the data before any "
        "modelling. Everything here is descriptive and in-sample. The predictive study, with purged "
        "walk-forward validation, lives in `meridian/report.py`. Research, not financial advice.",
    ),
    (
        "code",
        "import sys, tempfile\nfrom pathlib import Path\n\nsys.path.insert(0, '..')\n"
        "import numpy as np\nimport pandas as pd\nfrom IPython.display import Image, display\n\n"
        "from meridian import data, features, plotting, regimes, report, stats\n\n"
        "df = data.load(data.DEFAULT_SERIES, '1990-01-01')\n"
        "print(df.index.min().date(), df.index.max().date(), len(df))\ndf.tail()",
    ),
    (
        "md",
        "## Features are backward-looking only\n\nEvery column at row *t* is computed from rows "
        "up to *t*. Targets look forward and are kept in a separate frame.",
    ),
    ("code", "X = features.build(df)\nX.describe().T[['mean', 'std', 'min', 'max']].round(3)"),
    (
        "md",
        "## Why regimes: volatility clusters\n\nA Ljung-Box test on daily returns versus squared "
        "returns. Both reject independence, but the squared-returns statistic is far larger: the size "
        "of moves is more persistent than their direction, which is what a regime-switching "
        "volatility model is built to capture.",
    ),
    (
        "code",
        "r = np.log(df['SP500']).diff().dropna()\n"
        "print('returns         Q, p =', stats.ljung_box(r))\n"
        "print('squared returns Q, p =', stats.ljung_box(r**2))",
    ),
    (
        "md",
        "## Fitting the regimes\n\nA three-state Gaussian HMM on realised volatility, VIX and "
        "21-day momentum. States are ordered by volatility. The labels shown here are Viterbi-smoothed "
        "over the full sample, so they use hindsight: fine for a picture, never an input to a model.",
    ),
    (
        "code",
        "Xr = X[features.REGIME_COLUMNS].dropna()\n"
        "labels, model = regimes.fit_hmm(Xr, k=3, seed=0)\n"
        "ret = r.loc[labels.index]\n"
        "pd.DataFrame({\n"
        "    'share': labels.value_counts(normalize=True).sort_index(),\n"
        "    'ann_vol': ret.groupby(labels).std() * np.sqrt(252),\n"
        "    'avg_spell_days': report.run_lengths(labels),\n"
        "}).round(3)",
    ),
    (
        "code",
        "out = Path(tempfile.mkdtemp())\n"
        "fig = plotting.plot_regimes(df['SP500'].loc[labels.index], labels, X['rv21'].loc[labels.index],\n"
        "                            'S&P 500 by volatility regime (Gaussian HMM, in-sample Viterbi)')\n"
        "display(Image(filename=plotting.save(fig, out / 'regimes.png')))",
    ),
    (
        "md",
        "## What the regimes imply for the next month\n\nForward 21-day returns by regime. The "
        "target here is the *label* the models later try to predict, so this is a description of the "
        "problem and not a tradable rule.",
    ),
    (
        "code",
        "fwd = features.targets(df, 21)['fwd_ret'].loc[labels.index].dropna()\n"
        "fig, ax = plotting.new_figure(4.4)\n"
        "for k in range(3):\n"
        "    v = fwd[labels.loc[fwd.index] == k] * 100\n"
        "    ax.hist(v, bins=40, alpha=0.55, color=plotting.REGIME_COLORS[k],\n"
        "            label=f'{plotting.REGIME_NAMES[k]}: mean {v.mean():+.1f}%, P(up) {(v > 0).mean():.0%}')\n"
        "ax.axvline(0, color=plotting.MUTED, lw=1)\n"
        "ax.set_xlabel('Forward 21-day log return (%)'); ax.set_ylabel('Days')\n"
        "ax.set_title('Next-month return distribution by regime (in-sample)')\nax.legend()\n"
        "display(Image(filename=plotting.save(fig, out / 'forward.png')))",
    ),
]


def build() -> Path:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in CELLS
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(HERE)}}).execute()
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = [
            o
            for o in cell.outputs
            if o.output_type in {"display_data", "execute_result"}
            and "image/png" in o.get("data", {})
        ]
        for o in cell.outputs:
            o["data"] = {"image/png": o["data"]["image/png"]}
            o["metadata"] = {}
            if o.output_type == "execute_result":
                o["execution_count"] = None
        cell.execution_count = None
        cell.metadata = {}
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    path = HERE / "01_explore.ipynb"
    nbformat.write(nb, path)
    return path


if __name__ == "__main__":
    print("wrote", build().name)
