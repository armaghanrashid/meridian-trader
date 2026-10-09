"""End-to-end study: python -m meridian.report  (or `make report`).

Loads FRED data, fits regimes, runs the leakage checks and the purged walk-forward comparison,
then writes reports/report.md, reports/results.json and reports/figures/*.png (copied to
docs/media/ for the README). Conclusions are generated from the numbers, so a null result is
reported as a null result.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import adjusted_rand_score

from . import data, features, leakage, models, plotting, regimes, stats, validation

ROOT = Path(__file__).resolve().parent.parent
TRIALS = ["logreg", "gbm", "logreg+regime", "gbm+regime"]  # candidates that could be "selected"


@dataclass(frozen=True)
class Config:
    horizon: int = 21
    train: int = 756  # ~3 years
    test: int = 126  # ~6 months
    embargo: int = 5
    k: int = 3
    seed: int = 0
    base_cost_bps: float = 5.0
    bps_grid: tuple[float, ...] = (0, 1, 2, 5, 10, 15, 25, 50)
    n_boot: int = 2000
    n_shuffles: int = 20
    trial_counts: tuple[int, ...] = (4, 10, 50)
    series: list[str] = field(default_factory=lambda: list(data.DEFAULT_SERIES))


def run_lengths(labels: pd.Series) -> pd.Series:
    """Average spell length (trading days) of each regime."""
    spells: dict[int, list[int]] = {}
    for lab, grp in groupby(labels.to_numpy()):
        spells.setdefault(int(lab), []).append(len(list(grp)))
    return pd.Series({k: float(np.mean(v)) for k, v in sorted(spells.items())})


def regime_table(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    r = np.log(df["SP500"]).diff().loc[labels.index]
    g = r.groupby(labels)
    return pd.DataFrame(
        {
            "share_of_days": labels.value_counts(normalize=True).sort_index(),
            "ann_return": g.mean() * features.TRADING_DAYS,
            "ann_vol": g.std() * np.sqrt(features.TRADING_DAYS),
            "avg_spell_days": run_lengths(labels),
        }
    )


def evaluate(df: pd.DataFrame, cfg: Config) -> dict:
    """Run the whole study and return every number the report quotes."""
    out: dict = {"config": {k: v for k, v in cfg.__dict__.items() if k != "series"}}
    ds = features.dataset(df, cfg.horizon)
    n = len(ds.X)
    out["data"] = {
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "n_days": len(df),
        "n_samples": n,
    }

    # ---- descriptive regimes (in-sample, smoothed: for the picture and the table only)
    Xr = features.build(df)[features.REGIME_COLUMNS].dropna()
    hmm_labels, _ = regimes.fit_hmm(Xr, k=cfg.k, seed=cfg.seed)
    km_labels, _ = regimes.fit_kmeans(Xr, k=cfg.k, seed=cfg.seed)
    ret1 = np.log(df["SP500"]).diff().dropna()
    out["regimes"] = {
        "hmm": regime_table(df, hmm_labels).round(4).to_dict(orient="index"),
        "kmeans": regime_table(df, km_labels).round(4).to_dict(orient="index"),
        "ari_hmm_vs_kmeans": float(adjusted_rand_score(hmm_labels, km_labels)),
        "ljung_box_q_returns": stats.ljung_box(ret1)[0],
        "ljung_box_q_squared_returns": stats.ljung_box(ret1**2)[0],
        "ljung_box_p_returns": stats.ljung_box(ret1)[1],
        "ljung_box_p_squared_returns": stats.ljung_box(ret1**2)[1],
    }

    # ---- splits and leakage checks
    folds = validation.purged_walk_forward(n, cfg.train, cfg.test, cfg.embargo, cfg.horizon)
    for tr, te in folds:
        validation.assert_no_overlap(tr, te, cfg.horizon, cfg.embargo)
    leaky = leakage.with_future_canary(features.build, cfg.horizon)
    specs = {s.name: s for s in models.default_specs(cfg.seed)}
    shuffle = leakage.shuffled_label_test(
        ds.X, ds.y, folds, specs["logreg"], n_shuffles=cfg.n_shuffles, seed=cfg.seed
    )
    canary_preds = models.walk_forward(
        ds.X.assign(**{leakage.CANARY_COLUMN: ds.fwd_ret}), ds.y, folds, specs["logreg"]
    )
    out["leakage"] = {
        "n_folds": len(folds),
        "split_integrity_checked_folds": len(folds),
        "features_flagged_by_timestamp_check": validation.find_lookahead_columns(
            features.build, df
        ),
        "canary_flagged_by_timestamp_check": validation.find_lookahead_columns(leaky, df),
        "shuffled_label_balanced_accuracy": shuffle.mean_balanced_accuracy,
        "shuffled_label_null_ci": [shuffle.ci_low, shuffle.ci_high],
        "shuffled_label_passed": shuffle.passed,
        "canary_model_balanced_accuracy": models.balanced_accuracy(
            canary_preds["y"], canary_preds["pred"]
        ),
    }

    # ---- walk-forward predictions, accuracy
    preds = {
        name: models.walk_forward(ds.X, ds.y, folds, s, seed=cfg.seed) for name, s in specs.items()
    }
    summary = {
        name: models.summarize(p, cfg.horizon, cfg.seed, cfg.n_boot) for name, p in preds.items()
    }
    long_pred = preds["always_long"]["pred"].to_numpy(dtype=float)
    y_oos = preds["always_long"]["y"].to_numpy(dtype=float)
    for name, p in preds.items():
        arr = np.column_stack([y_oos, p["pred"].to_numpy(dtype=float), long_pred])
        point, lo, hi = stats.block_bootstrap_ci(
            arr,
            lambda m: float((m[:, 0] == m[:, 1]).mean() - (m[:, 0] == m[:, 2]).mean()),
            cfg.horizon,
            cfg.n_boot,
            seed=cfg.seed,
        )
        summary[name]["acc_minus_always_long"] = [point, lo, hi]
    out["accuracy"] = summary

    # ---- strategies: long when the model says up, flat otherwise
    idx = preds["always_long"].index
    nxt = ds.next_ret.loc[idx].to_numpy()
    pos = {name: p["pred"].to_numpy(dtype=float) for name, p in preds.items()}
    base = cfg.base_cost_bps
    rets = {name: stats.net_returns(pos[name], nxt, base) for name in pos}
    bh = rets["always_long"]
    daily_sr = {name: float(r.mean() / r.std(ddof=1)) for name, r in rets.items()}
    sr_var = float(np.var([daily_sr[t] for t in TRIALS], ddof=1))
    strat = {}
    for name, r in rets.items():
        _, dlo, dhi = stats.block_bootstrap_ci(
            np.column_stack([r, bh]),
            lambda m: stats.sharpe(m[:, 0]) - stats.sharpe(m[:, 1]),
            cfg.horizon,
            cfg.n_boot,
            seed=cfg.seed,
        )
        skew, kurt = float(sps.skew(r)), float(sps.kurtosis(r, fisher=False))
        strat[name] = {
            "net_sharpe": stats.sharpe(r),
            "gross_sharpe": stats.sharpe(pos[name] * nxt),
            "ann_return": float(r.mean() * features.TRADING_DAYS),
            "max_drawdown": stats.max_drawdown(r),
            "turnover_per_day": stats.turnover(pos[name]),
            "time_in_market": float(pos[name].mean()),
            "sharpe_minus_buy_hold_ci": [stats.sharpe(r) - stats.sharpe(bh), dlo, dhi],
            "psr_vs_zero": stats.probabilistic_sharpe_ratio(
                daily_sr[name], 0.0, len(r), skew, kurt
            ),
            "dsr": {
                str(nt): stats.deflated_sharpe_ratio(daily_sr[name], sr_var, nt, len(r), skew, kurt)
                for nt in cfg.trial_counts
            },
        }
    out["strategies"] = strat
    out["strategy_meta"] = {
        "oos_days": len(idx),
        "oos_start": str(idx.min().date()),
        "oos_end": str(idx.max().date()),
        "base_cost_bps": base,
        "daily_sr_variance_across_trials": sr_var,
        "n_trials_main": len(TRIALS),
    }
    costs = {n_: stats.cost_sensitivity(pos[n_], nxt, cfg.bps_grid) for n_ in pos}
    out["cost_sensitivity"] = {n_: t["sharpe"].round(4).to_dict() for n_, t in costs.items()}

    out["_frames"] = {  # not serialised
        "price": df["SP500"].loc[hmm_labels.index],
        "labels": hmm_labels,
        "vol": features.build(df)["rv21"].loc[hmm_labels.index],
        "costs": {n_: t["sharpe"] for n_, t in costs.items()},
        "equity": {n_: pd.Series(r, index=idx) for n_, r in rets.items()},
    }
    return out


def verdict(res: dict) -> list[str]:
    """Plain-language conclusions derived from the numbers (never hard-coded)."""
    acc, strat = res["accuracy"], res["strategies"]
    cfg = res["config"]
    lines = []
    best = max(TRIALS, key=lambda m: strat[m]["net_sharpe"])
    s, a = strat[best], acc[best]
    bh = strat["always_long"]
    d_acc = a["acc_minus_always_long"]
    d_sh = s["sharpe_minus_buy_hold_ci"]
    dsr_main = s["dsr"][str(res["strategy_meta"]["n_trials_main"])]
    any_acc = [m for m in TRIALS if acc[m]["acc_minus_always_long"][1] > 0]
    any_sh = [m for m in TRIALS if strat[m]["sharpe_minus_buy_hold_ci"][1] > 0]
    reliable = bool(any_sh) and dsr_main >= 0.95
    lines.append(
        f"Best candidate by net Sharpe at {res['strategy_meta']['base_cost_bps']:g} bps: `{best}` "
        f"(net Sharpe {s['net_sharpe']:.2f} vs {bh['net_sharpe']:.2f} for buy-and-hold). "
        f"Its accuracy is {a['accuracy']:.3f} against {acc['always_long']['accuracy']:.3f} for "
        f"always-long ({d_acc[0]:+.3f}, 95% CI {d_acc[1]:+.3f} to {d_acc[2]:+.3f})."
    )
    lines.append(
        f"Sharpe difference vs buy-and-hold: {d_sh[0]:+.2f} (95% CI {d_sh[1]:+.2f} to {d_sh[2]:+.2f}). "
        f"Deflated Sharpe ratio with {res['strategy_meta']['n_trials_main']} trials: {dsr_main:.2f}."
    )
    if reliable:
        lines.append(
            "Result: at least one candidate shows a Sharpe advantage over buy-and-hold whose "
            "interval excludes zero and whose deflated Sharpe is 0.95 or higher. Treat this as a "
            "hypothesis to replicate on other data, not a finding: one market, one decade, and a "
            "single hold-out path."
        )
    else:
        lines.append(
            "Result: no statistically reliable edge. "
            + (
                "Accuracy edges over always-long exist for "
                + ", ".join(f"`{m}`" for m in any_acc)
                + " but do not survive translation into a cost-adjusted, deflated Sharpe."
                if any_acc and not any_sh
                else "No candidate beats always-long on accuracy with an interval excluding zero."
                if not any_acc
                else "The Sharpe advantage is not distinguishable from zero after deflation."
            )
            + f" The best candidate is long on {s['time_in_market']:.0%} of days, so most of its"
            " return is the equity risk premium that buy-and-hold already collects."
        )
    sh = res["leakage"]
    lines.append(
        f"Leakage checks: the shuffled-label balanced accuracy was {sh['shuffled_label_balanced_accuracy']:.3f} "
        f"(null interval {sh['shuffled_label_null_ci'][0]:.3f} to {sh['shuffled_label_null_ci'][1]:.3f}, "
        f"{'passed' if sh['shuffled_label_passed'] else 'FAILED'}); the planted future-feature canary "
        f"{'was flagged' if sh['canary_flagged_by_timestamp_check'] else 'WAS NOT FLAGGED'} by the "
        f"timestamp check and a model fed it reached {sh['canary_model_balanced_accuracy']:.3f} balanced accuracy "
        f"(the signature of leakage); the real features were "
        f"{'clean' if not sh['features_flagged_by_timestamp_check'] else 'FLAGGED'}."
    )
    _ = cfg
    return lines


def _fmt_p(p: float) -> str:
    return "< 1e-300" if p < 1e-300 else f"{p:.1e}"


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    head = "| " + " | ".join([df.index.name or ""] + cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    rows = []
    for idx, r in df.iterrows():
        cells = [str(idx)] + [
            format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in r
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def render_markdown(res: dict) -> str:
    cfg, d, acc, st = res["config"], res["data"], res["accuracy"], res["strategies"]
    meta, lk, rg = res["strategy_meta"], res["leakage"], res["regimes"]
    names = ["majority", "always_long", "logreg", "gbm", "logreg+regime", "gbm+regime"]

    acc_tbl = pd.DataFrame(
        {
            "accuracy": [acc[n]["accuracy"] for n in names],
            "95% CI": [
                f"{acc[n]['accuracy_ci'][0]:.3f} to {acc[n]['accuracy_ci'][1]:.3f}" for n in names
            ],
            "balanced acc.": [acc[n]["balanced_accuracy"] for n in names],
            "95% CI ": [
                f"{acc[n]['balanced_accuracy_ci'][0]:.3f} to {acc[n]['balanced_accuracy_ci'][1]:.3f}"
                for n in names
            ],
            "acc. minus always-long (95% CI)": [
                "-"
                if n == "always_long"
                else f"{acc[n]['acc_minus_always_long'][0]:+.3f} ({acc[n]['acc_minus_always_long'][1]:+.3f} to {acc[n]['acc_minus_always_long'][2]:+.3f})"
                for n in names
            ],
            "% days long": [f"{acc[n]['long_rate']:.0%}" for n in names],
        },
        index=pd.Index(names, name="model"),
    )
    st_tbl = pd.DataFrame(
        {
            "net Sharpe": [st[n]["net_sharpe"] for n in names],
            "gross Sharpe": [st[n]["gross_sharpe"] for n in names],
            "ann. return": [f"{st[n]['ann_return']:.1%}" for n in names],
            "max drawdown": [f"{st[n]['max_drawdown']:.1%}" for n in names],
            "turnover/day": [st[n]["turnover_per_day"] for n in names],
            "Sharpe minus buy-hold (95% CI)": [
                "-"
                if n == "always_long"
                else f"{st[n]['sharpe_minus_buy_hold_ci'][0]:+.2f} ({st[n]['sharpe_minus_buy_hold_ci'][1]:+.2f} to {st[n]['sharpe_minus_buy_hold_ci'][2]:+.2f})"
                for n in names
            ],
        },
        index=pd.Index(names, name="strategy"),
    )
    trials = [str(t) for t in cfg["trial_counts"]]
    dsr_tbl = pd.DataFrame(
        {"PSR vs 0": [st[n]["psr_vs_zero"] for n in names]}
        | {f"DSR, {t} trials": [st[n]["dsr"][t] for n in names] for t in trials},
        index=pd.Index(names, name="strategy"),
    )
    cost_tbl = pd.DataFrame(res["cost_sensitivity"]).T
    cost_tbl.index.name = "net Sharpe at cost (bps)"
    cost_tbl.columns = [f"{float(c):g}" for c in cost_tbl.columns]

    def reg_tbl(key: str) -> pd.DataFrame:
        t = pd.DataFrame(rg[key]).T
        t.index = [plotting.REGIME_NAMES[int(i)] for i in t.index]
        t.index.name = "regime"
        t["share_of_days"] = t["share_of_days"].map("{:.0%}".format)
        t["ann_return"] = t["ann_return"].map("{:+.1%}".format)
        t["ann_vol"] = t["ann_vol"].map("{:.1%}".format)
        t["avg_spell_days"] = t["avg_spell_days"].map("{:.1f}".format)
        return t.rename(
            columns={"avg_spell_days": "avg spell (days)", "share_of_days": "share of days"}
        )

    parts = [
        "# Meridian: market-regime research report",
        "",
        "Generated by `python -m meridian.report`. Research, not financial advice.",
        "",
        "## Conclusions",
        "",
        *[f"- {line}" for line in verdict(res)],
        "",
        "## Data and protocol",
        "",
        f"- FRED daily series SP500, VIXCLS, DGS10, DGS2; {d['start']} to {d['end']} "
        f"({d['n_days']} trading days, {d['n_samples']} labelled samples after warm-up and the unlabelled tail).",
        f"- Target: does the S&P 500 close higher {cfg['horizon']} trading days later (direction of the log return).",
        f"- Purged walk-forward, rolling window: train {cfg['train']} days, embargo {cfg['embargo']}, "
        f"test {cfg['test']} days, labels purged where they overlap the test window; {lk['n_folds']} folds, "
        f"{meta['oos_days']} out-of-sample days ({meta['oos_start']} to {meta['oos_end']}).",
        "- Strategy: long the index when the model predicts up, flat otherwise, decided at the close "
        f"and earning the next day's return. Costs are charged per unit of position change "
        f"(base case {meta['base_cost_bps']:g} bps). Risk-free rate taken as zero.",
        "- Confidence intervals are moving-block bootstrap (block length = label horizon) because "
        "overlapping 21-day labels make observations dependent.",
        "",
        "## Regimes (descriptive, in-sample)",
        "",
        f"Volatility clustering is present: the Ljung-Box statistic (10 lags) is "
        f"{rg['ljung_box_q_squared_returns']:.0f} for squared daily returns "
        f"(p {_fmt_p(rg['ljung_box_p_squared_returns'])}) against {rg['ljung_box_q_returns']:.0f} "
        f"for returns (p {_fmt_p(rg['ljung_box_p_returns'])}): the size of moves is far more "
        "persistent than their direction. States are ordered by realised volatility. Labels below "
        "are Viterbi-smoothed over the whole sample, so they use hindsight and are never fed to a model.",
        "",
        "Gaussian HMM:",
        "",
        _md_table(reg_tbl("hmm"), ".3f"),
        "",
        "k-means baseline (no time structure):",
        "",
        _md_table(reg_tbl("kmeans"), ".3f"),
        "",
        f"Adjusted Rand index between the two labelings: {rg['ari_hmm_vs_kmeans']:.3f}. "
        "The HMM's persistence prior is visible in the longer average spells."
        + (
            " The stressed regime has a positive in-sample mean return, the opposite of what an "
            "'avoid high volatility' rule assumes: the smoothed path also labels the sharp rebounds "
            "that follow sell-offs as stressed."
            if rg["hmm"][max(rg["hmm"], key=int)]["ann_return"] > 0
            else ""
        ),
        "",
        "![Regimes](figures/regimes_price.png)",
        "",
        "## Leakage checks",
        "",
        f"- Split integrity asserted on all {lk['n_folds']} folds (no train/test overlap, labels purged, embargo respected).",
        f"- Timestamp check on the real features, flagged columns: {lk['features_flagged_by_timestamp_check'] or 'none'}.",
        f"- Planted canary `{leakage.CANARY_COLUMN}` flagged: {lk['canary_flagged_by_timestamp_check']}; "
        f"a model given it scores {lk['canary_model_balanced_accuracy']:.3f} balanced accuracy.",
        f"- Shuffled-label test ({cfg['n_shuffles']} shuffles): mean balanced accuracy "
        f"{lk['shuffled_label_balanced_accuracy']:.3f}, null 95% interval "
        f"{lk['shuffled_label_null_ci'][0]:.3f} to {lk['shuffled_label_null_ci'][1]:.3f}: "
        f"{'passed' if lk['shuffled_label_passed'] else 'failed'}.",
        "",
        "## Out-of-sample direction accuracy",
        "",
        _md_table(acc_tbl),
        "",
        "![Accuracy](figures/walkforward_accuracy.png)",
        "",
        f"## Strategy results ({meta['base_cost_bps']:g} bps per unit of position change)",
        "",
        _md_table(st_tbl),
        "",
        "![Equity](figures/equity_curves.png)",
        "",
        "## Deflated Sharpe ratio",
        "",
        f"PSR is the probability the true Sharpe exceeds zero. DSR additionally deflates for selecting the "
        f"best of N trials (Bailey and Lopez de Prado, 2014), using the variance of per-period Sharpe "
        f"across the {meta['n_trials_main']} candidate models ({meta['daily_sr_variance_across_trials']:.3g}). "
        f"Four is a floor: features, windows, k and horizon were also chosen by the author, so the 10 and "
        f"50 trial columns are the more honest reading.",
        "",
        _md_table(dsr_tbl),
        "",
        "## Cost sensitivity",
        "",
        _md_table(cost_tbl),
        "",
        "![Cost sensitivity](figures/cost_sensitivity.png)",
        "",
        "## Limitations",
        "",
        "- One market and about ten years of history (FRED publishes only a rolling decade of SP500), "
        "with one realised path: a single regime-rich stretch that includes the 2020 crash and the 2022 bear market.",
        "- Index levels exclude dividends; VIX is observed at the close, so a live implementation would "
        "need the same timestamp discipline. No slippage model beyond the flat per-trade cost.",
        "- Hyper-parameters were fixed before looking at results to avoid adding selection bias; "
        "a tuned model would need a nested validation scheme.",
        "",
    ]
    return "\n".join(parts)


def _distinct(series: dict) -> dict:
    """Drop `majority` from line charts when it is identical to `always_long` (it usually is:
    in a rising market the training-window majority class is 'up')."""
    if "majority" in series and np.allclose(
        np.asarray(series["majority"], dtype=float), np.asarray(series["always_long"], dtype=float)
    ):
        return {k: v for k, v in series.items() if k != "majority"}
    return series


def make_figures(res: dict, out_dir: Path) -> list[Path]:
    fr, acc = res["_frames"], res["accuracy"]
    names = ["majority", "always_long", "logreg", "gbm", "logreg+regime", "gbm+regime"]
    rows = pd.DataFrame(
        {
            "acc": [acc[n]["accuracy"] for n in names],
            "acc_lo": [acc[n]["accuracy_ci"][0] for n in names],
            "acc_hi": [acc[n]["accuracy_ci"][1] for n in names],
            "bal": [acc[n]["balanced_accuracy"] for n in names],
            "bal_lo": [acc[n]["balanced_accuracy_ci"][0] for n in names],
            "bal_hi": [acc[n]["balanced_accuracy_ci"][1] for n in names],
        },
        index=names,
    )
    paths = [
        plotting.save(
            plotting.plot_regimes(
                fr["price"],
                fr["labels"],
                fr["vol"],
                "S&P 500 by volatility regime (Gaussian HMM, 3 states, in-sample Viterbi)",
            ),
            out_dir / "regimes_price.png",
        ),
        plotting.save(
            plotting.plot_accuracy(rows, acc["always_long"]["accuracy"]),
            out_dir / "walkforward_accuracy.png",
        ),
        plotting.save(
            plotting.plot_costs(
                fr["costs"], "Net Sharpe as trading costs rise (walk-forward, long/flat)"
            ),
            out_dir / "cost_sensitivity.png",
        ),
        plotting.save(
            plotting.plot_equity(
                _distinct(fr["equity"]), "Out-of-sample growth of 1.0, net of 5 bps costs"
            ),
            out_dir / "equity_curves.png",
        ),
    ]
    return paths


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    df = data.load(cfg.series, "1990-01-01", None)
    res = evaluate(df, cfg)
    reports = ROOT / "reports"
    figs = make_figures(res, reports / "figures")
    media = ROOT / "docs" / "media"
    media.mkdir(parents=True, exist_ok=True)
    for p in figs:
        shutil.copyfile(p, media / p.name)
    shutil.copyfile(reports / "figures" / "regimes_price.png", media / "hero.png")
    (reports / "report.md").write_text(render_markdown(res))
    serial = {k: v for k, v in res.items() if not k.startswith("_")}
    (reports / "results.json").write_text(json.dumps(serial, indent=2, default=float) + "\n")
    print("wrote reports/report.md, reports/results.json and reports/figures/*.png")
    for line in verdict(res):
        print(" -", line)
    return res


if __name__ == "__main__":
    main()
