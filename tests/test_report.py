import numpy as np
import pytest
from PIL import Image

from meridian import data, report


@pytest.fixture(scope="module")
def small_result():
    cfg = report.Config(train=400, test=100, n_boot=100, n_shuffles=3, bps_grid=(0, 5, 25))
    return report.evaluate(data.synthetic(seed=7, n=1600), cfg)


def test_evaluate_produces_every_reported_quantity(small_result):
    res = small_result
    assert set(res["accuracy"]) == {
        "majority",
        "always_long",
        "logreg",
        "gbm",
        "logreg+regime",
        "gbm+regime",
    }
    assert res["leakage"]["canary_flagged_by_timestamp_check"] == ["canary_future"]
    assert res["leakage"]["features_flagged_by_timestamp_check"] == []
    assert res["leakage"]["shuffled_label_passed"]
    assert res["leakage"]["canary_model_balanced_accuracy"] > 0.75
    s = res["strategies"]["always_long"]
    assert s["time_in_market"] == 1.0
    assert 0.0 <= s["psr_vs_zero"] <= 1.0
    assert s["dsr"]["4"] <= s["psr_vs_zero"] + 1e-12  # deflation never helps
    assert res["strategies"]["gbm"]["gross_sharpe"] >= res["strategies"]["gbm"]["net_sharpe"] - 1e-9


def test_markdown_contains_all_sections_and_disclaimer(small_result):
    md = report.render_markdown(small_result)
    for heading in (
        "## Conclusions",
        "## Leakage checks",
        "## Out-of-sample direction accuracy",
        "## Deflated Sharpe ratio",
        "## Cost sensitivity",
        "## Limitations",
    ):
        assert heading in md
    assert "not financial advice" in md
    assert "Users/" not in md  # no machine-specific paths


def test_figures_are_1600px_wide(small_result, tmp_path):
    paths = report.make_figures(small_result, tmp_path)
    assert {p.name for p in paths} == {
        "regimes_price.png",
        "walkforward_accuracy.png",
        "cost_sensitivity.png",
        "equity_curves.png",
    }
    for p in paths:
        with Image.open(p) as im:
            assert im.width == 1600
            assert not im.info.get("Software")  # metadata stripped


def _fake(best_ci_lo, dsr, acc_ci_lo):
    def strat(sh, ci, d):
        return {
            "net_sharpe": sh,
            "time_in_market": 0.6,
            "sharpe_minus_buy_hold_ci": [sh - 0.5, *ci],
            "dsr": {"4": d},
        }

    acc = {"accuracy": 0.6, "acc_minus_always_long": [0.01, acc_ci_lo, 0.1]}
    return {
        "config": {},
        "strategy_meta": {"n_trials_main": 4, "base_cost_bps": 5.0},
        "accuracy": {m: dict(acc) for m in report.TRIALS} | {"always_long": {"accuracy": 0.6}},
        "strategies": {
            "always_long": strat(0.5, [0, 0], 0.5),
            **{m: strat(0.4, [best_ci_lo, 0.5], dsr) for m in report.TRIALS},
        },
        "leakage": {
            "shuffled_label_balanced_accuracy": 0.5,
            "shuffled_label_null_ci": [0.47, 0.53],
            "shuffled_label_passed": True,
            "canary_flagged_by_timestamp_check": ["canary_future"],
            "canary_model_balanced_accuracy": 0.96,
            "features_flagged_by_timestamp_check": [],
        },
    }


def test_verdict_reports_a_null_result_as_null():
    text = " ".join(report.verdict(_fake(best_ci_lo=-0.3, dsr=0.7, acc_ci_lo=-0.05)))
    assert "no statistically reliable edge" in text
    assert "hypothesis to replicate" not in text


def test_verdict_only_claims_an_edge_when_ci_and_dsr_both_support_it():
    edge = " ".join(report.verdict(_fake(best_ci_lo=0.1, dsr=0.97, acc_ci_lo=0.02)))
    assert "hypothesis to replicate" in edge
    no_dsr = " ".join(report.verdict(_fake(best_ci_lo=0.1, dsr=0.80, acc_ci_lo=0.02)))
    assert "no statistically reliable edge" in no_dsr


def test_run_lengths():
    import pandas as pd

    labels = pd.Series([0, 0, 0, 1, 1, 0, 2])
    out = report.run_lengths(labels)
    assert out[0] == pytest.approx(2.0) and out[1] == 2.0 and out[2] == 1.0
    assert np.isfinite(out).all()
