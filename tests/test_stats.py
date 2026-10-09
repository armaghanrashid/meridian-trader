import numpy as np
import pytest

from meridian import stats

# Independent reference: closed-form Bailey & Lopez de Prado (2014) evaluated with the
# standard library's NormalDist (no scipy), for the inputs below.
REF = {"sr": 0.0875, "n": 1250, "skew": -0.5, "kurt": 5.0, "trials": 10, "var": 0.0004}
REF_EMAX = 0.031491966026914994
REF_DSR = 0.9732210666924266
REF_PSR0 = 0.9987185315171988


def test_expected_max_sharpe_matches_reference():
    assert stats.expected_max_sharpe(REF["var"], REF["trials"]) == pytest.approx(REF_EMAX, abs=1e-6)


def test_deflated_sharpe_matches_reference_within_1e6():
    dsr = stats.deflated_sharpe_ratio(
        REF["sr"], REF["var"], REF["trials"], REF["n"], REF["skew"], REF["kurt"]
    )
    assert dsr == pytest.approx(REF_DSR, abs=1e-6)


def test_probabilistic_sharpe_against_zero_matches_reference():
    psr = stats.probabilistic_sharpe_ratio(REF["sr"], 0.0, REF["n"], REF["skew"], REF["kurt"])
    assert psr == pytest.approx(REF_PSR0, abs=1e-6)


def test_deflation_properties():
    args = (0.05, 0.0004)
    one = stats.deflated_sharpe_ratio(0.05, 0.0004, 1, 1000, 0.0, 3.0)
    psr = stats.probabilistic_sharpe_ratio(0.05, 0.0, 1000, 0.0, 3.0)
    assert one == pytest.approx(psr, abs=1e-12)  # a single trial is not deflated
    vals = [stats.deflated_sharpe_ratio(*args, k, 1000, 0.0, 3.0) for k in (2, 10, 100, 1000)]
    assert vals == sorted(vals, reverse=True)  # more trials -> harsher


def test_sharpe_is_annualised_mean_over_std():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 1000)
    assert stats.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert stats.sharpe(np.zeros(10)) == 0.0


def test_block_bootstrap_ci_covers_truth_and_widens_under_autocorrelation():
    rng = np.random.default_rng(1)
    iid = rng.normal(0.5, 1.0, 600)
    point, lo, hi = stats.block_bootstrap_ci(iid, np.mean, block_len=5, n_boot=1500, seed=0)
    assert lo < 0.5 < hi and point == pytest.approx(iid.mean())
    assert 0.8 * 0.16 < hi - lo < 1.4 * 0.16  # about 2*1.96/sqrt(600)

    ar = np.zeros(1500)
    eps = rng.normal(size=1500)
    for i in range(1, 1500):
        ar[i] = 0.9 * ar[i - 1] + eps[i]
    _, lo1, hi1 = stats.block_bootstrap_ci(ar, np.mean, 1, 1000, seed=0)
    _, lo40, hi40 = stats.block_bootstrap_ci(ar, np.mean, 40, 1000, seed=0)
    assert (hi40 - lo40) > 1.5 * (hi1 - lo1)


def test_block_bootstrap_supports_paired_statistics():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 2, 400).astype(float)
    pair = np.column_stack([a, a])
    point, lo, hi = stats.block_bootstrap_ci(
        pair, lambda m: m[:, 0].mean() - m[:, 1].mean(), 10, 300, seed=0
    )
    assert point == lo == hi == 0.0


def test_turnover_counts_entry_and_exit():
    assert stats.turnover(np.array([0, 1, 1, 0])) == pytest.approx(2 / 4)
    assert stats.turnover(np.array([1, 1, 1, 1])) == pytest.approx(1 / 4)  # one entry


def test_cost_sensitivity_is_monotone_and_zero_cost_equals_gross():
    rng = np.random.default_rng(3)
    ret = rng.normal(0.0004, 0.01, 800)
    pos = rng.integers(0, 2, 800).astype(float)  # flips about every other day
    table = stats.cost_sensitivity(pos, ret, [0, 2, 5, 10, 25])
    gross = stats.sharpe(pos * ret)
    assert table.loc[0, "sharpe"] == pytest.approx(gross)
    assert table["sharpe"].is_monotonic_decreasing
    assert table.loc[25, "ann_return"] < table.loc[0, "ann_return"]


def test_max_drawdown():
    r = np.log(np.array([1.1, 0.9, 1.2, 0.6, 1.0]))
    # equity: 1.1, 0.99, 1.188, 0.7128, 0.7128 -> worst fall from 1.188 to 0.7128
    assert stats.max_drawdown(r) == pytest.approx(-0.4)
