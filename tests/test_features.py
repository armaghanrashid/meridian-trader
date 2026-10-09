import numpy as np
import pandas as pd

from meridian import features


def test_build_columns_and_sanity(feats):
    assert list(feats.columns) == features.FEATURE_COLUMNS
    clean = feats.dropna()
    assert len(clean) > 0.95 * len(feats)
    assert (clean["drawdown"] <= 1e-12).all()
    assert (clean["rv21"] > 0).all()
    assert np.isfinite(clean.to_numpy()).all()


def test_features_are_strictly_backward_looking(synth, feats):
    # Value at t must not change when every observation after t is deleted.
    for t in (150, 700, 1500):
        truncated = features.build(synth.iloc[: t + 1])
        pd.testing.assert_series_equal(
            truncated.iloc[-1], feats.iloc[t], check_names=False, rtol=1e-10, atol=1e-12
        )


def test_spread_and_vix_definitions(synth, feats):
    pd.testing.assert_series_equal(
        feats["curve"], synth["DGS10"] - synth["DGS2"], check_names=False
    )
    pd.testing.assert_series_equal(feats["vix"], synth["VIXCLS"], check_names=False)
    expected = synth["VIXCLS"].diff(5)
    pd.testing.assert_series_equal(feats["vix_chg5"], expected, check_names=False)


def test_targets_look_forward_exactly_h_days(synth):
    tg = features.targets(synth, horizon=21)
    p = np.log(synth["SP500"])
    t = 300
    assert np.isclose(tg["fwd_ret"].iloc[t], p.iloc[t + 21] - p.iloc[t])
    assert tg["y"].iloc[t] == float(tg["fwd_ret"].iloc[t] > 0)
    assert tg["y"].iloc[-21:].isna().all()
    assert np.isclose(tg["next_ret"].iloc[t], p.iloc[t + 1] - p.iloc[t])


def test_dataset_is_clean_and_contiguous(dataset):
    assert not dataset.X.isna().any().any()
    assert set(dataset.y.unique()) <= {0, 1}
    assert dataset.X.index.equals(dataset.y.index)
    gaps = dataset.X.index.to_series().diff().dropna().dt.days
    assert gaps.max() <= 5
