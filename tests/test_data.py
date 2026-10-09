import numpy as np
import pandas as pd

from meridian import data


def test_synthetic_is_reproducible_and_well_formed():
    a = data.synthetic(seed=1, n=500)
    b = data.synthetic(seed=1, n=500)
    c = data.synthetic(seed=2, n=500)
    pd.testing.assert_frame_equal(a, b)
    assert not a["SP500"].equals(c["SP500"])
    assert {"SP500", "VIXCLS", "DGS10", "DGS2", "true_regime"} <= set(a.columns)
    assert isinstance(a.index, pd.DatetimeIndex)
    assert a.index.is_monotonic_increasing
    assert (a["SP500"] > 0).all()
    assert set(a["true_regime"].unique()) == {0, 1, 2}


def test_synthetic_regimes_are_ordered_by_volatility():
    df = data.synthetic(seed=3, n=4000)
    r = np.log(df["SP500"]).diff()
    vol = r.groupby(df["true_regime"]).std()
    assert vol.is_monotonic_increasing


def test_parse_fred_csv_handles_missing_markers():
    text = "observation_date,VIXCLS\n2020-01-02,12.5\n2020-01-03,.\n2020-01-06,13.0\n"
    s = data.parse_fred_csv(text, "VIXCLS")
    assert s.name == "VIXCLS"
    assert np.isnan(s.loc["2020-01-03"])
    assert s.loc["2020-01-06"] == 13.0


def test_load_merges_slices_and_caches(tmp_path):
    calls = []

    def fake_fetch(series_id):
        calls.append(series_id)
        idx = pd.bdate_range("2020-01-01", periods=30)
        base = {"SP500": 3000.0, "VIXCLS": 15.0, "DGS10": 1.8, "DGS2": 1.5}[series_id]
        vals = base + np.arange(len(idx), dtype=float)
        csv = f"observation_date,{series_id}\n"
        csv += "\n".join(f"{d:%Y-%m-%d},{v}" for d, v in zip(idx, vals, strict=True))
        return csv

    kw = {"cache_dir": tmp_path, "fetch": fake_fetch}
    df = data.load(["SP500", "VIXCLS", "DGS10", "DGS2"], "2020-01-06", "2020-01-17", **kw)
    assert list(df.columns) == ["SP500", "VIXCLS", "DGS10", "DGS2"]
    assert df.index.min() >= pd.Timestamp("2020-01-06")
    assert df.index.max() <= pd.Timestamp("2020-01-17")
    assert len(calls) == 4
    data.load(["SP500", "VIXCLS"], "2020-01-06", "2020-01-17", **kw)
    assert len(calls) == 4  # served from the cache
    assert len(list(tmp_path.glob("*.csv"))) == 4


def test_load_drops_days_without_an_index_level_and_fills_short_gaps(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=10)

    def fake_fetch(series_id):
        rows = ["observation_date," + series_id]
        for i, d in enumerate(idx):
            missing = (series_id == "SP500" and i == 3) or (series_id == "DGS2" and i in (4, 5))
            rows.append(f"{d:%Y-%m-%d},{'.' if missing else 100 + i}")
        return "\n".join(rows)

    df = data.load(
        ["SP500", "DGS2"], "2020-01-01", "2020-01-31", cache_dir=tmp_path, fetch=fake_fetch
    )
    assert idx[3] not in df.index  # no index level -> not a trading day
    assert not df.isna().any().any()  # short rate gaps forward-filled
