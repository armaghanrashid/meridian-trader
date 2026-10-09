import numpy as np
import pytest

from meridian import features, leakage, validation


def test_known_split_geometry():
    folds = validation.purged_walk_forward(100, train=40, test=10, embargo=5)
    assert len(folds) == 5
    tr, te = folds[0]
    assert tr[0] == 0 and tr[-1] == 39
    assert te[0] == 45 and te[-1] == 54
    tr2, te2 = folds[1]
    assert tr2[0] == 10 and te2[0] == 55  # rolling window, contiguous test blocks


@pytest.mark.parametrize("horizon", [0, 5, 21])
@pytest.mark.parametrize("embargo", [0, 5, 30])
def test_purged_splits_never_overlap_the_embargo_window(horizon, embargo):
    folds = validation.purged_walk_forward(
        400, train=120, test=30, embargo=embargo, horizon=horizon
    )
    assert folds
    last_test_end = -1
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0
        assert te[0] - tr[-1] - 1 >= embargo  # gap at least the embargo
        assert tr[-1] + horizon < te[0]  # train labels end before test starts
        assert te[0] > last_test_end  # test blocks ascend and never repeat
        last_test_end = te[-1]
        validation.assert_no_overlap(tr, te, horizon=horizon, embargo=embargo)


def test_purge_drops_train_samples_whose_label_reaches_the_test_window():
    (tr, te), *_ = validation.purged_walk_forward(100, train=40, test=10, embargo=5, horizon=21)
    assert te[0] == 45
    assert tr[-1] == 23  # t + 21 < 45


def test_expanding_window_keeps_the_origin():
    folds = validation.purged_walk_forward(100, train=40, test=10, embargo=5, expanding=True)
    assert all(tr[0] == 0 for tr, _ in folds)
    assert len(folds[-1][0]) > len(folds[0][0])


def test_assert_no_overlap_rejects_bad_splits():
    with pytest.raises(validation.LeakageError):
        validation.assert_no_overlap(np.arange(0, 50), np.arange(40, 60), horizon=0, embargo=0)
    with pytest.raises(validation.LeakageError):  # label of last train row reaches test
        validation.assert_no_overlap(np.arange(0, 40), np.arange(45, 55), horizon=21, embargo=5)
    with pytest.raises(validation.LeakageError):  # gap smaller than embargo
        validation.assert_no_overlap(np.arange(0, 40), np.arange(42, 52), horizon=0, embargo=5)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        validation.purged_walk_forward(10, train=40, test=10, embargo=5)


def test_canary_is_flagged_and_clean_features_pass(synth):
    assert validation.find_lookahead_columns(features.build, synth) == []
    leaky = leakage.with_future_canary(features.build, horizon=21)
    assert validation.find_lookahead_columns(leaky, synth) == [leakage.CANARY_COLUMN]


def test_a_subtler_one_day_peek_is_also_caught(synth):
    def peeky(df):
        out = features.build(df)
        out["tomorrow"] = np.log(df["SP500"]).shift(-1) - np.log(df["SP500"])
        return out

    assert validation.find_lookahead_columns(peeky, synth) == ["tomorrow"]
