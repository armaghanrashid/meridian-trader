import numpy as np
import pytest

from meridian import leakage, models, validation


def _folds(n):
    return validation.purged_walk_forward(n, train=500, test=100, embargo=5, horizon=21)


def test_null_ci_shrinks_with_sample_size():
    lo1, hi1 = leakage.balanced_accuracy_null_ci(100, 100)
    lo2, hi2 = leakage.balanced_accuracy_null_ci(1000, 1000)
    assert lo1 < 0.5 < hi1 and lo2 < 0.5 < hi2
    assert (hi2 - lo2) < (hi1 - lo1)


def test_shuffled_label_test_passes_on_honest_pipeline(dataset):
    spec = next(s for s in models.default_specs() if s.name == "logreg")
    res = leakage.shuffled_label_test(
        dataset.X, dataset.y, _folds(len(dataset.X)), spec, n_shuffles=5, seed=0
    )
    assert res.passed
    assert res.ci_low <= res.mean_balanced_accuracy <= res.ci_high
    assert len(res.balanced_accuracies) == 5


def test_unshuffled_run_exposes_a_leak_that_the_shuffle_test_cannot(dataset):
    # With the canary in X the model "predicts" unshuffled labels almost perfectly, but with
    # shuffled labels it must still be at chance; the *unshuffled* run is what exposes the leak.
    X = dataset.X.copy()
    X[leakage.CANARY_COLUMN] = dataset.fwd_ret
    spec = next(s for s in models.default_specs() if s.name == "logreg")
    preds = models.walk_forward(X, dataset.y, _folds(len(X)), spec)
    summary = models.summarize(preds, horizon=21)
    assert summary["balanced_accuracy"] > 0.9  # implausible, hence the canary
    assert leakage.is_implausible(summary["balanced_accuracy"])


def test_is_implausible_threshold():
    assert not leakage.is_implausible(0.56)
    assert leakage.is_implausible(0.90)
    with pytest.raises(ValueError):
        leakage.is_implausible(1.2)


def test_canary_column_is_future_return(synth):
    from meridian import features

    build = leakage.with_future_canary(features.build, horizon=21)
    out = build(synth)
    expected = np.log(synth["SP500"]).shift(-21) - np.log(synth["SP500"])
    np.testing.assert_allclose(out[leakage.CANARY_COLUMN].dropna(), expected.dropna())
