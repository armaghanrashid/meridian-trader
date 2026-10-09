import numpy as np
import pytest

from meridian import models, validation


def _folds(n, train=500, test=100):
    return validation.purged_walk_forward(n, train=train, test=test, embargo=5, horizon=21)


def _spec(name):
    return next(s for s in models.default_specs() if s.name == name)


def test_baselines():
    maj = models.MajorityClass().fit(np.zeros((5, 2)), np.array([1, 1, 0, 1, 0]))
    assert (maj.predict(np.zeros((3, 2))) == 1).all()
    assert maj.predict_proba(np.zeros((2, 2)))[:, 1] == pytest.approx([0.6, 0.6])
    assert (models.AlwaysLong().fit(None, None).predict(np.zeros((4, 1))) == 1).all()


def test_default_specs_cover_baselines_and_models():
    names = [s.name for s in models.default_specs()]
    assert names == [
        "majority",
        "always_long",
        "logreg",
        "gbm",
        "logreg+regime",
        "gbm+regime",
    ]


def test_walk_forward_output_shape_and_ordering(dataset):
    folds = _folds(len(dataset.X))
    preds = models.walk_forward(dataset.X, dataset.y, folds, _spec("logreg"))
    n_test = sum(len(te) for _, te in folds)
    assert len(preds) == n_test
    assert preds.index.is_unique and preds.index.is_monotonic_increasing
    assert preds["proba"].between(0, 1).all()
    assert set(preds["pred"].unique()) <= {0, 1}
    assert preds["fold"].is_monotonic_increasing
    assert preds["y"].equals(dataset.y.loc[preds.index])


def test_folds_do_not_see_the_future(dataset):
    # Re-running on data truncated right after fold 0's test block must reproduce fold 0.
    folds = _folds(len(dataset.X))
    full = models.walk_forward(dataset.X, dataset.y, folds[:1], _spec("logreg+regime"))
    cut = folds[0][1][-1] + 1
    short = models.walk_forward(
        dataset.X.iloc[:cut], dataset.y.iloc[:cut], folds[:1], _spec("logreg+regime")
    )
    np.testing.assert_allclose(full["proba"], short["proba"], atol=1e-9)


def test_positive_control_pipeline_detects_a_planted_signal(dataset):
    # In the synthetic world the regime drives drift, so a working pipeline must find an edge.
    folds = _folds(len(dataset.X))
    preds = models.walk_forward(dataset.X, dataset.y, folds, _spec("logreg+regime"))
    s = models.summarize(preds, horizon=21, seed=0)
    assert s["balanced_accuracy_ci"][0] > 0.5
    base = models.summarize(
        models.walk_forward(dataset.X, dataset.y, folds, _spec("majority")), horizon=21, seed=0
    )
    assert s["balanced_accuracy"] > base["balanced_accuracy"] + 0.05
    assert s["accuracy"] > base["accuracy"]


def test_summarize_reports_cis_that_contain_the_point_estimate(dataset):
    folds = _folds(len(dataset.X))
    preds = models.walk_forward(dataset.X, dataset.y, folds, _spec("always_long"))
    s = models.summarize(preds, horizon=21, seed=0)
    lo, hi = s["accuracy_ci"]
    assert lo <= s["accuracy"] <= hi
    assert s["n"] == len(preds)
