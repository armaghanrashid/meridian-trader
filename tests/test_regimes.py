import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from meridian import features, regimes


def _regime_x(feats):
    return feats[features.REGIME_COLUMNS].dropna()


def test_hmm_recovers_planted_regimes(synth, feats):
    X = _regime_x(feats)
    labels, model = regimes.fit_hmm(X, k=3, seed=0)
    truth = synth.loc[X.index, "true_regime"]
    assert adjusted_rand_score(truth, labels) > 0.6
    assert len(labels) == len(X)
    assert model.k == 3


def test_labels_are_aligned_by_volatility(synth, feats):
    X = _regime_x(feats)
    labels, _ = regimes.fit_hmm(X, k=3, seed=0)
    mean_vol = X["rv21"].groupby(labels).mean()
    assert mean_vol.is_monotonic_increasing
    km_labels, _ = regimes.fit_kmeans(X, k=3, seed=0)
    assert X["rv21"].groupby(km_labels).mean().is_monotonic_increasing


def test_kmeans_baseline_is_scored_on_the_same_footing(synth, feats):
    X = _regime_x(feats)
    labels, _ = regimes.fit_kmeans(X, k=3, seed=0)
    truth = synth.loc[X.index, "true_regime"]
    assert adjusted_rand_score(truth, labels) > 0.3


def test_filtered_probabilities_are_causal_and_normalised(feats):
    X = _regime_x(feats)
    train, test = X.iloc[:1200], X.iloc[1200:1500]
    _, model = regimes.fit_hmm(train, k=3, seed=0)
    full = model.filter(test)
    assert full.shape == (len(test), 3)
    np.testing.assert_allclose(full.sum(axis=1), 1.0, atol=1e-9)
    # Filtering at t must equal filtering on the data truncated at t.
    cut = model.filter(test.iloc[:100])
    np.testing.assert_allclose(cut[-1], full[99], atol=1e-9)


def test_fit_is_deterministic_given_seed(feats):
    X = _regime_x(feats).iloc[:900]
    a, _ = regimes.fit_hmm(X, k=3, seed=5)
    b, _ = regimes.fit_hmm(X, k=3, seed=5)
    assert isinstance(a, pd.Series)
    assert (a.to_numpy() == b.to_numpy()).all()
