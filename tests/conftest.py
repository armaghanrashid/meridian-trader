import pytest

from meridian import data, features


@pytest.fixture(scope="session")
def synth():
    return data.synthetic(seed=7, n=3000)


@pytest.fixture(scope="session")
def feats(synth):
    return features.build(synth)


@pytest.fixture(scope="session")
def dataset(synth):
    return features.dataset(synth, horizon=21)
