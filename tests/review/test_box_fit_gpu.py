"""Fit box's rankers trained on the GPU (nnslr_tools.box_fit.gpu) are the numpy ones: the same trees and the same MLP
weights on the same small made-up data (run with .venv-gpu's Python on a machine with a card; skipped elsewhere)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA card')

from nnslr_tools.box_fit import edges, gbdt, gpu  # noqa: E402


def test_the_trees_on_the_gpu_predict_as_the_numpy_ones():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 6))
    y = np.tanh(X[:, 0] * 2 - X[:, 3]) + 0.1 * rng.normal(size=600)
    kw = dict(n_trees=20, depth=4, min_leaf=10, seed=7)
    ours, theirs = gbdt.pack(gpu.train_trees(X, y, **kw)), gbdt.pack(gbdt.train(X, y, **kw))
    assert np.allclose(gbdt.predict(ours, X), gbdt.predict(theirs, X), atol=1e-9)


def test_the_mlps_on_the_gpu_are_the_numpy_ones():
    rng = np.random.default_rng(5)
    cases = []
    for _ in range(40):
        n = int(rng.integers(3, 30))
        F = rng.normal(size=(n, 8))
        cases.append((F, np.clip(0.5 + 0.3 * F[:, 1], 0, 1)))
    kw = dict(epochs=3, ncand=12, top=4, batch=8)
    ours = gpu.train_rankers(cases, [1, 2], **kw)
    for seed, (mu, sd, ps) in zip([1, 2], ours):
        mu0, sd0, ps0 = edges.train_ranker(cases, seed=seed, **kw)
        assert np.allclose(mu, mu0) and np.allclose(sd, sd0)
        for p, p0 in zip(ps, ps0):
            assert np.allclose(p, p0, atol=1e-9), np.abs(p - p0).max()


def test_without_a_card_it_stops():
    with pytest.raises(gpu.NoCard):
        gpu.device('cpu')
