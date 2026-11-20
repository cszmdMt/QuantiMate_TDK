# -*- coding: utf-8 -*-
"""HPO (V8-8) tesztjei: keresési tér-határok, seedelt determinizmus,
OOF-lefedettség, default-trial jelenlét."""
import numpy as np
import pytest

from quantimate.models.hpo import (SEARCH_SPACE, oof_losses, random_search,
                            sample_config)

HORIZONS = (1,)
QUANTS = (0.25, 0.5, 0.75)


def _toy_data(n: int = 260, seed: int = 3):
    """Kis szintetikus feladat: 2 informatív + 3 zaj-jellemző."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 0.6 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(0, 0.5, n)
    Y = y[:, None]
    train_pos = np.arange(n)
    return X, Y, train_pos


class TestSampleConfig:
    def test_within_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            hp = sample_config(rng)
            for name, (kind, lo, hi) in SEARCH_SPACE.items():
                assert lo <= hp[name] <= hi, name

    def test_seeded_determinism(self):
        a = sample_config(np.random.default_rng(42))
        b = sample_config(np.random.default_rng(42))
        assert a == b


class TestOOF:
    def test_full_coverage_no_nan(self):
        X, Y, train_pos = _toy_data()
        losses = oof_losses(X, Y, train_pos, HORIZONS, QUANTS,
                            {"n_estimators": 25}, n_folds=3, purge=5,
                            embargo=2, seed=42)
        assert len(losses) == len(train_pos)
        assert not np.isnan(losses).any()
        assert (losses >= 0).all()


class TestRandomSearch:
    def test_default_is_trial_zero_and_deterministic(self):
        X, Y, train_pos = _toy_data()
        kw = dict(n_trials=3, n_folds=2, seed=42, verbose=False)
        r1 = random_search(X, Y, train_pos, HORIZONS, QUANTS, **kw)
        r2 = random_search(X, Y, train_pos, HORIZONS, QUANTS, **kw)
        assert r1["trials"][0]["default"] is True
        assert r1["best_hp"] == r2["best_hp"]
        assert r1["best_score"] == pytest.approx(r2["best_score"])
        assert r1["loss_matrix"].shape == (len(train_pos), 3)
        # a győztes pontszáma definíció szerint nem rosszabb a defaultnál
        assert r1["best_score"] <= r1["default_score"] + 1e-12
