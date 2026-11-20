"""V7-6 tesztek: PatchTST/iTransformer kvantilis-fej alak+monotonitás+
determinizmus; Chronos import-guard (nincs chronos → tiszta RuntimeError)."""
import numpy as np
import pytest

from quantimate.models import (build_itransformer, build_patchtst,
                               predict_quantiles, train_quantile_model)

QUANTS = (0.05, 0.25, 0.5, 0.75, 0.95)
HORIZONS = (1, 5, 21)


@pytest.mark.parametrize("builder", [build_patchtst, build_itransformer])
def test_arch_shape_and_monotone(builder):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (48, 90, 8)).astype("float32")
    Y = rng.normal(0, 1, (48, 3)).astype("float32")
    m, _ = train_quantile_model(builder, X, Y, X[:12], Y[:12], seed=42, epochs=1,
                                quantiles=QUANTS, horizons=HORIZONS)
    q = predict_quantiles(m, X[:12], n_horizons=3, n_quantiles=5)
    assert q.shape == (12, 3, 5)
    assert (np.diff(q, axis=-1) >= 0).all()


def test_patchtst_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (48, 90, 8)).astype("float32")
    Y = rng.normal(0, 1, (48, 3)).astype("float32")
    m1, _ = train_quantile_model(build_patchtst, X, Y, X[:12], Y[:12], seed=7,
                                 epochs=2, quantiles=QUANTS, horizons=HORIZONS)
    m2, _ = train_quantile_model(build_patchtst, X, Y, X[:12], Y[:12], seed=7,
                                 epochs=2, quantiles=QUANTS, horizons=HORIZONS)
    np.testing.assert_array_equal(predict_quantiles(m1, X[:12]),
                                  predict_quantiles(m2, X[:12]))


def test_chronos_guard_when_unavailable():
    """Ha a chronos nincs telepítve, a wrapper érthető RuntimeError-t dob
    (a hívó ezt elkapja és őszintén kihagyja a szakaszt)."""
    from quantimate.models.models_zeroshot import chronos_available, zeroshot_quantiles
    if not chronos_available():
        with pytest.raises(RuntimeError, match="nem elérhető"):
            zeroshot_quantiles([np.zeros(90)], HORIZONS, QUANTS)
    else:
        pytest.skip("chronos elérhető — a guard-ág nem tesztelhető itt")
