"""Platt-kalibráció — javít a torzított valószínűségeken, kis mintán stabil."""
import numpy as np

from quantimate.validation.calibration import PlattCalibrator, reliability_curve


def test_platt_fixes_biased_probabilities():
    """Torzított (0.5 felé nyomott) nyers p-ken a Platt val-Brier-t javít."""
    rng = np.random.default_rng(0)
    n = 369                                    # a V5 val-méreténél is stabil
    p_true = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0, 1, n) < p_true).astype(int)
    p_raw = 0.5 + 0.3 * (p_true - 0.5)         # összenyomott, alulkonfidens

    cal = PlattCalibrator().fit(p_raw, y)
    p_cal = cal.predict(p_raw)

    brier_raw = np.mean((p_raw - y) ** 2)
    brier_cal = np.mean((p_cal - y) ** 2)
    assert brier_cal < brier_raw


def test_platt_output_valid_probabilities():
    rng = np.random.default_rng(1)
    p_raw = rng.uniform(0, 1, 200)
    y = rng.integers(0, 2, 200)
    p_cal = PlattCalibrator().fit(p_raw, y).predict(p_raw)
    assert ((p_cal > 0) & (p_cal < 1)).all()


def test_reliability_curve_shapes():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p).astype(int)
    xs, ys, ns = reliability_curve(p, y, n_bins=10)
    assert len(xs) == len(ys) == len(ns)
    assert (ns >= 5).all()
    # Jól kalibrált adaton a görbe közel diagonális
    assert np.abs(xs - ys).mean() < 0.15
