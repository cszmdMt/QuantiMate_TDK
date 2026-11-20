"""V7-5 tesztek: GBDT determinizmus (két futás bitre azonos),
kvantilis-monotonitás, tabuláris nézet kauzalitása, TELJES lánc-interfész
(Platt→CQR→Merton→backtest a GBDT-kimeneten)."""
import numpy as np
import pandas as pd
import pytest

from quantimate.models.models_gbdt import (GBDTQuantileModel, tabular_view,
                                    train_gbdt_quantiles)

HORIZONS = (1, 5)
QUANTS = (0.05, 0.25, 0.5, 0.75, 0.95)


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    n, f = 500, 8
    X = rng.normal(0, 1, (n, f))
    signal = 0.5 * X[:, 0] - 0.3 * X[:, 1]
    Y = np.column_stack([signal + rng.normal(0, 0.5, n),
                         signal + rng.normal(0, 0.8, n)])
    train_pos = np.arange(0, 350)
    val_pos = np.arange(360, 450)
    return X, Y, train_pos, val_pos


def test_two_runs_bit_identical(synth):
    """Azonos seeddel két tanítás bitre azonos predikciót ad."""
    X, Y, tr, va = synth
    m1 = train_gbdt_quantiles(X, Y, tr, va, HORIZONS, QUANTS, seed=42)
    m2 = train_gbdt_quantiles(X, Y, tr, va, HORIZONS, QUANTS, seed=42)
    np.testing.assert_array_equal(m1.predict_quantiles(X[va]),
                                  m2.predict_quantiles(X[va]))


def test_quantiles_sorted_and_shaped(synth):
    X, Y, tr, va = synth
    m = train_gbdt_quantiles(X, Y, tr, va, HORIZONS, QUANTS, seed=7)
    q = m.predict_quantiles(X[va])
    assert q.shape == (len(va), len(HORIZONS), len(QUANTS))
    assert (np.diff(q, axis=-1) >= 0).all(), "Kvantilis-keresztezés rendezés után"


def test_gbdt_learns_signal(synth):
    """A medián-fej korrelál a valódi jellel (sanity)."""
    X, Y, tr, va = synth
    m = train_gbdt_quantiles(X, Y, tr, va, HORIZONS, QUANTS, seed=42)
    med = m.predict_quantiles(X[va])[:, 0, 2]
    corr = np.corrcoef(med, 0.5 * X[va, 0] - 0.3 * X[va, 1])[0, 1]
    assert corr > 0.7


def test_tabular_view_lags_are_causal():
    """A lag-oszlopok kizárólag múltbeli hozamot hordoznak (pozitív shift)."""
    idx = pd.bdate_range("2024-01-01", periods=30)
    feats = pd.DataFrame({"NVDA_log_ret": np.arange(30, dtype=float),
                          "egyeb": np.ones(30)}, index=idx)
    tab = tabular_view(feats, "NVDA", lags=(2, 5))
    assert tab["NVDA_log_ret_lag2"].iloc[10] == feats["NVDA_log_ret"].iloc[8]
    assert tab["NVDA_log_ret_lag5"].iloc[10] == feats["NVDA_log_ret"].iloc[5]
    assert tab["NVDA_log_ret_lag5"].iloc[:5].isna().all()


def test_full_chain_interface_on_gbdt_output(synth):
    """V7-5: a kalibráció→konformális→Merton→backtest lánc hibátlanul lefut
    a GBDT kvantilis-kimenetén."""
    from quantimate.backtest import merton_fraction, run_backtest, sigma_from_quantiles
    from quantimate.validation.calibration import PlattCalibrator
    from quantimate.validation.conformal import aci_bands, conformity_scores
    from quantimate.models import prob_up_from_quantiles

    X, Y, tr, va = synth
    m = train_gbdt_quantiles(X, Y, tr, va, HORIZONS, QUANTS, seed=42)
    q_va = m.predict_quantiles(X[va])

    p_up = prob_up_from_quantiles(q_va[:, 0, :], QUANTS)
    y_dir = (Y[va, 0] > 0).astype(int)
    p_cal = PlattCalibrator().fit(p_up, y_dir).predict(p_up)
    assert ((p_cal > 0) & (p_cal < 1)).all()

    half = len(va) // 2
    scores = conformity_scores(Y[va[:half], 0], q_va[:half, 0, 0], q_va[:half, 0, -1])
    lo, hi, alphas, errs = aci_bands(Y[va[half:], 0], q_va[half:, 0, 0],
                                     q_va[half:, 0, -1], scores, alpha_target=0.15)
    assert np.isfinite(lo).all() and np.isfinite(hi).all()

    sig_hat = sigma_from_quantiles(q_va[:, 0, 0], q_va[:, 0, -1])
    pos = np.where(p_cal >= 0.5,
                   merton_fraction(q_va[:, 0, 2], sig_hat, sigma_floor=0.05), 0.0)
    strat, equity, _ = run_backtest(pos, np.tanh(Y[va, 0]) * 0.02, cost=0.0005)
    assert np.isfinite(equity).all() and len(equity) == len(va)
