"""Backtest-invariánsok + Merton-frakció viselkedés + kézi 3 napos numerikus példa."""
import numpy as np
import pytest

from quantimate.backtest import (merton_fraction, risk_report, run_backtest,
                                 sigma_from_quantiles)


@pytest.fixture()
def r_log():
    rng = np.random.default_rng(0)
    return rng.normal(0.0005, 0.03, 250)


def test_zero_position_equity_one(r_log):
    """Invariáns: pos≡0 → equity≡1.0 minden nap."""
    _, equity, _ = run_backtest(np.zeros_like(r_log), r_log, cost=0.001)
    np.testing.assert_array_equal(equity, np.ones_like(equity))


def test_full_position_no_cost_equals_buy_and_hold(r_log):
    """Invariáns: pos≡1, cost=0 → equity bitre = buy&hold."""
    _, equity, _ = run_backtest(np.ones_like(r_log), r_log, cost=0.0)
    bh = np.cumprod(1 + np.expm1(r_log))
    np.testing.assert_allclose(equity, bh, rtol=1e-14)


def test_cost_never_helps(r_log):
    """Invariáns: cost>0 → equity ≤ költségmentes változat minden ponton."""
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 0.5, len(r_log))
    _, eq_free, _ = run_backtest(pos, r_log, cost=0.0)
    _, eq_cost, _ = run_backtest(pos, r_log, cost=0.0005)
    assert (eq_cost <= eq_free + 1e-15).all()


def test_hand_computed_three_day_example():
    """Kézi példa: pos=[0.5, 0.5, 0], r_log=[ln(1.02), ln(0.99), ln(1.01)],
    cost=0.001. r_simple=[0.02, −0.01, 0.01]; turnover=[0.5, 0, 0.5]."""
    r_log = np.log([1.02, 0.99, 1.01])
    pos = np.array([0.5, 0.5, 0.0])
    strat, equity, turnover = run_backtest(pos, r_log, cost=0.001)
    exp_r1 = 0.5 * 0.02 - 0.001 * 0.5           # 0.0095
    exp_r2 = 0.5 * (-0.01) - 0.0                # −0.005
    exp_r3 = 0.0 - 0.001 * 0.5                  # −0.0005
    np.testing.assert_allclose(strat, [exp_r1, exp_r2, exp_r3], rtol=1e-12)
    np.testing.assert_allclose(
        equity, np.cumprod([1 + exp_r1, 1 + exp_r2, 1 + exp_r3]), rtol=1e-12)


def test_log_vs_simple_matters():
    """A log-hozam naiv cumprod-ja eltér a helyes egyszerű hozamostól —
    ez volt a V5 M2 hibája."""
    r_log = np.array([0.05, -0.05] * 50)
    _, eq_correct, _ = run_backtest(np.ones_like(r_log), r_log, cost=0.0)
    eq_wrong = np.cumprod(1 + r_log)            # a V5-ös hibás képlet
    assert abs(eq_correct[-1] - eq_wrong[-1]) > 1e-3


def test_merton_fraction_bounds_and_floor():
    f = merton_fraction(np.array([0.001, -0.002, 0.1]),
                        np.array([0.02, 0.02, 1e-9]),
                        cap=0.5, sigma_floor=0.01)
    assert (f >= 0).all() and (f <= 0.5).all()
    assert f[1] == 0.0                          # negatív μ → nincs long pozíció
    # σ-padló: az 1e-9 vol nem robbantja fel a frakciót
    assert f[2] == 0.5


def test_sigma_from_quantiles_normal_case():
    """Normál eloszlásnál (q95−q05)/3.29 ≈ σ."""
    sigma = 0.02
    q05, q95 = -1.645 * sigma, 1.645 * sigma
    assert sigma_from_quantiles(np.array([q05]), np.array([q95]))[0] == \
        pytest.approx(sigma, rel=1e-3)


def test_risk_report_simple_returns():
    r = np.array([0.01, -0.005, 0.002] * 84)
    rep = risk_report(r, "teszt")
    assert rep["MaxDD"] <= 0
    assert rep["VaR(95%)"] >= 0
    assert np.isfinite(rep["Sharpe"])
