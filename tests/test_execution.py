"""V7-14 tesztek: Corwin–Schultz spread nemnegatív+kauzális, dinamikus
költség, részleges igazítás és no-trade sáv forgalom-csökkentése."""
import numpy as np
import pandas as pd
import pytest

from quantimate.backtest.execution import (corwin_schultz_spread, dynamic_cost,
                                  no_trade_band, partial_rebalance, turnover)


def _hl_frame(n=100, seed=0, spread=0.01):
    rng = np.random.default_rng(seed)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2024-01-01", periods=n)
    high = pd.Series(mid * (1 + spread / 2 + np.abs(rng.normal(0, 0.003, n))), index=idx)
    low = pd.Series(mid * (1 - spread / 2 - np.abs(rng.normal(0, 0.003, n))), index=idx)
    return high, low


def test_cs_spread_nonnegative():
    high, low = _hl_frame()
    s = corwin_schultz_spread(high, low)
    assert (s.dropna() >= 0).all()
    assert np.isnan(s.iloc[0])                 # az első napnak nincs t-1 párja


def test_cs_spread_causal():
    """A t-edik CS-érték a jövő elrontásától független (t és t-1 H/L-t használ)."""
    high, low = _hl_frame()
    base = corwin_schultz_spread(high, low)
    high2 = high.copy(); low2 = low.copy()
    high2.iloc[50:] *= 1.5; low2.iloc[50:] *= 0.5
    fut = corwin_schultz_spread(high2, low2)
    np.testing.assert_array_equal(base.iloc[:50].to_numpy(),
                                  fut.iloc[:50].to_numpy())


def test_cs_spread_detects_wider_spread():
    """Szélesebb valódi spread → nagyobb becslés (átlagban)."""
    h_narrow, l_narrow = _hl_frame(spread=0.005, seed=1)
    h_wide, l_wide = _hl_frame(spread=0.03, seed=1)
    assert corwin_schultz_spread(h_wide, l_wide).mean() > \
        corwin_schultz_spread(h_narrow, l_narrow).mean()


def test_dynamic_cost_floor():
    cost = dynamic_cost(np.array([0.0, 0.002, 0.02]), fixed_floor=0.0005)
    assert cost[0] == 0.0005                    # padló
    assert cost[1] == 0.001                     # spread/2
    assert cost[2] == 0.01


def test_partial_rebalance_reduces_turnover():
    """Zajos cél-pozíció → a részleges igazítás csökkenti a forgalmat."""
    rng = np.random.default_rng(2)
    target = np.clip(0.3 + rng.normal(0, 0.15, 200), 0, 0.5)
    full_to = turnover(target)
    partial_to = turnover(partial_rebalance(target, tau=0.3))
    assert partial_to < full_to * 0.8


def test_no_trade_band_reduces_turnover():
    rng = np.random.default_rng(3)
    target = np.clip(0.3 + rng.normal(0, 0.1, 200), 0, 0.5)
    band_to = turnover(no_trade_band(target, delta=0.08))
    assert band_to < turnover(target)


def test_partial_rebalance_converges():
    """Konstans célra a részleges igazítás a célhoz konvergál."""
    target = np.full(50, 0.4)
    out = partial_rebalance(target, tau=0.5)
    assert out[-1] == pytest.approx(0.4, abs=1e-6)
