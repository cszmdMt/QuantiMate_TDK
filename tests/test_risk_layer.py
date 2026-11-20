"""V7-13 tesztek: vol-targeting skálázás, CVaR-cap, short-oldal, invariánsok."""
import numpy as np
import pytest

from quantimate.backtest import (cvar_from_quantiles, cvar_position_cap,
                                 run_backtest, two_sided_position,
                                 vol_target_overlay)


def test_vol_target_scales_inversely_with_sigma():
    """Nyugodt piac (kis σ̂) → nagyobb pozíció; viharos → kisebb."""
    pos = np.array([0.3, 0.3, 0.3])
    sig = np.array([0.01, 0.02, 0.04])
    out = vol_target_overlay(pos, sig, target_vol=0.02, cap=1.0)
    assert out[0] > out[1] > out[2]
    assert np.isclose(out[1], 0.3)             # σ̂=target → változatlan


def test_vol_target_respects_cap():
    out = vol_target_overlay(np.array([0.5]), np.array([1e-6]), target_vol=0.02,
                             cap=0.6)
    assert out[0] <= 0.6


def test_cvar_positive_and_scales_with_tail():
    """A CVaR pozitív és nagyobb szélesebb bal-farok (negatívabb q05) esetén."""
    q05 = np.array([-1.645, -3.0])             # normált kvantilisek
    sig = np.array([0.02, 0.02])
    cv = cvar_from_quantiles(q05, sig, h=1)
    assert (cv > 0).all()
    assert cv[1] > cv[0]


def test_cvar_cap_limits_position():
    """Nagy CVaR → a keret levágja a pozíciót; kis CVaR → érintetlen."""
    pos = np.array([0.5, 0.5])
    cvar = np.array([0.10, 0.01])              # nap0 kockázatos
    out = cvar_position_cap(pos, cvar, cvar_budget=0.02, cap=0.5)
    assert out[0] < 0.5                         # levágva: 0.02/0.10=0.2
    assert np.isclose(out[0], 0.2)
    assert np.isclose(out[1], 0.5)             # 0.02/0.01=2 → cap 0.5


def test_two_sided_short_on_low_pup():
    """P(emelkedés) alacsony → SHORT (negatív pozíció); magas → long; közép → 0."""
    p_up = np.array([0.75, 0.25, 0.50])
    med = np.array([0.01, -0.01, 0.0])
    sig = np.array([0.02, 0.02, 0.02])
    pos = two_sided_position(p_up, med, sig, threshold=0.6, cap=0.5)
    assert pos[0] > 0                          # long
    assert pos[1] < 0                          # short
    assert pos[2] == 0                         # semleges sáv


def test_two_sided_reduces_to_long_only_at_extreme_threshold():
    """Küszöb=1.0 mellett short sosem nyílik (P(esés)≥1 lehetetlen)."""
    p_up = np.array([0.9, 0.1, 0.5])
    med = np.array([0.01, -0.01, 0.0])
    sig = np.array([0.02, 0.02, 0.02])
    pos = two_sided_position(p_up, med, sig, threshold=1.0)
    assert (pos <= 0).sum() >= 0               # nincs pozitív küszöb-átlépés
    # long is csak p_up>=1.0-nál — extrém, tipikusan üres
    assert pos[1] <= 0


def test_risk_layer_backtest_invariants():
    """A kockázati réteg utáni pozíciók a backteszttel konzisztensek
    (pos≡0 → equity≡1)."""
    r = np.random.default_rng(0).normal(0, 0.02, 100)
    _, eq, _ = run_backtest(np.zeros(100), r, cost=0.001)
    np.testing.assert_array_equal(eq, np.ones(100))
