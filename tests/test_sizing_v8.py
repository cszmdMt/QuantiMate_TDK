"""V8-11 konformális-sáv pozícióméretezés tesztek: bizonytalanság-érzékenység,
cap, long-only, kauzális simítás, val-kalibráció kitettség-egyezése."""
import numpy as np

from quantimate.backtest import (calibrate_sizing_k, conformal_sizing,
                                 ewma_smooth, merton_fraction)


def test_wider_band_means_smaller_position():
    med = np.full(50, 0.01)
    narrow = conformal_sizing(med, np.full(50, 0.02), k=1.0)
    wide = conformal_sizing(med, np.full(50, 0.08), k=1.0)
    assert (narrow[10:] > wide[10:]).all()


def test_long_only_and_cap():
    med = np.array([-0.02, 0.0, 0.01, 5.0])
    pos = conformal_sizing(med, np.full(4, 0.02), k=1.0, cap=0.5)
    assert pos[0] == 0.0 and pos[1] == 0.0
    assert 0 < pos[2] <= 0.5
    assert pos[3] == 0.5          # cap érvényesül


def test_ewma_smooth_is_causal():
    """A múltbeli simított érték nem változik, ha a jövő megváltozik."""
    rng = np.random.default_rng(0)
    w = rng.uniform(0.01, 0.05, 100)
    a = ewma_smooth(w, lam=0.9)
    w2 = w.copy()
    w2[60:] = 10.0
    b = ewma_smooth(w2, lam=0.9)
    np.testing.assert_array_equal(a[:60], b[:60])
    assert (b[60:] > a[60:]).all()


def test_smoothing_reduces_position_noise():
    """Zajos sávszélesség mellett a simított pozíció-sor szórása kisebb,
    mint a nyers (λ=0) változaté — a kötelező simítás értelme."""
    rng = np.random.default_rng(1)
    med = np.full(300, 0.01)
    width = 0.04 + rng.normal(0, 0.01, 300).clip(-0.02, 0.02)
    smooth = conformal_sizing(med, width, k=1.0, lam=0.94, cap=np.inf)
    raw = conformal_sizing(med, width, k=1.0, lam=0.0, cap=np.inf)
    assert np.std(np.diff(smooth)) < np.std(np.diff(raw))


def test_calibrated_k_matches_reference_exposure():
    """A val-kalibrált k mellett az átlagkitettség a fél-Mertonéval egyezik
    (cap-be nem ütköző tartományban) — az A/B kitettség-semleges."""
    rng = np.random.default_rng(2)
    med = np.abs(rng.normal(0.005, 0.003, 400))
    width = rng.uniform(0.03, 0.06, 400)
    sigma = rng.uniform(0.015, 0.03, 400)
    ref = merton_fraction(med, sigma, half=True, cap=0.2)
    k = calibrate_sizing_k(med, width, ref, cap=0.2)
    pos = conformal_sizing(med, width, k=k, cap=np.inf)
    np.testing.assert_allclose(np.mean(pos), np.mean(np.abs(ref)), rtol=1e-9)


def test_zero_signal_gives_zero_k():
    assert calibrate_sizing_k(np.zeros(10), np.full(10, 0.05),
                              np.full(10, 0.1)) == 1.0


def test_v8_live_preset_only_gate_passed_diffs():
    """A V8_LIVE presetben KIZÁRÓLAG a kapun átment
    beállítás térhet el a defaulttól — jelenleg egyedül a sizing_mode."""
    from dataclasses import fields

    from quantimate.config import DEFAULT_CONFIG, V8_LIVE_CONFIG

    diffs = {f.name for f in fields(DEFAULT_CONFIG)
             if getattr(DEFAULT_CONFIG, f.name) != getattr(V8_LIVE_CONFIG, f.name)}
    assert diffs == {"sizing_mode"}
    assert V8_LIVE_CONFIG.sizing_mode == "conformal"
