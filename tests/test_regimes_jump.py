"""V7-8 tesztek: KÖTELEZŐ prefix-invariancia (kauzalitás), a jump-modell
elkülöníti a rezsimeket, perzisztencia, offline-dekódolás NEM kauzális (kontroll)."""
import numpy as np
import pytest

from quantimate.models.regimes_jump import (JumpModel, fit_jump_model, persistence,
                                     _assign_states_dp)


@pytest.fixture(scope="module")
def two_regime_series():
    """Két rezsim: nyugodt (kis vol, pozitív hozam) és turbulens (nagy vol)."""
    rng = np.random.default_rng(0)
    T = 600
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        p_stay = 0.98
        states[t] = states[t - 1] if rng.random() < p_stay else 1 - states[t - 1]
    mu = np.array([[0.002, 0.01], [-0.003, 0.08]])
    x = mu[states] + rng.normal(0, 0.004, (T, 2))
    # A valós pipeline RobustScaler-rel egység-skálára hozza a jellemzőket —
    # a jump-penalty (λ) ehhez van kalibrálva; a teszt is standardizál.
    x = (x - x[:400].mean(axis=0)) / x[:400].std(axis=0)
    return x, states


@pytest.fixture(scope="module")
def fitted(two_regime_series):
    x, _ = two_regime_series
    model = fit_jump_model(x[:400], n_states=2, lam=5.0, seed=7)
    return model, x


def test_forward_states_prefix_invariant(fitted):
    """A t-ig tartó prefixen dekódolt forward-állapot BITRE azonos a
    teljes soron dekódolt t-edikkel — a kauzalitás bizonyítéka."""
    model, x = fitted
    full = model.forward_states(x)
    for t in (1, 10, 100, 300, len(x) - 1):
        prefix = model.forward_states(x[: t + 1])
        assert prefix[-1] == full[t], f"t={t}: a forward jump-állapot nem kauzális!"


def test_forward_proba_prefix_invariant(fitted):
    model, x = fitted
    full = model.forward_proba(x)
    for t in (5, 50, 250):
        prefix = model.forward_proba(x[: t + 1])
        np.testing.assert_allclose(prefix[-1], full[t], atol=1e-12)


def test_offline_dp_differs_from_forward_causal():
    """Kontroll (illusztráció): egy magányos 'tüske' state-1 felé, state-0
    környezetben, magas λ mellett. A GLOBÁLIS (offline) optimum végigmarad
    state-0-ban (a tüskénél váltani + visszaváltani 2λ-t költene), a per-t
    FORWARD argmin viszont a tüskénél state-1-re billen (lokálisan olcsóbb
    emisszió). A kettő tehát eltér — ezért kell a forward-út a jellemzőhöz."""
    # λ=0.7 ∈ (0.5, 1): a forward a tüskénél vált (λ < 1 emisszió-megtakarítás),
    # az offline stay-cost (1) < switch-and-return (2λ=1.4) → végig state-0.
    mu = np.array([[0.0], [1.0]])
    x = np.zeros((11, 1))
    x[5, 0] = 1.0                      # egyetlen tüske a state-1 centroid felé
    model = JumpModel(2, lam=0.7, means=mu)

    forward = model.forward_states(x)
    offline = _assign_states_dp(x, mu, lam=0.7)
    assert offline[5] == 0, "A magas λ-nak végig state-0-ban kellene tartania (offline)"
    assert forward[5] == 1, "A forward per-t argmin a tüskénél state-1-re billen"
    assert not np.array_equal(forward, offline)


def test_jump_model_separates_regimes(fitted, two_regime_series):
    """A dekódolt állapotok korrelálnak a valódi rezsimekkel."""
    model, x = fitted
    _, true_states = two_regime_series
    pred = model.forward_states(x)
    # az egyezés vagy a komplemens (címke-permutáció) legalább 80%
    agree = max((pred == true_states).mean(), (pred != true_states).mean())
    assert agree > 0.8


def test_persistence_positive(fitted):
    model, x = fitted
    p = persistence(model.forward_states(x))
    assert p > 1.0
