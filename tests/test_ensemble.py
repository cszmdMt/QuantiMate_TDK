"""V7-7 tesztek: Vincent-átlag kézi példán, monotonitás, súly-szabályok,
val-only tag-választás, „az átlag veri a tagokat" szintetikus eseten."""
import numpy as np
import pytest

from quantimate.models.ensemble import (choose_ensemble, crps_based_weights,
                                 vincentile_average)
from quantimate.models import pinball_loss_np

QUANTS = (0.05, 0.25, 0.5, 0.75, 0.95)


def test_vincentile_hand_example():
    """Két tag egyenletes átlaga kvantilisenként = számtani közép."""
    a = np.array([[[0.0, 1.0, 2.0, 3.0, 4.0]]])
    b = np.array([[[2.0, 3.0, 4.0, 5.0, 6.0]]])
    out = vincentile_average([a, b])
    np.testing.assert_allclose(out, [[[1.0, 2.0, 3.0, 4.0, 5.0]]])


def test_vincentile_weighted_and_sorted():
    a = np.array([[[0.0, 1.0, 2.0, 3.0, 4.0]]])
    b = np.array([[[4.0, 3.0, 2.0, 1.0, 0.0]]])   # szándékosan fordított
    out = vincentile_average([a, b], weights=[0.75, 0.25])
    assert (np.diff(out, axis=-1) >= 0).all(), "A kimenet nem monoton"
    assert out.shape == a.shape


def test_crps_weights_rules():
    w0 = crps_based_weights([0.5, 0.6, 0.7], eta=0.0)
    np.testing.assert_allclose(w0, 1 / 3)
    w = crps_based_weights([0.5, 0.6, 0.7], eta=3.0)
    assert w[0] > w[1] > w[2] and np.isclose(w.sum(), 1.0)


def _member_preds(seed, n=400, bias=0.0, noise=0.4):
    """Szintetikus tag: a valódi kvantilisek zajos/torzított becslése."""
    rng = np.random.default_rng(seed)
    true_q = np.array([-1.645, -0.674, 0.0, 0.674, 1.645])
    q = true_q[None, None, :] + bias + rng.normal(0, noise, (n, 1, 5))
    return np.sort(q, axis=-1)


def test_averaging_beats_noisy_members():
    """Zajos, ellentétesen torzított tagok átlaga jobb bármelyik tagnál."""
    rng = np.random.default_rng(0)
    n = 400
    Y = rng.normal(0, 1, (n, 1))
    members = {"a": _member_preds(1, n, bias=+0.3),
               "b": _member_preds(2, n, bias=-0.3),
               "c": _member_preds(3, n, bias=+0.2)}
    best_single = min(pinball_loss_np(Y, q, quantiles=QUANTS)
                      for q in members.values())
    ens = vincentile_average(list(members.values()))
    assert pinball_loss_np(Y, ens, quantiles=QUANTS) < best_single


def test_choose_ensemble_val_only_and_deterministic():
    rng = np.random.default_rng(5)
    n = 300
    Y = rng.normal(0, 1, (n, 1))
    members = {f"m{i}": _member_preds(10 + i, n, bias=0.1 * (i - 2))
               for i in range(5)}
    r1 = choose_ensemble(members, Y, QUANTS)
    r2 = choose_ensemble(members, Y, QUANTS)
    assert r1["members"] == r2["members"] and r1["eta"] == r2["eta"]
    assert r1["val_pinball"] <= min(
        pinball_loss_np(Y, q, quantiles=QUANTS) for q in members.values()) + 1e-12
    assert np.isclose(np.sum(r1["weights"]), 1.0)
    assert len(r1["grid"]) >= 6
