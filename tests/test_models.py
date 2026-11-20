"""Modell-modul tesztek: pinball loss helyessége, kvantilis-rendezés,
P(y>0)-interpoláció, RevIN alak/normálás, determinizmus."""
import numpy as np
import pytest

from quantimate.models import (QUANTILES, RevIN, make_pinball_loss,
                               pinball_loss_np, prob_up_from_quantiles)


def test_pinball_loss_hand_example():
    """Kézzel számolt pinball: y=0, ŷ_q=1 minden q-ra → L_q = (1−q)·1."""
    import tensorflow as tf
    loss_fn = make_pinball_loss(quantiles=(0.1, 0.5, 0.9), n_horizons=1)
    y_true = tf.constant([[0.0]])
    y_pred = tf.constant([[1.0, 1.0, 1.0]])
    # e = −1 → max(q·e, (q−1)e) = (1−q); átlag = ((0.9)+(0.5)+(0.1))/3 = 0.5
    val = float(loss_fn(y_true, y_pred).numpy())
    assert val == pytest.approx(0.5, abs=1e-6)
    # numpy megfelelő ugyanezt adja
    np_val = pinball_loss_np(np.array([0.0]), np.array([[1.0, 1.0, 1.0]]),
                             quantiles=(0.1, 0.5, 0.9))
    assert np_val == pytest.approx(val, abs=1e-6)


def test_pinball_asymmetry():
    """Alulbecslésnél a magas kvantilis büntet jobban — a loss aszimmetrikus."""
    over = pinball_loss_np(np.array([0.0]), np.array([[1.0]]), quantiles=(0.9,))
    under = pinball_loss_np(np.array([0.0]), np.array([[-1.0]]), quantiles=(0.9,))
    assert under == pytest.approx(0.9)
    assert over == pytest.approx(0.1)


def test_prob_up_interpolation():
    taus = QUANTILES
    # Szimmetrikus kvantilisek 0 körül → P(y>0) = 0.5
    q = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    assert prob_up_from_quantiles(q, taus)[0] == pytest.approx(0.5)
    # Minden kvantilis pozitív → P(y>0) magas
    q_pos = np.array([[0.5, 1.0, 1.5, 2.0, 2.5]])
    assert prob_up_from_quantiles(q_pos, taus)[0] > 0.9
    # Minden kvantilis negatív → P(y>0) alacsony
    q_neg = -q_pos[:, ::-1]
    assert prob_up_from_quantiles(q_neg, taus)[0] < 0.1
    # A 0 pont a q25=−1 és q50=+1 között félúton → P(Y≤0)=0.375
    q_mid = np.array([[-3.0, -1.0, 1.0, 3.0, 5.0]])
    assert prob_up_from_quantiles(q_mid, taus)[0] == pytest.approx(1 - 0.375)


def test_revin_normalises_and_learnable_shape():
    import tensorflow as tf
    x = tf.random.normal((4, 20, 7)) * 5.0 + 3.0
    layer = RevIN()
    y = layer(x).numpy()
    # Kezdeti (gamma=1, beta=0) állapotban ablakonként ~0 átlag, ~1 szórás
    np.testing.assert_allclose(y.mean(axis=1), 0.0, atol=1e-4)
    np.testing.assert_allclose(y.std(axis=1), 1.0, atol=1e-2)
    assert len(layer.trainable_weights) == 2


def test_training_deterministic_same_seed():
    """Ugyanazzal a seeddel két tanítás bitre azonos predikciót ad (2.5)."""
    from quantimate.models import build_nlinear, train_quantile_model, predict_quantiles
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (64, 10, 3)).astype(np.float32)
    Y = rng.normal(0, 1, (64, 3)).astype(np.float32)
    Xv, Yv = X[:16], Y[:16]

    m1, _ = train_quantile_model(build_nlinear, X, Y, Xv, Yv, seed=7, epochs=2)
    p1 = predict_quantiles(m1, Xv)
    m2, _ = train_quantile_model(build_nlinear, X, Y, Xv, Yv, seed=7, epochs=2)
    p2 = predict_quantiles(m2, Xv)
    np.testing.assert_array_equal(p1, p2)


def test_predicted_quantiles_sorted():
    from quantimate.models import build_nlinear, train_quantile_model, predict_quantiles
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (64, 10, 3)).astype(np.float32)
    Y = rng.normal(0, 1, (64, 3)).astype(np.float32)
    m, _ = train_quantile_model(build_nlinear, X, Y, X[:16], Y[:16], seed=42, epochs=1)
    q = predict_quantiles(m, X[:16])
    assert q.shape == (16, 3, 5)
    assert (np.diff(q, axis=-1) >= 0).all(), "A kvantilisek nem monotonok rendezés után"
