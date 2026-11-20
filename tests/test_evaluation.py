"""7.1: PT-teszt kézzel számolt értékkel, MCC/PT degenerált eset, DM, Wilson,
CRPS-pinball reláció, küszöb-optimalizálás, Bayes-plafon."""
import numpy as np
import pytest

from quantimate.models.bayes import bayes_update, news_signal
from quantimate.validation.evaluation import (best_threshold_by_mcc, crps_from_quantiles,
                                   deflated_sharpe, diebold_mariano,
                                   direction_metrics, pesaran_timmermann,
                                   spearman_ic, wilson_ci)


def test_pesaran_timmermann_hand_example():
    """Kézzel számolt PT: n=10, Py=0.6, Pŷ=0.6, acc=0.8 →
    P*=0.52, V(P̂)=0.02496, V(P*)=0.004224, PT=0.28/√0.020736≈1.9444."""
    y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    yh = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    res = pesaran_timmermann(y, yh)
    assert res["acc"] == pytest.approx(0.8)
    assert res["acc_expected"] == pytest.approx(0.52)
    assert res["stat"] == pytest.approx(0.28 / np.sqrt(0.020736), rel=1e-6)
    assert 0 < res["pvalue"] < 0.05


def test_pesaran_timmermann_degenerate():
    """Konstans ('mindig fel') előrejelzés → nevező 0 → n/a."""
    y = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    yh = np.ones_like(y)
    res = pesaran_timmermann(y, yh)
    assert np.isnan(res["stat"])
    assert "n/a" in res["note"]


def test_direction_metrics_degenerate_flag():
    y = np.array([1, 0, 1, 0, 1])
    p = np.full(5, 0.9)
    m = direction_metrics(y, p)
    assert m["degenerált"] is True
    assert m["MCC"] == 0.0


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    n = 500
    loss_b = rng.uniform(0.5, 1.5, n)
    loss_a = loss_b - 0.3                      # 'a' egyértelműen jobb
    res = diebold_mariano(loss_a, loss_b, h=1)
    assert res["stat"] < -2
    assert res["pvalue"] < 0.05
    same = diebold_mariano(loss_b, loss_b.copy(), h=1)
    assert np.isnan(same["stat"])              # zérus variancia → n/a


def test_spearman_ic_guard():
    assert np.isnan(spearman_ic(np.ones(50), np.random.default_rng(0).normal(size=50))["ic"])
    res = spearman_ic(np.arange(50.0), np.arange(50.0))
    assert res["ic"] == pytest.approx(1.0)


def test_crps_is_twice_mean_pinball():
    from quantimate.models import pinball_loss_np
    rng = np.random.default_rng(2)
    q = (0.05, 0.25, 0.5, 0.75, 0.95)
    y = rng.normal(size=40)
    pred = np.sort(rng.normal(size=(40, 5)), axis=-1)
    assert crps_from_quantiles(y, pred, q) == pytest.approx(
        2 * pinball_loss_np(y, pred, quantiles=q))


def test_wilson_ci_covers_point_estimate():
    lo, hi = wilson_ci(310, 370)
    assert lo < 310 / 370 < hi
    assert hi - lo < 0.1                       # n=370-nél a CI ±~4 pp


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(3)
    trials_few = rng.normal(0.05, 0.02, 3)
    trials_many = rng.normal(0.05, 0.02, 50)
    d_few = deflated_sharpe(0.08, trials_few, 370, skew=0.0, kurt_raw=3.0)
    d_many = deflated_sharpe(0.08, trials_many, 370, skew=0.0, kurt_raw=3.0)
    assert d_many["sr0"] > d_few["sr0"]        # több próba → magasabb mérce


def test_best_threshold_applied_from_val():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 400)
    y = (p + rng.normal(0, 0.3, 400) > 0.55).astype(int)
    t = best_threshold_by_mcc(y, p)
    assert 0.40 <= t <= 0.60


def test_bayes_lr_never_exits_cap_band():
    """Az LR (az odds-szorzó) extrém hír-jelnél sem lép ki
    [1/1.5, 1.5]-ből → a poszterior-odds arány korlátos."""
    extreme_cases = [
        {"sentiment": 1.0, "confidence": 1.0, "fundamental_score": 1.0,
         "geopolitical_risk": 0.0},
        {"sentiment": -1.0, "confidence": 1.0, "fundamental_score": -1.0,
         "geopolitical_risk": 1.0},
        {"sentiment": 5.0, "confidence": 3.0, "fundamental_score": 4.0,
         "geopolitical_risk": -2.0},           # szándékosan érvénytelen inputok
    ]
    for case in extreme_cases:
        for prior in (0.2, 0.5, 0.8):
            post = bayes_update(prior, case, k=1.2)
            prior_odds = prior / (1 - prior)
            post_odds = post / (1 - post)
            lr_implied = post_odds / prior_odds
            assert 1 / 1.5 - 1e-9 <= lr_implied <= 1.5 + 1e-9


def test_bayes_neutral_is_identity():
    neutral = {"sentiment": 0.0, "confidence": 0.0, "fundamental_score": 0.0,
               "geopolitical_risk": 0.5}
    assert news_signal(neutral) == pytest.approx(0.0)
    assert bayes_update(0.55, neutral) == pytest.approx(0.55)
