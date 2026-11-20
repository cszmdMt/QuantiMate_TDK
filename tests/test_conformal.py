"""7.1: szintetikus Gauss-adaton a CQR lefedettség a névleges ±3 pp-en belül
(n=2000); ACI alapviselkedés."""
import numpy as np
import pytest

from quantimate.validation.conformal import (aci_bands, apply_band, conformity_scores,
                                  cqr_delta, empirical_coverage)


@pytest.mark.parametrize("nominal", [0.85, 0.50])
def test_cqr_coverage_synthetic_gauss(nominal):
    """Szándékosan félrekalibrált (túl keskeny) kvantilis-modell → a CQR
    korrekció a névleges lefedettségre húzza, ±3 pp-en belül n=2000-nél."""
    rng = np.random.default_rng(0)
    n_cal, n_test = 2000, 2000
    y_cal = rng.normal(0, 1, n_cal)
    y_test = rng.normal(0, 1, n_test)

    alpha = 1 - nominal
    z = -np.sort(-np.array([np.abs(np.quantile(rng.normal(0, 1, 10**6),
                                               alpha / 2))]))[0]
    # "Modell": a valódi σ=1 helyett σ=0.6-ot hisz → túl keskeny nyers sáv
    q_lo_cal = np.full(n_cal, -z * 0.6)
    q_hi_cal = np.full(n_cal, z * 0.6)
    q_lo_te = np.full(n_test, -z * 0.6)
    q_hi_te = np.full(n_test, z * 0.6)

    raw_cov = empirical_coverage(y_test, q_lo_te, q_hi_te)
    assert raw_cov < nominal - 0.03, "A nyers sávnak alul kell fednie a teszthez"

    scores = conformity_scores(y_cal, q_lo_cal, q_hi_cal)
    delta = cqr_delta(scores, alpha)
    lo, hi = apply_band(q_lo_te, q_hi_te, delta)
    cov = empirical_coverage(y_test, lo, hi)
    assert abs(cov - nominal) <= 0.03, f"CQR lefedettség {cov:.3f} vs névleges {nominal}"


def test_aci_adapts_and_stays_causal():
    """ACI: eloszlás-eltolódásnál (teszt-σ nő) a lefedettség a cél közelében
    marad; az α-pálya a clip-sávban mozog."""
    rng = np.random.default_rng(1)
    n_cal, n_test = 800, 800
    y_cal = rng.normal(0, 1, n_cal)
    y_test = rng.normal(0, 1.5, n_test)     # eltolódott (szélesebb) teszt

    q_lo_cal, q_hi_cal = np.full(n_cal, -1.0), np.full(n_cal, 1.0)
    q_lo_te, q_hi_te = np.full(n_test, -1.0), np.full(n_test, 1.0)
    scores = conformity_scores(y_cal, q_lo_cal, q_hi_cal)

    lo, hi, alphas, errs = aci_bands(y_test, q_lo_te, q_hi_te, scores,
                                     alpha_target=0.15, gamma=0.01)
    cov = 1 - errs.mean()
    assert abs(cov - 0.85) < 0.06, f"ACI lefedettség messze a céltól: {cov:.3f}"
    assert (alphas >= 0.01).all() and (alphas <= 0.3).all()


def test_cqr_delta_finite_sample_monotone():
    scores = np.arange(100, dtype=float)
    assert cqr_delta(scores, alpha=0.5) <= cqr_delta(scores, alpha=0.1)
