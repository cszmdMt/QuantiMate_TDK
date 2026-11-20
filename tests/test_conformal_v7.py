"""V7-9 tesztek: a marginális CQR kategórián belül téveszthet, a Mondrian
javít; kis kategória (n<min_n) → globális visszaesés jelölve."""
import numpy as np
import pytest

from quantimate.validation.conformal import (agaci_bands, apply_mondrian_band,
                                  conformity_scores, coverage_by_category,
                                  cqr_delta, empirical_coverage,
                                  mondrian_cqr_delta, nexcp_delta,
                                  weighted_quantile)

NOMINAL = 0.85
ALPHA = 1 - NOMINAL


@pytest.fixture(scope="module")
def two_regime_data():
    """Két rezsim, ELTÉRŐ zajszinttel: a modell sávja mindkettőn azonos (túl
    keskeny a zajosra, túl bő a nyugodtra) — pont a Mondrian esete."""
    rng = np.random.default_rng(0)
    n = 4000
    cats = (rng.uniform(0, 1, n) < 0.4).astype(int)     # 1 = turbulens
    sigma = np.where(cats == 1, 3.0, 1.0)
    y = rng.normal(0, sigma)
    q_lo = np.full(n, -1.8)
    q_hi = np.full(n, 1.8)
    half = n // 2
    return (y[:half], q_lo[:half], q_hi[:half], cats[:half],
            y[half:], q_lo[half:], q_hi[half:], cats[half:])


def test_marginal_cqr_fails_per_category(two_regime_data):
    y_cal, lo_cal, hi_cal, c_cal, y_te, lo_te, hi_te, c_te = two_regime_data
    scores = conformity_scores(y_cal, lo_cal, hi_cal)
    d = cqr_delta(scores, ALPHA)
    lo, hi = lo_te - d, hi_te + d

    assert abs(empirical_coverage(y_te, lo, hi) - NOMINAL) < 0.03, \
        "A marginális lefedettségnek rendben kell lennie"
    rows = coverage_by_category(y_te, lo, hi, c_te)
    covs = {r["kategória"]: r["empirikus"] for r in rows}
    assert covs["0"] > NOMINAL + 0.05, "A nyugodt rezsimben túlfedésnek kell lennie"
    assert covs["1"] < NOMINAL - 0.05, "A turbulens rezsimben alulfedésnek kell lennie"


def test_mondrian_fixes_per_category(two_regime_data):
    y_cal, lo_cal, hi_cal, c_cal, y_te, lo_te, hi_te, c_te = two_regime_data
    scores = conformity_scores(y_cal, lo_cal, hi_cal)
    deltas = mondrian_cqr_delta(scores, c_cal, ALPHA, min_n=50)
    lo, hi = apply_mondrian_band(lo_te, hi_te, c_te, deltas)

    for r in coverage_by_category(y_te, lo, hi, c_te):
        assert abs(r["empirikus"] - NOMINAL) < 0.05, \
            f"Mondrian után a(z) {r['kategória']} rezsim lefedettsége billeg: {r['empirikus']:.3f}"
    assert not deltas[0]["fallback"] and not deltas[1]["fallback"]


def test_small_category_falls_back_to_global(two_regime_data):
    """Ritka kategória → globális δ, fallback-jelzéssel."""
    y_cal, lo_cal, hi_cal, c_cal, *_ = two_regime_data
    c_rare = c_cal.copy()
    c_rare[:] = 0
    c_rare[:20] = 2                                     # 20 elemű ritka kategória
    scores = conformity_scores(y_cal, lo_cal, hi_cal)
    deltas = mondrian_cqr_delta(scores, c_rare, ALPHA, min_n=50)
    assert deltas[2]["fallback"] is True
    assert deltas[2]["delta"] == deltas["__global__"]
    assert deltas[0]["fallback"] is False


def test_coverage_by_category_wilson_contains_point():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 300)
    rows = coverage_by_category(y, np.full(300, -2.0), np.full(300, 2.0),
                                np.zeros(300, dtype=int),
                                category_names={0: "minden"})
    r = rows[0]
    assert r["kategória"] == "minden" and r["n"] == 300
    assert r["Wilson alsó"] <= r["empirikus"] <= r["Wilson felső"]

# ── V7-10: NexCP + AgACI ──────────────────────────────────────────────────────

def test_weighted_quantile_uniform_matches_numpy():
    v = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    w = np.ones(5)
    assert weighted_quantile(v, w, 0.5) == pytest.approx(np.quantile(v, 0.5,
                                                         method="higher"))


def test_nexcp_rho_one_equals_standard_quantile():
    """ρ=1 → a NexCP a standard CQR-korrekcióval egyezik (súlyozatlan)."""
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 500)
    d_nexcp = nexcp_delta(scores, alpha=0.15, rho=1.0)
    d_cqr = cqr_delta(scores, alpha=0.15)
    assert d_nexcp == pytest.approx(d_cqr, abs=1e-9)


def test_nexcp_downweights_old_scores():
    """Ha a régi pontszámok nagyok, az új kicsik, a lecsengő súly (ρ<1) kisebb
    korrekciót ad, mint a súlyozatlan."""
    scores = np.concatenate([np.full(200, 5.0), np.full(200, 0.5)])  # régi nagy
    d_weighted = nexcp_delta(scores, alpha=0.15, rho=0.97)
    d_uniform = nexcp_delta(scores, alpha=0.15, rho=1.0)
    assert d_weighted < d_uniform


def test_agaci_coverage_near_target_under_shift():
    """AgACI eloszlás-eltolódásnál a cél közelében tartja a lefedettséget,
    az α-pálya a clip-sávban marad."""
    rng = np.random.default_rng(1)
    n_cal, n_te = 600, 600
    y_cal = rng.normal(0, 1, n_cal)
    y_te = rng.normal(0, 1.6, n_te)                 # szélesebb teszt
    q_lo_cal, q_hi_cal = np.full(n_cal, -1.0), np.full(n_cal, 1.0)
    q_lo_te, q_hi_te = np.full(n_te, -1.0), np.full(n_te, 1.0)
    scores = conformity_scores(y_cal, q_lo_cal, q_hi_cal)

    lo, hi, alphas, errs = agaci_bands(y_te, q_lo_te, q_hi_te, scores,
                                       alpha_target=0.15)
    cov = 1 - errs.mean()
    assert abs(cov - 0.85) < 0.07
    assert (alphas >= 0.01).all() and (alphas <= 0.3).all()


# ── V7-11: CopulaCPTS együttes lefedettség ────────────────────────────────────

def test_copula_joint_beats_naive_and_matches_bonferroni_direction():
    """Korrelált horizont-hibáknál a CopulaCPTS együttes régiója SZŰKEBB
    marginális szintet igényel, mint a (túl konzervatív) Bonferroni, miközben
    a naiv (független) sáv alulfedi az együttest."""
    from quantimate.validation.conformal import (bonferroni_level, copula_joint_level,
                                      joint_coverage)
    rng = np.random.default_rng(0)
    n, H = 3000, 3
    # erősen korrelált horizont-pontszámok (közös piaci sokk)
    common = rng.normal(0, 1, (n, 1))
    cal = common + rng.normal(0, 0.3, (n, H))
    target = 0.85

    u_cop = copula_joint_level(cal, target)
    u_bonf = bonferroni_level(target, H)
    # a Bonferroni konzervatívabb (magasabb marginális szint) korrelált hibáknál
    assert u_cop < u_bonf

    # teszt: a kopula-szint együttes lefedettsége a cél közelében
    test = common[:1000] * 0 + rng.normal(0, 1, (1000, 1)) + rng.normal(0, 0.3, (1000, H))
    # küszöb = a kalibrációs pontszámok u_cop-kvantilise horizontonként
    thr = np.quantile(cal, u_cop, axis=0)
    cov = joint_coverage(test, thr)
    assert cov >= target - 0.08         # együttes lefedettség a cél közelében/felett


def test_bonferroni_level_formula():
    from quantimate.validation.conformal import bonferroni_level
    assert bonferroni_level(0.85, 3) == pytest.approx(1 - 0.15 / 3)
    assert bonferroni_level(0.9, 1) == pytest.approx(0.9)


def test_joint_coverage_all_horizons_required():
    from quantimate.validation.conformal import joint_coverage
    scores = np.array([[0.1, 0.2, 0.3], [0.1, 0.9, 0.1], [0.5, 0.5, 0.5]])
    thr = np.array([0.6, 0.6, 0.6])
    # sor0: mind ≤0.6 → benne; sor1: 0.9>0.6 → kívül; sor2: mind ≤0.6 → benne
    assert joint_coverage(scores, thr) == pytest.approx(2 / 3)
