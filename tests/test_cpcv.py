"""V7-15 tesztek: CPCV split-diszjunktság+purge, PBO CSCV (overfit vs valódi
jel), SPA (valódi él vs zaj), stationary bootstrap tulajdonságok."""
import numpy as np
import pytest

from quantimate.validation.validation_cpcv import (cpcv_splits, pbo_cscv, spa_test,
                                        stationary_bootstrap_indices)


def test_cpcv_paths_count_and_disjoint():
    """C(8,2)=28 útvonal; minden úton a train és teszt diszjunkt, és a
    teszt-blokkok körüli purge+embargo sáv sincs a trainben."""
    n, n_groups, k, purge, embargo = 2000, 8, 2, 21, 5
    paths = list(cpcv_splits(n, n_groups, k, purge, embargo))
    assert len(paths) == 28

    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    for tr, te in paths:
        assert len(set(tr) & set(te)) == 0
        # a teszt-blokk határai körüli gap egyik sora sincs a trainben
        for g in range(n_groups):
            block = np.arange(bounds[g], bounds[g + 1])
            if set(block) <= set(te):
                lo = max(0, bounds[g] - (purge + embargo))
                hi = min(n, bounds[g + 1] + (purge + embargo))
                assert len(set(range(lo, hi)) & set(tr)) == 0


def test_pbo_high_for_pure_noise():
    """Csupa zaj-konfigurációk → a PBO ~0.5 körül (nincs valódi győztes)."""
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (1000, 20))
    res = pbo_cscv(R, n_blocks=10)
    assert 0.3 < res["pbo"] < 0.7
    assert res["n_combinations"] == 252   # C(10,5)


def test_pbo_low_for_one_genuine_edge():
    """Egy valódi él + sok zaj → alacsony PBO (a győztes OOS is jó)."""
    rng = np.random.default_rng(1)
    T = 1500
    R = rng.normal(0, 0.01, (T, 15))
    R[:, 0] += 0.0015                     # 0. konfiguráció valódi pozitív drift
    res = pbo_cscv(R, n_blocks=10)
    assert res["pbo"] < 0.2


def test_spa_detects_genuine_strategy():
    """Egy valóban jobb stratégia + zajosak → kicsi SPA-p (H0 elvetve)."""
    rng = np.random.default_rng(2)
    T, K = 1500, 8
    bench = rng.normal(0.0002, 0.01, T)
    # A többi stratégia a benchmark zajos, enyhén rosszabb mása → d_bar ~ 0 alatt,
    # a 3. egyértelmű (nagy, kis szórású) többlettel — így best_k stabil.
    strat = bench[:, None] + rng.normal(-0.0001, 0.004, (T, K))
    strat[:, 3] = bench + rng.normal(0.0012, 0.003, T)
    res = spa_test(strat, bench, n_boot=500, seed=1)
    assert res["pvalue"] < 0.05
    assert res["best_k"] == 3


def test_spa_null_when_no_edge():
    """Ha egyik stratégia sem jobb a benchmarknál → nagy SPA-p."""
    rng = np.random.default_rng(3)
    T, K = 1000, 8
    bench = rng.normal(0.0005, 0.01, T)
    strat = bench[:, None] + rng.normal(-0.0002, 0.008, (T, K))  # kissé rosszabbak
    res = spa_test(strat, bench, n_boot=500, seed=2)
    assert res["pvalue"] > 0.10


def test_spa_pvalue_is_one_when_every_strategy_is_clearly_worse():
    """Hansen (2005): T = max(max_k t_k, 0). Ha minden jelölt szignifikánsan
    ROSSZABB a benchmarknál, a statisztika 0 és a p-érték 1 — nem ~0,5, amit a
    nullánál vágás nélküli változat adott (félrevezető "majdnem jó" olvasat)."""
    rng = np.random.default_rng(4)
    T = 1000
    bench = rng.normal(0.0005, 0.01, T)
    strat = bench[:, None] + rng.normal(-0.002, 0.004, (T, 3))   # ~16 standard hibával rosszabbak
    res = spa_test(strat, bench, n_boot=500, seed=3)
    assert res["stat"] == 0.0
    assert res["pvalue"] == 1.0


def test_spa_single_candidate_is_a_calibrated_one_sided_test():
    """K=1: egyoldali bootstrap-próba (a kísérleti kapu 2. lépcsője így használja).
    Valódi többletnél kicsi a p-érték; nulla többletnél az elutasítási arány a
    névleges 10% körül marad (150 független ismétlés, rögzített seed)."""
    rng = np.random.default_rng(5)
    T = 1500
    bench = rng.normal(0.0, 0.01, T)
    assert spa_test(bench + rng.normal(0.0006, 0.004, T), bench, n_boot=500, seed=4)["pvalue"] < 0.01

    rejections = 0
    for i in range(150):
        bench = rng.normal(0.0, 0.01, 250)
        same = bench + rng.normal(0.0, 0.004, 250)
        rejections += spa_test(same, bench, n_boot=200, seed=i)["pvalue"] < 0.10
    assert 0.03 <= rejections / 150 <= 0.19, rejections / 150


def test_stationary_bootstrap_covers_and_deterministic():
    rng = np.random.default_rng(7)
    idx = stationary_bootstrap_indices(200, avg_block=10, rng=rng)
    assert idx.shape == (200,) and idx.min() >= 0 and idx.max() < 200
    r1 = stationary_bootstrap_indices(50, 5, np.random.default_rng(1))
    r2 = stationary_bootstrap_indices(50, 5, np.random.default_rng(1))
    np.testing.assert_array_equal(r1, r2)
