"""V7-12 tesztek: triple-barrier helyes gát-érintés, purge újraszámítás,
meta-címke logika, meta-pozíció szűrés+méretezés, determinizmus."""
import numpy as np
import pandas as pd
import pytest

from quantimate.models.metalabel import (make_meta_labels, meta_position_size,
                                  required_purge, triple_barrier_labels)


def test_triple_barrier_hits_upper():
    """Monoton emelkedő ár → a felső gátat éri el (label +1) a σ-tól függő napon."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = pd.Series(100 * np.exp(np.arange(10) * 0.03), index=idx)  # +3%/nap
    sigma = pd.Series(np.full(10, 0.02), index=idx)
    tb = triple_barrier_labels(close, sigma, pt=1.5, sl=1.5, max_holding=5)
    # a 0. napon: felső gát = 1.5*0.02*sqrt(1)=0.03 az 1. napra; cum log-ret nap1 = 0.03 → érint
    assert tb["tb_label"].iloc[0] == 1
    assert tb["tb_holding"].iloc[0] >= 1


def test_triple_barrier_hits_lower():
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = pd.Series(100 * np.exp(-np.arange(10) * 0.03), index=idx)  # −3%/nap
    sigma = pd.Series(np.full(10, 0.02), index=idx)
    tb = triple_barrier_labels(close, sigma, pt=1.5, sl=1.5, max_holding=5)
    assert tb["tb_label"].iloc[0] == -1


def test_triple_barrier_time_limit():
    """Oldalazó ár (a gátakat nem éri el) → időkorlát-címke (0), holding=max."""
    idx = pd.bdate_range("2024-01-01", periods=20)
    rng = np.random.default_rng(0)
    close = pd.Series(100 + rng.normal(0, 0.05, 20).cumsum() * 0.01, index=idx)
    sigma = pd.Series(np.full(20, 0.1), index=idx)   # nagy σ → messzi gátak
    tb = triple_barrier_labels(close, sigma, pt=3.0, sl=3.0, max_holding=5)
    assert tb["tb_label"].iloc[0] == 0
    assert tb["tb_holding"].iloc[0] == 5


def test_triple_barrier_last_row_no_future():
    idx = pd.bdate_range("2024-01-01", periods=5)
    close = pd.Series([100., 101, 102, 103, 104], index=idx)
    sigma = pd.Series(np.full(5, 0.02), index=idx)
    tb = triple_barrier_labels(close, sigma)
    assert tb["tb_label"].iloc[-1] == 0 and tb["tb_holding"].iloc[-1] == 0


def test_required_purge_recalc():
    """Purge = max(H_max, max_holding)."""
    assert required_purge(21, 5) == 21
    assert required_purge(21, 30) == 30      # hosszabb tartás → nagyobb purge
    assert required_purge(5, 10) == 10


def test_meta_labels_agreement():
    """Meta-címke 1, ha az elsődleges irány egyezik a realizált hozam előjelével."""
    primary = np.array([1, 1, -1, -1, 1])
    tb_ret = np.array([0.02, -0.01, -0.03, 0.02, 0.0])
    meta = make_meta_labels(primary, tb_ret)
    # nap0: +1 vs +0.02 → 1; nap1: +1 vs −0.01 → 0; nap2: −1 vs −0.03 → 1;
    # nap3: −1 vs +0.02 → 0; nap4: +1 vs 0.0 → sign(0)=0 ≠ 1 → 0
    assert list(meta) == [1, 0, 1, 0, 0]


def test_meta_position_filters_and_sizes():
    """Alacsony P_meta → kis pozíció; az elsődleges irány adja az előjelet."""
    p_meta = np.array([0.9, 0.3, 0.8])
    primary = np.array([1, 1, -1])
    base = np.array([0.4, 0.4, 0.4])
    pos = meta_position_size(p_meta, primary, base)
    np.testing.assert_allclose(pos, [0.36, 0.12, -0.32])


def test_meta_model_deterministic():
    from quantimate.models.metalabel import train_meta_model
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (300, 5))
    y = (X[:, 0] + rng.normal(0, 0.5, 300) > 0).astype(int)
    tr, va = np.arange(200), np.arange(210, 280)
    m1 = train_meta_model(X, y, tr, va, seed=7)
    m2 = train_meta_model(X, y, tr, va, seed=7)
    np.testing.assert_array_equal(m1.predict_proba(X[va]), m2.predict_proba(X[va]))
