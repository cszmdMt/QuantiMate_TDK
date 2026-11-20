"""V7-3 tesztek: d=1 ≡ első differencia, súlyképlet kézi ellenőrzés,
train-only d*, ADF-stacionaritás, kauzalitás."""
import numpy as np
import pandas as pd
import pytest

from quantimate.features.fracdiff import ffd_weights, find_min_d, frac_diff_ffd


def test_weights_hand_computed():
    """w_0=1, w_1=-d, w_2=d(d-1)/2 (a képletből, legújabb→régebbi)."""
    d = 0.5
    w = ffd_weights(d, thresh=1e-10, max_width=4)
    # w visszafelé rendezett: [..., w_2, w_1, w_0]
    assert w[-1] == pytest.approx(1.0)
    assert w[-2] == pytest.approx(-d)
    assert w[-3] == pytest.approx(-(-d) * (d - 1) / 2)


def test_d_one_equals_first_difference():
    """d=1 → az FFD az első differenciát adja (w = [-1, 1])."""
    s = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 1, 100)),
                  index=pd.bdate_range("2024-01-01", periods=100))
    ffd = frac_diff_ffd(s, 1.0, thresh=1e-10)
    diff = s.diff()
    np.testing.assert_allclose(ffd.dropna().to_numpy(),
                               diff.dropna().to_numpy(), atol=1e-9)


def test_ffd_is_causal():
    """A t-edik FFD-érték csak a t-ig tartó múltból számol → a jövő elrontása
    nem befolyásolja."""
    rng = np.random.default_rng(1)
    s = pd.Series(np.cumsum(rng.normal(0, 1, 200)),
                  index=pd.bdate_range("2024-01-01", periods=200))
    base = frac_diff_ffd(s, 0.4)
    s2 = s.copy()
    s2.iloc[150:] += 100
    fut = frac_diff_ffd(s2, 0.4)
    np.testing.assert_array_equal(base.iloc[:150].to_numpy(),
                                  fut.iloc[:150].to_numpy())


def test_find_min_d_train_only():
    """A d* és a stacionaritás kizárólag a train-szakaszból; a train-utáni sor
    elrontása nem változtatja a választott d-t."""
    rng = np.random.default_rng(2)
    n = 800
    idx = pd.bdate_range("2020-01-01", periods=n)
    s = pd.Series(np.cumsum(rng.normal(0.01, 1, n)), index=idx, name="logprice")
    train_end = idx[500]

    r1 = find_min_d(s, train_end)
    s2 = s.copy()
    s2.iloc[520:] *= 10
    r2 = find_min_d(s2, train_end)
    assert r1["d"] == r2["d"]


def test_ffd_makes_random_walk_stationary():
    """Bolyongás (nem stacionárius) FFD után ADF-stacionárius lesz d<1-nél,
    és megtart valamennyi memóriát (korreláció az eredetivel > 0)."""
    rng = np.random.default_rng(3)
    n = 1500
    idx = pd.bdate_range("2018-01-01", periods=n)
    s = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=idx, name="rw")
    res = find_min_d(s, idx[-1])
    assert res["d"] <= 1.0
    assert res["adf_pvalue"] < 0.05
    assert res["corr_with_original"] > 0.0
