"""V7-2 tesztek: release-lag fegyelem (makro-oszlop csak release≤t
információt hordoz), VIX-term jellemzők számítása, hiányzó forrás nem tör."""
import numpy as np
import pandas as pd
import pytest

from quantimate.features.macro import apply_release_lag


def test_release_lag_is_causal():
    """A t-napi makro-érték a t-lag napi nyers érték (múltra tolva);
    a t napi és későbbi nyers értékek NEM szerepelnek a t-edik pozícióban."""
    idx = pd.bdate_range("2024-01-01", periods=20)
    raw = pd.Series(np.arange(20, dtype=float), index=idx)
    lagged = apply_release_lag(raw, lag_days=1)
    # a t-edik lag-érték = a (t-1)-edik nyers érték
    assert lagged.iloc[5] == raw.iloc[4]
    assert lagged.iloc[1] == raw.iloc[0]
    assert np.isnan(lagged.iloc[0])            # az első napra nincs korábbi érték


def test_release_lag_no_future_leak():
    """A jövő elrontása nem befolyásolja a lag-olt sor korábbi értékeit."""
    idx = pd.bdate_range("2024-01-01", periods=30)
    raw = pd.Series(np.arange(30, dtype=float), index=idx)
    base = apply_release_lag(raw, lag_days=2)
    raw2 = raw.copy()
    raw2.iloc[15:] += 1000
    fut = apply_release_lag(raw2, lag_days=2)
    # a 15. pozícióig (a lag miatt 17-ig) változatlan
    np.testing.assert_array_equal(base.iloc[:17].to_numpy(),
                                  fut.iloc[:17].to_numpy())


def test_release_lag_larger_lag():
    idx = pd.bdate_range("2024-01-01", periods=40)
    raw = pd.Series(np.arange(40, dtype=float), index=idx)
    lagged = apply_release_lag(raw, lag_days=21)   # havi adat ~1 hónap lag
    assert lagged.iloc[25] == raw.iloc[4]
    assert lagged.iloc[:21].isna().all()
