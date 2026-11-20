"""Jellemző-modul tesztek: naptár, earnings feature-flag, index-készlet, on/id."""
import numpy as np
import pandas as pd
import pytest

from quantimate.features import (calendar_features, earnings_features,
                                 compute_index_features, overnight_intraday,
                                 generate_earnings_candidates)


def test_calendar_features_values():
    idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01",
                            "2024-02-16", "2024-02-19"])
    cal = calendar_features(idx)
    # 2024-01-31 az utolsó januári nap az indexben → hónap-vége
    assert cal.loc["2024-01-31", "cal_month_end"] == 1.0
    assert cal.loc["2024-01-30", "cal_month_end"] == 0.0
    # 2024-02-16 péntek, a hónap 3. pénteke → opex
    assert cal.loc["2024-02-16", "cal_opex_friday"] == 1.0
    assert cal.loc["2024-02-19", "cal_opex_friday"] == 0.0
    # sin²+cos² = 1
    np.testing.assert_allclose(cal["cal_dow_sin"] ** 2 + cal["cal_dow_cos"] ** 2, 1.0)


def test_earnings_features_flag_and_distances():
    idx = pd.bdate_range("2024-01-01", periods=20)
    with pytest.warns(UserWarning):
        z = earnings_features(idx, None)
    assert (z == 0).all().all()

    dates = pd.DatetimeIndex([idx[10]])
    e = earnings_features(idx, dates, clip_days=30)
    assert e["earn_days_to_next"].iloc[8] == 2
    assert e["earn_days_to_next"].iloc[10] == 0
    assert e["earn_days_since_prev"].iloc[12] == 2
    # earnings-hét: ±2 kereskedési nap
    assert e["earn_week"].iloc[8] == 1.0
    assert e["earn_week"].iloc[12] == 1.0
    assert e["earn_week"].iloc[7] == 0.0
    assert e["earn_week"].iloc[13] == 0.0
    # Az első napokon a "since_prev" a clipet kapja (nincs korábbi earnings)
    assert e["earn_days_since_prev"].iloc[0] == 30


def test_generate_earnings_candidates_finds_quarterly_jumps():
    """Szintetikus soron: nagy mozgás febr/máj napokon → jelölt; márciusi
    ugyanakkora mozgás → kiszűrve (nem jelentési hónap)."""
    idx = pd.bdate_range("2023-01-02", periods=500)
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, len(idx))
    dates_big = [pd.Timestamp("2023-02-23"), pd.Timestamp("2023-05-25"),
                 pd.Timestamp("2023-03-15")]
    for d in dates_big:
        r[idx.get_loc(d)] = 0.15
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    cands = generate_earnings_candidates(close, threshold_sigma=4.0)
    assert pd.Timestamp("2023-02-23") in cands
    assert pd.Timestamp("2023-05-25") in cands
    assert pd.Timestamp("2023-03-15") not in cands


def test_index_features_no_volume_needed():
    idx = pd.bdate_range("2024-01-01", periods=40)
    d = pd.DataFrame({"VIX_Close": np.linspace(15, 25, 40)}, index=idx)
    f = compute_index_features(d, "VIX")
    assert list(f.columns) == ["VIX_level", "VIX_chg5", "VIX_dist_sma20"]
    assert f["VIX_chg5"].iloc[10] == pytest.approx(d["VIX_Close"].iloc[10]
                                                   - d["VIX_Close"].iloc[5])


def test_overnight_intraday_decomposition():
    idx = pd.bdate_range("2024-01-01", periods=5)
    d = pd.DataFrame({
        "NVDA_Open": [100, 102, 101, 105, 104],
        "NVDA_Close": [101, 103, 104, 103, 106],
    }, dtype=float, index=idx)
    f = overnight_intraday(d, "NVDA")
    # r_on + r_id = teljes napi log-hozam (close-to-close)
    total = np.log(d["NVDA_Close"] / d["NVDA_Close"].shift(1))
    np.testing.assert_allclose((f["NVDA_r_overnight"] + f["NVDA_r_intraday"]).iloc[1:],
                               total.iloc[1:], rtol=1e-12)
