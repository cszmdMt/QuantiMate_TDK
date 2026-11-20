"""V8-1 intraday-aggregátum tesztek — hálózat NÉLKÜL, szintetikus barokon:
RTH-szűrés (tz), félnap, jellemző-képletek, prefix-invariancia,
konzisztencia-őr, availability-mask, PIT-regisztráció."""
import numpy as np
import pandas as pd
import pytest

from quantimate.data import pit
from quantimate.data.intraday import (INTRADAY_FEATURES, build_intraday_daily,
                                 consistency_report, filter_rth)


def make_bars(day: str, closes, volume=None, start="09:30", freq="60min",
              tz="America/New_York"):
    """Szintetikus 60m barok egy napra: Open az előző záró, H/L a kettő köré."""
    n = len(closes)
    idx = pd.date_range(f"{day} {start}", periods=n, freq=freq, tz=tz)
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0] * 0.999], closes[:-1]])
    return pd.DataFrame({
        "Open": opens,
        "High": np.maximum(opens, closes) * 1.001,
        "Low": np.minimum(opens, closes) * 0.999,
        "Close": closes,
        "Volume": (np.full(n, 1e6) if volume is None
                   else np.asarray(volume, dtype=float)),
    }, index=idx)


def test_rth_filter_excludes_pre_post_market():
    day = make_bars("2026-06-01", [100, 101, 102, 103, 104, 105, 106])
    pre = make_bars("2026-06-01", [99], start="08:30")
    post = make_bars("2026-06-01", [107], start="16:00")
    allb = pd.concat([pre, day, post]).sort_index()
    rth = filter_rth(allb)
    assert len(rth) == 7
    assert rth.index.min().time().isoformat() == "09:30:00"
    assert rth.index.max().time().isoformat() == "15:30:00"


def test_daily_feature_formulas():
    closes = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0]
    vol = [3e6, 1e6, 1e6, 1e6, 1e6, 1e6, 2e6]
    bars = make_bars("2026-06-01", closes, volume=vol)
    out = build_intraday_daily(bars)
    assert list(out.columns) == list(INTRADAY_FEATURES)
    assert len(out) == 1
    row = out.iloc[0]

    o = 100.0 * 0.999
    r = np.diff(np.log(np.concatenate([[o], closes])))
    assert row["rv_1h"] == pytest.approx(np.sum(r ** 2))
    bpv = (np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    assert row["bpv"] == pytest.approx(bpv)
    assert row["jump"] == pytest.approx(max(row["rv_1h"] - bpv, 0.0))
    assert row["jump"] >= 0
    assert row["r_first_hour"] == pytest.approx(r[0])
    assert row["r_last_hour"] == pytest.approx(r[-1])
    assert row["volume_share_last_hour"] == pytest.approx(2e6 / 10e6)
    assert 0.0 <= row["intraday_trend_r2"] <= 1.0
    assert row["n_bars"] == 7.0
    # VWAP-gap előjel-értelmes: záró a VWAP felett → pozitív
    typical = ((np.maximum(np.r_[o, closes[:-1]], closes) * 1.001
                + np.minimum(np.r_[o, closes[:-1]], closes) * 0.999
                + closes) / 3)
    vwap = np.sum(typical * vol) / np.sum(vol)
    assert row["vwap_gap"] == pytest.approx(closes[-1] / vwap - 1.0)


def test_half_day_and_single_bar():
    half = make_bars("2026-11-27", [100, 100.5, 101, 101.5])   # félnap: 4 bar
    one = make_bars("2026-06-02", [50.0])
    out = build_intraday_daily(pd.concat([half, one]))
    assert out.loc["2026-11-27", "n_bars"] == 4.0
    assert np.isfinite(out.loc["2026-11-27", "rv_1h"])
    # egy-baros nap: nincs bar-pár → bpv/jump/trend NaN, de rv és hozamok élnek
    assert out.loc["2026-06-02", "n_bars"] == 1.0
    assert np.isnan(out.loc["2026-06-02", "bpv"])
    assert np.isnan(out.loc["2026-06-02", "jump"])
    assert np.isfinite(out.loc["2026-06-02", "rv_1h"])


def test_prefix_invariance_closed_day():
    """Egy lezárt nap jellemzője nem változik későbbi barok érkezésével."""
    d1 = make_bars("2026-06-01", [100, 101, 102, 103, 104, 105, 106])
    d2 = make_bars("2026-06-02", [106, 107, 106.5, 108, 107.5, 109, 110])
    full = build_intraday_daily(pd.concat([d1, d2]))
    prefix = build_intraday_daily(d1)
    a = full.loc["2026-06-01"].to_numpy(dtype=float)
    b = prefix.loc["2026-06-01"].to_numpy(dtype=float)
    assert ((a == b) | (np.isnan(a) & np.isnan(b))).all()


def test_availability_mask_reindex():
    """A 60m lefedettség előtti napok NaN-ok maradnak (nincs feltöltés)."""
    bars = make_bars("2026-06-01", [100, 101, 102, 103, 104, 105, 106])
    daily = build_intraday_daily(bars)
    idx = pd.DatetimeIndex(["2024-01-02", "2026-06-01", "2026-06-02"])
    out = daily.reindex(idx)
    assert out.loc["2024-01-02"].isna().all()
    assert np.isfinite(out.loc["2026-06-01", "rv_1h"])
    assert out.loc["2026-06-02"].isna().all()


def test_consistency_report():
    """Napi zárás ≈ utolsó RTH-bar záró; a tol feletti eltérés bukik."""
    bars = pd.concat([
        make_bars("2026-06-01", [100, 101, 102, 103, 104, 105, 106]),
        make_bars("2026-06-02", [106, 107, 108, 109, 110, 111, 112]),
    ])
    daily_close = pd.Series([106.0, 120.0],
                            index=pd.DatetimeIndex(["2026-06-01", "2026-06-02"]))
    rep = consistency_report(daily_close, bars, tol=0.01)
    assert bool(rep.loc["2026-06-01", "ok"]) is True
    assert bool(rep.loc["2026-06-02", "ok"]) is False   # 112 vs 120


def test_pit_registered():
    """V8-10: minden intraday-oszlop regisztrált a PIT-registryben."""
    pit.assert_all_registered(list(INTRADAY_FEATURES))
    meta = pit.lookup("rv_1h")
    assert meta.lag_days == 0 and "60m" in meta.source
