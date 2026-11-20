"""Adatréteg, elfogadási kritériumok: adjclose-korrekció, faktor-őr, cache-azonosság."""
import numpy as np
import pandas as pd
import pytest

from quantimate.data import apply_adjustment, drop_null_ohlc_rows, parse_chart_result


def _synthetic_split_frame():
    """20 napos árfolyam 2:1 split-tel a 10. napon.

    A nyers Close a split napján feleződik (100 → 50); az AdjClose úgy van
    visszamenőleg korrigálva, hogy a valódi (befektetői) hozam folytonos:
    minden nap pontosan +1% adjusted hozam.
    """
    n = 20
    idx = pd.bdate_range("2024-01-02", periods=n)
    adj = 50.0 * 1.01 ** np.arange(n)          # folytonos, +1%/nap
    raw_close = adj.copy()
    raw_close[:10] *= 2.0                       # split előtt a nyers ár dupla
    f = adj / raw_close
    df = pd.DataFrame({
        "Open": raw_close * 0.99,
        "High": raw_close * 1.02,
        "Low": raw_close * 0.98,
        "Close": raw_close,
        "Volume": np.full(n, 1e6),
        "AdjClose": adj,
    }, index=idx)
    return df, f


def test_adjusted_log_returns_continuous_across_split():
    """(i) A korrigált sorozat log-hozama folytonos — nincs −50%-os műugrás."""
    df, _ = _synthetic_split_frame()
    adj = apply_adjustment(df)
    log_ret = np.log(adj["Close"] / adj["Close"].shift(1)).dropna()
    assert np.allclose(log_ret.values, np.log(1.01), atol=1e-12), \
        "A split-napon ugrás maradt a korrigált hozamban"
    # A nyers sorozatban viszont OTT az ugrás — a teszt tényleg a korrekciót méri.
    raw_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    assert raw_ret.min() < np.log(0.6)


def test_adjustment_scales_full_ohlc_not_volume():
    """A faktor az Open/High/Low-t is skálázza, a Volume-ot nem."""
    df, f = _synthetic_split_frame()
    adj = apply_adjustment(df)
    for col in ("Open", "High", "Low"):
        assert np.allclose(adj[col].values, (df[col] * f).values)
    assert np.array_equal(adj["Volume"].values, df["Volume"].values)
    assert "AdjClose" not in adj.columns


def test_adjustment_factor_bounds():
    """(ii) f_t ∈ (0, 1.05] — ezen kívüli faktor kivételt dob."""
    df, f = _synthetic_split_frame()
    assert ((f > 0) & (f <= 1.05)).all()
    bad = df.copy()
    bad.loc[bad.index[5], "AdjClose"] = bad.loc[bad.index[5], "Close"] * 1.5
    with pytest.raises(ValueError):
        apply_adjustment(bad)
    bad2 = df.copy()
    bad2.loc[bad2.index[5], "AdjClose"] = -1.0
    with pytest.raises(ValueError):
        apply_adjustment(bad2)


def test_cache_roundtrip_bit_identical(tmp_path):
    """(iii) Parquet-be írva és visszatöltve bitre azonos DataFrame."""
    df, _ = _synthetic_split_frame()
    adj = apply_adjustment(df)
    adj.columns = [f"NVDA_{c}" for c in adj.columns]
    p = tmp_path / "NVDA.parquet"
    adj.to_parquet(p)
    back = pd.read_parquet(p)
    # check_freq=False: a parquet a bdate_range freq-metaadatát nem tárolja;
    # a valós Yahoo-indexnek nincs freq-je, az ADAT bitazonossága a kritérium.
    pd.testing.assert_frame_equal(adj, back, check_exact=True, check_freq=False)


def test_null_rows_dropped_and_duplicates_keep_last(capsys):
    """Null-OHLC sorok kiesnek; duplikált timestampből az utolsó marad."""
    res = {
        "timestamp": [1704205800, 1704292200, 1704292200, 1704378600],
        "indicators": {
            "quote": [{
                "open":   [1.0, 2.0, 2.5, None],
                "high":   [1.1, 2.1, 2.6, 3.1],
                "low":    [0.9, 1.9, 2.4, 2.9],
                "close":  [1.0, 2.0, 2.5, 3.0],
                "volume": [10, 20, 25, 30],
            }],
            "adjclose": [{"adjclose": [1.0, 2.0, 2.5, 3.0]}],
        },
    }
    df = parse_chart_result(res, "TEST")
    assert len(df) == 3, "A duplikált timestampből egynek kell maradnia"
    assert df["Close"].iloc[1] == 2.5, "Duplikátumnál az utolsó érték marad"
    df = drop_null_ohlc_rows(df, "TEST")
    assert len(df) == 2, "A null-Open sornak ki kell esnie"


def test_timezone_normalisation_yields_trading_date():
    """UTC-ből New York-i napra normálva a helyes kereskedési dátumot kapjuk.

    2024-01-02 14:30 UTC = 09:30 New York → a dátum 2024-01-02 kell legyen
    (a naiv UTC-normalizálás is ezt adná; az esti — pl. 21:00 UTC = 16:00 NY —
    záró timestampnél viszont a naiv UTC és a NY-dátum már eltérhetne másnapra
    váltó piacoknál, ezért a NY-konverzió az auditált, helyes út).
    """
    res = {
        "timestamp": [1704205800],  # 2024-01-02 14:30:00 UTC
        "indicators": {"quote": [{"open": [1.0], "high": [1.0], "low": [1.0],
                                  "close": [1.0], "volume": [1]}],
                       "adjclose": [{"adjclose": [1.0]}]},
    }
    df = parse_chart_result(res, "TEST")
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert df.index.tz is None
