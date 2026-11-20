# -*- coding: utf-8 -*-
"""Pooled tanítás (V8-6) tesztjei — hálózat NÉLKÜL, szintetikus kerettel:
generikus átnevezés, oszlopkészlet-azonosság, dátum-maszk, tick_id, kauzalitás."""
import numpy as np
import pandas as pd
import pytest

from quantimate.features import pooled
from quantimate.features.pooled import (POOLED_UNIVERSE, build_pooled_training,
                               generic_feature_frame, generic_tabular,
                               nvda_generic_views, ticker_design)
from quantimate.data import pit
HORIZONS = (1, 5)


def _synthetic_frame(ticker: str, n: int = 320, seed: int = 7) -> pd.DataFrame:
    """OHLCV-keret a tickerre + SPY/QQQ/SMH kontextus + VIX/VXN/TNX (Close)."""
    rng = np.random.default_rng(seed + sum(map(ord, ticker)))
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame(index=idx)
    for t in (ticker, "SPY", "QQQ", "SMH"):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        df[f"{t}_Open"] = px * (1 + rng.normal(0, 0.003, n))
        df[f"{t}_High"] = px * (1 + np.abs(rng.normal(0, 0.006, n)))
        df[f"{t}_Low"] = px * (1 - np.abs(rng.normal(0, 0.006, n)))
        df[f"{t}_Close"] = px
        df[f"{t}_Volume"] = rng.integers(1_000_000, 5_000_000, n).astype(float)
    for t in ("VIX", "VXN", "TNX"):
        df[f"{t}_Close"] = 20 + np.cumsum(rng.normal(0, 0.3, n))
    return df


class TestGenericFeatures:
    def test_rename_and_no_ticker_prefix(self):
        g = generic_feature_frame(_synthetic_frame("AAA"), "AAA")
        assert "tgt_log_ret" in g.columns and "tgt_rsi14" in g.columns
        assert not any(c.startswith("AAA_") for c in g.columns)

    def test_identical_columns_across_tickers(self):
        """A pooled tér kulcsa: MINDEN ticker ugyanazt az oszlopkészletet adja."""
        ga = generic_feature_frame(_synthetic_frame("AAA"), "AAA")
        gb = generic_feature_frame(_synthetic_frame("BBBB"), "BBBB")
        assert list(ga.columns) == list(gb.columns)

    def test_values_match_prefixed_computation(self):
        """Az átnevezés nem változtat értéket: tgt_log_ret = log(C/C-1)."""
        df = _synthetic_frame("AAA")
        g = generic_feature_frame(df, "AAA")
        expected = np.log(df["AAA_Close"] / df["AAA_Close"].shift(1))
        pd.testing.assert_series_equal(g["tgt_log_ret"], expected,
                                       check_names=False)

    def test_lags_are_causal(self):
        g = generic_feature_frame(_synthetic_frame("AAA"), "AAA")
        tab = generic_tabular(g)
        pd.testing.assert_series_equal(tab["tgt_log_ret_lag2"],
                                       g["tgt_log_ret"].shift(2),
                                       check_names=False)


class TestTickerDesign:
    def test_no_nan_and_targets(self):
        tab, tg = ticker_design(_synthetic_frame("AAA"), "AAA", HORIZONS)
        assert not tab.isna().any().any()
        assert {"y_h1", "y_h5", "fwd_ret_h1"} <= set(tg.columns)
        # az utolsó max(H) sor címkéje jogosan NaN (jövő kell hozzá)
        assert tg["y_h5"].iloc[:-5].notna().all()
        assert tg["y_h5"].iloc[-5:].isna().all()


class TestPooledTraining:
    def test_mask_tickid_and_shapes(self, monkeypatch):
        frames = {"AAA": _synthetic_frame("AAA"), "BBBB": _synthetic_frame("BBBB")}
        monkeypatch.setattr(pooled, "load_ticker_frame",
                            lambda t, years=10, cache_dir=None: frames[t])
        all_dates = frames["AAA"].index
        train_dates = all_dates[100:200]
        out = build_pooled_training(train_dates, HORIZONS,
                                    tickers=("AAA", "BBBB"), verbose=False)
        assert out["X"].shape[0] == out["Y"].shape[0]
        assert out["feature_names"][-1] == "tick_id"
        tick_col = out["X"][:, -1]
        assert set(np.unique(tick_col)) == {0.0, 1.0}
        # a maszk: egy ticker sem adhat több sort a train-ablaknál
        assert all(v <= len(train_dates)
                   for v in out["rows_per_ticker"].values())
        assert not np.isnan(out["X"]).any() and not np.isnan(out["Y"]).any()

    def test_unavailable_ticker_honest_skip(self, monkeypatch):
        def _loader(t, years=10, cache_dir=None):
            if t == "DEAD":
                raise ConnectionError("nincs adat")
            return _synthetic_frame(t)
        monkeypatch.setattr(pooled, "load_ticker_frame", _loader)
        train_dates = _synthetic_frame("AAA").index[100:200]
        with pytest.warns(UserWarning, match="DEAD"):
            out = build_pooled_training(train_dates, HORIZONS,
                                        tickers=("AAA", "DEAD"), verbose=False)
        assert out["rows_per_ticker"]["DEAD"] == 0


class TestNvdaGenericView:
    def test_aligned_view(self):
        df = _synthetic_frame("NVDA")
        idx = df.index[120:]              # bemelegedés utáni ablak
        X = nvda_generic_views(df, idx)
        assert X.shape[0] == len(idx)
        assert (X[:, -1] == float(POOLED_UNIVERSE.index("NVDA"))).all()


def test_pooled_columns_pit_registered():
    """V8-10: a generikus oszlopcsalád és a tick_id regisztrált."""
    g = generic_feature_frame(_synthetic_frame("AAA"), "AAA")
    tab = generic_tabular(g)
    assert not pit.missing(list(tab.columns) + ["tick_id"])
