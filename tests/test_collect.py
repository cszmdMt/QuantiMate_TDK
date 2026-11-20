# -*- coding: utf-8 -*-
"""Collect-forward gyűjtések (V8.5 / V8-12) tesztjei — hálózat NÉLKÜL:
merge-logika, snapshot-séma, összegző-számítás, hash-lánc."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quantimate.data.collect import (SNAPSHOT_COLUMNS, _bs_call_delta, merge_bars,
                                options_summary, validate_snapshot)
from quantimate.backtest.live_ledger import append_prediction, verify_chain
from quantimate.data import pit
def _bars(start: str, n: int, base: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="h", tz="America/New_York")
    px = base + np.arange(n, dtype=float)
    return pd.DataFrame({"Open": px, "High": px + 1, "Low": px - 1,
                         "Close": px + 0.5, "Volume": 1e6}, index=idx)


class TestMergeBars:
    def test_sliding_window_keeps_old_bars(self):
        """A friss ablakból kicsúszott régi barok NEM vesznek el (append-only)."""
        archive = _bars("2024-01-02 09:30", 20)
        fresh = _bars("2024-01-02 19:30", 20, base=200.0)  # 10 bar átfedés
        merged = merge_bars(archive, fresh)
        assert len(merged) == 30
        assert merged.index.min() == archive.index.min()
        assert merged.index.max() == fresh.index.max()

    def test_overlap_keeps_last(self):
        """Átfedő időbélyegen a FRISS érték él (lezárt bar frissülhet)."""
        archive = _bars("2024-01-02 09:30", 5)
        fresh = _bars("2024-01-02 09:30", 5, base=500.0)
        merged = merge_bars(archive, fresh)
        assert len(merged) == 5
        assert (merged["Close"] >= 500).all()

    def test_empty_archive(self):
        fresh = _bars("2024-01-02 09:30", 7)
        merged = merge_bars(None, fresh)
        pd.testing.assert_frame_equal(merged, fresh)

    def test_sorted_output(self):
        archive = _bars("2024-01-03 09:30", 3)
        fresh = _bars("2024-01-02 09:30", 3)
        assert merge_bars(archive, fresh).index.is_monotonic_increasing


def _chain(asof: dt.date) -> pd.DataFrame:
    """Szintetikus, de reális szerkezetű lánc TESZT-bemenetként (nem adat!)."""
    expiry = pd.Timestamp(asof + dt.timedelta(days=14))
    rows = []
    for strike in (80, 90, 100, 110, 120):
        # put-oldali IV-többlet (skew): OTM put drágább
        rows.append({"expiry": expiry, "type": "call", "strike": float(strike),
                     "last": 1.0, "bid": 0.9, "ask": 1.1, "volume": 10.0,
                     "open_interest": 100.0,
                     "implied_vol": 0.40 + 0.001 * (strike - 100)})
        rows.append({"expiry": expiry, "type": "put", "strike": float(strike),
                     "last": 1.0, "bid": 0.9, "ask": 1.1, "volume": 10.0,
                     "open_interest": 150.0,
                     "implied_vol": 0.40 + 0.004 * (100 - strike)})
    return pd.DataFrame(rows, columns=list(SNAPSHOT_COLUMNS))


class TestSnapshotSchema:
    def test_valid_chain_passes(self):
        validate_snapshot(_chain(dt.date(2026, 7, 15)))

    def test_missing_column_fails(self):
        broken = _chain(dt.date(2026, 7, 15)).drop(columns=["implied_vol"])
        with pytest.raises(AssertionError, match="implied_vol"):
            validate_snapshot(broken)

    def test_bad_type_fails(self):
        bad = _chain(dt.date(2026, 7, 15))
        bad.loc[0, "type"] = "swap"
        with pytest.raises(AssertionError):
            validate_snapshot(bad)


class TestOptionsSummary:
    def test_summary_values(self):
        asof = dt.date(2026, 7, 15)
        s = options_summary(_chain(asof), spot=100.0, asof=asof)
        assert s["putcall_oi"] == pytest.approx(1.5)          # 150/100 OI
        assert s["atm_iv"] == pytest.approx(0.40, abs=0.02)   # ATM ~ 0.40
        assert s["skew_25d"] > 0                              # put-IV többlet
        assert s["n_expiries"] == 1 and s["n_contracts"] == 10

    def test_empty_chain_gives_nan(self):
        empty = pd.DataFrame(columns=list(SNAPSHOT_COLUMNS))
        s = options_summary(empty, spot=100.0)
        assert np.isnan(s["atm_iv"]) and np.isnan(s["putcall_oi"])
        assert s["n_contracts"] == 0

    def test_atm_call_delta_near_half(self):
        d = _bs_call_delta(100.0, np.array([100.0]), np.array([0.4]),
                           np.array([14 / 365]))
        assert d[0] == pytest.approx(0.5, abs=0.03)


class TestCollectionsChain:
    def test_append_and_verify(self, tmp_path):
        ledger = tmp_path / "collections.jsonl"
        for i in range(3):
            append_prediction(ledger, {"kind": "options_snapshot",
                                       "asof": f"2026-07-1{i}",
                                       "sha256": "ab" * 32})
        state = verify_chain(ledger)
        assert state["valid"] and state["n"] == 3

    def test_tamper_detected(self, tmp_path):
        ledger = tmp_path / "collections.jsonl"
        append_prediction(ledger, {"kind": "options_snapshot", "asof": "a"})
        append_prediction(ledger, {"kind": "options_snapshot", "asof": "b"})
        lines = ledger.read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace('"asof": "a"', '"asof": "x"')
        assert '"asof": "x"' in lines[0], "a manipuláció nem történt meg"
        ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert not verify_chain(ledger)["valid"]


def test_snapshot_columns_not_pit_registered():
    """V8-12 tiltás: a snapshot-oszlopok NEM PIT-regisztráltak — ha valaha a
    pipeline-ba kerülnének, az assert_all_registered buknia kell."""
    assert pit.missing(["implied_vol", "open_interest", "skew_25d"])
