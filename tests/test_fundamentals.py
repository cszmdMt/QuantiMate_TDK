"""V8-2 fundamentum-modul tesztek — hálózat NÉLKÜL: Form 4 XML-parzolás,
filing-lag (PIT!), short-interest publikációs lépcső, Reg SHO shift(1),
availability-mask, PIT-regisztráció."""
import numpy as np
import pandas as pd

from quantimate.data import pit
from quantimate.data.fundamentals import (FUNDAMENTAL_FEATURES, insider_features,
                                     parse_form4_xml, short_interest_features,
                                     shortvol_features)

FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>50.0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>400</value></transactionShares>
        <transactionPricePerShare><value>60.0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>99999</value></transactionShares>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def test_parse_form4_only_open_market():
    """P/S kódok élnek (A=+, D=−), az M (award) kizárva."""
    txs = parse_form4_xml(FORM4_XML)
    assert len(txs) == 2
    assert txs[0]["shares"] == 1000 and txs[0]["value_usd"] == 50000.0
    assert txs[1]["shares"] == -400 and txs[1]["value_usd"] == -24000.0
    assert parse_form4_xml("nem xml") == []


def test_insider_filing_lag_is_pit():
    """A filing-napi tranzakció a KÖVETKEZŐ kereskedési napon jelenik meg
    (+1 lag: zárás utáni acceptance) — a filing napján még NaN/0."""
    idx = pd.bdate_range("2026-01-05", periods=30)
    tx = pd.DataFrame({"filing_date": [pd.Timestamp("2026-01-07")] * 2,
                       "code": ["P", "S"], "shares": [1000.0, -400.0],
                       "value_usd": [50000.0, -24000.0]})
    out = insider_features(idx, tx)
    # az első filing előtt: availability-NaN
    assert out.loc["2026-01-05"].isna().all()
    # a filing napján még nem látszik (lag) — a maszk miatt itt is NaN
    assert out.loc["2026-01-07"].isna().all()
    # másnaptól él a rolling összeg
    assert out.loc["2026-01-08", "insider_netbuy_sh_21"] == 600.0
    assert out.loc["2026-01-08", "insider_netbuy_usd_21"] == 26000.0
    # 21 nappal később kicsúszik az ablakból
    assert out.loc[idx[idx.get_loc("2026-01-08") + 21],
                   "insider_netbuy_sh_21"] == 0.0
    assert out.loc["2026-02-13", "insider_netbuy_sh_63"] == 600.0


def test_short_interest_publication_step():
    """A settlement + 9 munkanap ELŐTT a rekord nem látszik; utána lépcsőben él."""
    idx = pd.bdate_range("2026-01-02", periods=40)
    si = pd.DataFrame({
        "settlementDate": pd.to_datetime(["2026-01-15", "2026-01-30"]),
        "currentShortPositionQuantity": [1e6, 2e6],
        "daysToCoverQuantity": [1.5, 2.5],
        "changePercent": [3.0, 100.0],
    })
    out = short_interest_features(idx, si, pub_lag_bdays=9)
    pub1 = pd.Timestamp("2026-01-15") + pd.offsets.BDay(9)   # 2026-01-28
    assert out.loc[:pub1 - pd.offsets.BDay(1), "short_interest_qty"].isna().all()
    assert out.loc[pub1, "short_interest_qty"] == 1e6
    pub2 = pd.Timestamp("2026-01-30") + pd.offsets.BDay(9)
    assert out.loc[pub2 - pd.offsets.BDay(1), "short_interest_qty"] == 1e6
    assert out.loc[pub2, "short_interest_qty"] == 2e6
    assert out.loc[pub2, "short_interest_dtc"] == 2.5


def test_shortvol_shift_one_day():
    """A t-napi jellemző a t−1 nap aránya (zárás utáni publikáció)."""
    idx = pd.bdate_range("2026-03-02", periods=10)
    sv = pd.DataFrame({"date": idx[:9], "short_vol": np.arange(1, 10) * 100.0,
                       "total_vol": np.full(9, 1000.0)})
    out = shortvol_features(idx, sv)
    assert np.isnan(out.iloc[0]["shortvol_ratio"])
    assert out.iloc[1]["shortvol_ratio"] == 0.1     # az első nap 100/1000
    assert out.iloc[9]["shortvol_ratio"] == 0.9
    ma = out.iloc[9]["shortvol_ratio_ma5"]
    assert ma == np.mean([0.5, 0.6, 0.7, 0.8, 0.9])


def test_empty_sources_give_nan_columns():
    idx = pd.bdate_range("2026-01-02", periods=5)
    empty_tx = pd.DataFrame(columns=["filing_date", "code", "shares", "value_usd"])
    assert insider_features(idx, empty_tx).isna().all().all()
    empty_sv = pd.DataFrame(columns=["date", "short_vol", "total_vol"])
    assert shortvol_features(idx, empty_sv).isna().all().all()


def test_pit_registered():
    pit.assert_all_registered(list(FUNDAMENTAL_FEATURES))
    assert pit.lookup("insider_netbuy_usd_21").lag_days == 1
    assert pit.lookup("short_interest_dtc").lag_days == 9
    assert pit.lookup("shortvol_ratio").lag_days == 1
