"""V8-4 earnings-hitelesítés tesztek — hálózat NÉLKÜL (szintetikus adatokon):
napszak-osztályozás, effective_date-szabály, keresztellenőrzés-tolerancia,
betöltő. Az éles letöltést a scripts/verify_earnings.py végzi (cache-elve)."""
import pandas as pd
import pytest

from quantimate.config import DEFAULT_CONFIG
from quantimate.data.earnings import (classify_timing, cross_check, effective_date,
                                 load_verified_earnings)


def test_default_earnings_source_is_v6_equivalent():
    """Őr: a default a generált jelölt-naptár — a verified csak flaggel."""
    assert DEFAULT_CONFIG.earnings_source == "generated"


def test_classify_timing():
    assert classify_timing(pd.Timestamp("2026-05-20 16:21:19")) == "post_close"
    assert classify_timing(pd.Timestamp("2026-05-20 08:05:00")) == "pre_open"
    assert classify_timing(pd.Timestamp("2026-05-20 11:00:00")) == "intraday"
    assert classify_timing(pd.Timestamp("2026-05-20 16:00:00")) == "post_close"


def test_effective_date_rule():
    # post_close szerdán → csütörtök; pénteken → hétfő; pre_open → aznap
    assert effective_date(pd.Timestamp("2026-05-20"), "post_close") == \
        pd.Timestamp("2026-05-21")
    assert effective_date(pd.Timestamp("2026-05-22"), "post_close") == \
        pd.Timestamp("2026-05-25")
    assert effective_date(pd.Timestamp("2026-05-20"), "pre_open") == \
        pd.Timestamp("2026-05-20")


def _edgar_stub():
    return pd.DataFrame({
        "announce_date": pd.to_datetime(["2025-11-19", "2026-02-25", "2026-05-20"]),
        "acceptance_et": pd.to_datetime(["2025-11-19 16:20:44",
                                          "2026-02-25 16:31:25",
                                          "2026-05-20 16:21:19"]),
        "timing": ["post_close"] * 3,
        "effective_date": pd.to_datetime(["2025-11-20", "2026-02-26", "2026-05-21"]),
        "accession": ["a1", "a2", "a3"],
    })


def test_cross_check_matching_and_tolerance():
    yahoo = pd.DataFrame({
        "yahoo_date": pd.to_datetime(["2025-11-19",    # pontos egyezés
                                      "2026-02-26"]),  # +1 nap — tűrésen belül
        "eps_estimate": [0.75, 0.84], "eps_actual": [0.81, 0.89],
    })
    tbl = cross_check(_edgar_stub(), yahoo, start="2015-01-01", tolerance_days=1)
    assert tbl["match"].tolist() == [True, True, False]
    assert tbl["yahoo_delta_days"].tolist()[:2] == [0, 1]
    # a párosított sor hozza az EPS-t, a nem párosított nem
    assert tbl.loc[0, "eps_actual"] == 0.81
    assert pd.isna(tbl.loc[2, "eps_actual"])
    # a 2026-05-20 a Yahoo-lefedettség (2026-02-26) UTÁN van → edgar_only
    assert tbl["verified_by"].tolist() == ["both", "both", "edgar_only"]


def test_cross_check_preliminary_flag():
    """Párosítatlan filing az átfedésen belül, ±45 napon belüli párosított
    filinggel → előzetes/figyelmeztető 8-K (pl. 2019-01-28 guidance-vágás)."""
    edgar = pd.DataFrame({
        "announce_date": pd.to_datetime(["2019-01-28", "2019-02-14"]),
        "acceptance_et": pd.to_datetime(["2019-01-28 08:00", "2019-02-14 16:20"]),
        "timing": ["pre_open", "post_close"],
        "effective_date": pd.to_datetime(["2019-01-28", "2019-02-15"]),
        "accession": ["p1", "s1"],
    })
    yahoo = pd.DataFrame({"yahoo_date": pd.to_datetime(["2019-02-14",
                                                        "2026-01-01"]),
                          "eps_estimate": [0.5, 1.0], "eps_actual": [0.5, 1.0]})
    tbl = cross_check(edgar, yahoo, tolerance_days=1)
    assert tbl["verified_by"].tolist() == ["preliminary", "both"]
    assert tbl["is_preliminary"].tolist() == [True, False]


def test_cross_check_start_window():
    ed = _edgar_stub()
    tbl = cross_check(ed, pd.DataFrame({"yahoo_date": pd.to_datetime([]),
                                        "eps_estimate": [], "eps_actual": []}),
                      start="2026-01-01")
    assert len(tbl) == 2 and not tbl["match"].any()


def test_load_verified_earnings(tmp_path):
    p = tmp_path / "verified.csv"
    _edgar_stub().assign(match=True,
                         is_preliminary=[False, True, False]).to_csv(p, index=False)
    ann = load_verified_earnings(p, kind="announce")
    eff = load_verified_earnings(p, kind="effective")
    # az előzetes (2026-02-25) alapból kimarad — nem ütemezett esemény
    assert list(ann) == list(pd.to_datetime(["2025-11-19", "2026-05-20"]))
    assert eff[0] == pd.Timestamp("2025-11-20")
    all_ann = load_verified_earnings(p, include_preliminary=True)
    assert len(all_ann) == 3
    with pytest.raises(FileNotFoundError):
        load_verified_earnings(tmp_path / "nincs.csv")
