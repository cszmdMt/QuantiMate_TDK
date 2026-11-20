"""V8-3 geo-modul tesztek — hálózat NÉLKÜL: GPR elérhetőségi lag (naptári
shift a forrás-soron), EPU-trade publikációs lépcső, PIT-regisztráció."""
import numpy as np
import pandas as pd

from quantimate.data import pit
from quantimate.features.geo import GEO_FEATURES, GPR_LAG_CDAYS, gpr_features


def _gpr_stub(start="2026-01-01", n=60):
    idx = pd.date_range(start, periods=n, freq="D")   # teljes naptár (forrás)
    vals = np.arange(n, dtype=float)
    return pd.DataFrame({"GPRD": vals, "GPRD_ACT": vals * 2,
                         "GPRD_THREAT": vals * 3}, index=idx)


def test_gpr_lag_is_calendar_shift():
    """A t-napi gpr_level a t−2 NAPTÁRI nap indexe — hétfőn a szombati érték."""
    gpr = _gpr_stub()
    tidx = pd.bdate_range("2026-01-12", periods=10)   # hétfő
    out = gpr_features(tidx, gpr, lag_cdays=GPR_LAG_CDAYS)
    # 2026-01-12 hétfő → t−2 = 01-10 szombat, GPRD ott = 9.0 (0-tól indexelve)
    assert out.loc["2026-01-12", "gpr_level"] == 9.0
    assert out.loc["2026-01-13", "gpr_level"] == 10.0
    assert out.loc["2026-01-12", "gpr_act"] == 18.0
    assert out.loc["2026-01-12", "gpr_threat"] == 27.0


def test_gpr_never_uses_future():
    """Növekvő forrás-soron a jellemző mindig < a t-napi nyers érték
    (a lag pozitív) — kimerítő ellenőrzés minden kereskedési napra."""
    gpr = _gpr_stub()
    tidx = pd.bdate_range("2026-01-08", periods=30)
    out = gpr_features(tidx, gpr)
    raw_on_t = gpr["GPRD"].reindex(tidx)
    valid = out["gpr_level"].notna() & raw_on_t.notna()
    assert (out.loc[valid, "gpr_level"] < raw_on_t[valid]).all()


def test_gpr_stale_source_stops_after_limit():
    """Ha a forrás megáll (nem frissül), max 5 nap ffill után NaN — nem
    hazudunk friss értéket."""
    gpr = _gpr_stub(n=20)                              # utolsó nap: 01-20
    tidx = pd.bdate_range("2026-01-19", periods=15)    # jóval a forrás vége utánig
    out = gpr_features(tidx, gpr)
    assert out["gpr_level"].isna().any()
    assert np.isfinite(out["gpr_level"].iloc[0])


def test_eputrade_publication_step():
    """A hó M értéke csak hóvége+7 naptól él — a hónap elején még az előző."""
    from unittest.mock import patch

    import quantimate.features.geo as geo_mod

    monthly = pd.Series([100.0, 110.0, 120.0],
                        index=pd.to_datetime(["2026-01-01", "2026-02-01",
                                              "2026-03-01"]))
    tidx = pd.bdate_range("2026-02-02", "2026-04-30")
    with patch("quantimate.macro.fetch_fred_series", return_value=monthly):
        out = geo_mod.eputrade_features(tidx)
    # a januári érték február 7-től él (jan 31 + 7 nap)
    assert out.loc["2026-02-06", "eputrade_level"] != 100.0 or \
        pd.isna(out.loc["2026-02-06", "eputrade_level"])
    assert out.loc["2026-02-09", "eputrade_level"] == 100.0   # feb 7 utáni bday
    # a februári érték március 7-től (feb 28 + 7): március 6-án még a januári
    assert out.loc["2026-03-06", "eputrade_level"] == 100.0
    assert out.loc["2026-03-09", "eputrade_level"] == 110.0
    # az áprilisban a márciusi (ápr 7-től)
    assert out.loc["2026-04-08", "eputrade_level"] == 120.0


def test_pit_registered():
    pit.assert_all_registered(list(GEO_FEATURES))
    assert pit.lookup("gpr_level").lag_days == 2
    assert pit.lookup("eputrade_level").lag_days == 25
