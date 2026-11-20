"""Makro- és bizonytalansági jellemzők (V7-2): EPU, VIX-terminstruktúra, TNX.

A jellemzőkészlet ár-alapú; a piaci bizonytalanság exogén mérőszámai hiányoznak.
Három ingyenes forrás: (a) az Economic Policy Uncertainty index (Baker–Bloom–
Davis 2016; FRED USEPUINDXD), (b) a VIX-terminstruktúra (^VIX9D/^VIX/^VIX3M/
^VIX6M — a contango/backwardation meredekség előretekintő stressz-jel), (c) a
10 éves hozam (^TNX, már a light-készletben).

**Szivárgás-fegyelem — ITT A LÉNYEG.** Kétféle csapda:

1. *Publikációs késés.* Egy makro-adat a t napon még nem ismert, ha csak később
   jelenik meg → `apply_release_lag` konzervatív, dokumentált lag-gel tolja
   (pozitív shift, csak múlt). A napi EPU (USEPUINDXD) +1 nap lagot kap; a
   VIX-terminstruktúra piaci ZÁRÓ adat, aznap ismert (ugyanabból a záró
   pillanatból, mint a célváltozó) → NEM igényel lagot, csak ffill-t.
2. *Revízió (vintage).* A FRED mai letöltése a REVIDEÁLT értéket adja, amit a t
   napon senki nem látott. A napi EPU gyakorlatilag nem revideálódik → materiális
   hatás ~0. DE ha a készlet revideált sorokra bővül (GDP, CPI, …), a becsületes
   megoldás vintage-adat (ALFRED point-in-time); addig ilyen sort NEM veszünk be.

FIGYELEM: az `apply_release_lag` SOROKAT tol, nem naptári napokat — napi sornál
ez 1 nap; havi sornál 1 hónap, ami >1 hónapos publikációs késésnél alul-lag-elne,
ezért havi revideált sort csak vintage-adattal szabad bevonni. Release-lag pytest fut.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from quantimate.data.data import HEADERS, RETRY_DELAYS

VIX_TERM_TICKERS = ("^VIX9D", "^VIX", "^VIX3M", "^VIX6M")
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def apply_release_lag(series: pd.Series, lag_days: int) -> pd.Series:
    """Konzervatív publikációs lag: a t napon csak a t-lag_days-ig ismert érték
    használható. Pozitív shift (múltra tolás) — SOHA nem néz jövőbe."""
    return series.shift(lag_days)


def fetch_fred_series(series_id: str, cache_dir=None,
                      lag_days: int = 1) -> pd.Series:
    """FRED napi/havi sor letöltése kulcs nélkül (CSV-végpont) + release-lag.

    A napi EPU (USEPUINDXD) piaci nappal igazított; a +1 nap lag konzervatív
    (az adott nap indexe a következő nap reggelén válik ismertté).
    """
    from pathlib import Path

    from quantimate.config import DATA_DIR
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"fred_{series_id}.parquet"

    import datetime as dt
    if cache_file.exists():
        mtime = dt.date.fromtimestamp(cache_file.stat().st_mtime)
        if mtime == dt.date.today():
            s = pd.read_parquet(cache_file).iloc[:, 0]
            return apply_release_lag(s, lag_days)

    last_exc = None
    for delay in (0,) + RETRY_DELAYS:
        if delay:
            import time
            time.sleep(delay)
        try:
            r = requests.get(FRED_CSV_URL.format(series=series_id),
                             headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            s = df.dropna().set_index("date")["value"].rename(series_id)
            s.to_frame().to_parquet(cache_file)
            return apply_release_lag(s, lag_days)
        except Exception as exc:
            last_exc = exc
    if cache_file.exists():
        import warnings
        warnings.warn(f"{series_id}: hálózati hiba — régi cache-ből ({last_exc}).")
        return apply_release_lag(pd.read_parquet(cache_file).iloc[:, 0], lag_days)
    raise ConnectionError(f"FRED {series_id} elérhetetlen és nincs cache ({last_exc})")


def vix_term_features(index: pd.DatetimeIndex, cache_dir=None) -> pd.DataFrame:
    """VIX-terminstruktúra jellemzők a chart API-ról (Volume nélküli light-út).
    A meredekség (slope) = VIX3M − VIX9D; a contango-arány = VIX3M / VIX.
    Rövidebb historikumú tickernél (pl. ^VIX9D) NaN-audit + ffill(limit=3).
    """
    from quantimate.data.data import _safe_name, fetch_ohlcv

    cols = {}
    for t in VIX_TERM_TICKERS:
        try:
            d = fetch_ohlcv(t, cache_dir=cache_dir)
            cols[_safe_name(t)] = d[f"{_safe_name(t)}_Close"].reindex(index).ffill(limit=3)
        except Exception:
            import warnings
            warnings.warn(f"{t}: nem letölthető — kihagyva a VIX-term készletből.")

    out = pd.DataFrame(index=index)
    if "VIX9D" in cols and "VIX3M" in cols:
        out["vixterm_slope"] = cols["VIX3M"] - cols["VIX9D"]
    if "VIX" in cols and "VIX3M" in cols:
        out["vixterm_contango"] = cols["VIX3M"] / cols["VIX"]
    if "VIX6M" in cols and "VIX" in cols:
        out["vixterm_slope_long"] = cols["VIX6M"] - cols["VIX"]
    return out


def build_macro_features(index: pd.DatetimeIndex, cache_dir=None,
                         with_epu: bool = True) -> pd.DataFrame:
    """A makro-jellemzők összeállítása a jellemző-index naptárára igazítva.

    Minden oszlop kauzális: a VIX-term piaci záró adat (aznap ismert, csak ffill),
    az EPU +1 nap release-lag-gel. A hiányzó/elérhetetlen forrásokat kihagyjuk
    (a fő lánc nem törik)."""
    frames = [vix_term_features(index, cache_dir)]
    if with_epu:
        try:
            epu = fetch_fred_series("USEPUINDXD", cache_dir, lag_days=1)
            epu_d = epu.reindex(index).ffill(limit=5)
            frames.append(pd.DataFrame({
                "epu_level": epu_d,
                "epu_chg5": epu_d.diff(5),
                "epu_dist_sma20": (epu_d - epu_d.rolling(20).mean())
                / epu_d.rolling(20).mean(),
            }))
        except Exception:
            import warnings
            warnings.warn("EPU nem elérhető — kihagyva a makro-készletből.")
    return pd.concat(frames, axis=1)
