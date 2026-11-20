"""Geopolitikai modul (V8-3 CORE) — a macro.py lag-fegyelmének mintájára.

Két forrás:
1. **GPR napi index** (Caldara–Iacoviello, GPRD): a hivatalos oldal
   data_gpr_daily_recent **.dta** fájlja (a .xls xlrd-t igényelne — a CORE-ba
   új kötelező függőség nem vehető fel, a pandas read_stata beépített).
   A napi index visszamenőleg frissülhet és pár nap késéssel publikálódik →
   konzervatív 2 NAPTÁRI napos elérhetőségi lag (shift a forrás saját napi
   naptárán), az as-of letöltési dátum sidecar-metaadatban naplózva.
2. **EPU-trade kategória** (FRED EPUTRADE, havi): a hó M értéke a következő
   hónap elején publikálódik → elérhetőség = hóvége + 7 naptári nap, lépcsős
   as-of illesztés (SOHA nem sima ffill a megfigyelési dátumtól!).

EXT-tételek (Federal Register BIS-események, GDELT) a CORE-scope-on kívül —
dokumentált skip. Kézzel kurált eseménylista TILOS (hindsight-bias).
"""
from __future__ import annotations

import datetime as dt
import io
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quantimate.config import DATA_DIR
from quantimate.data.data import HEADERS, RETRY_DELAYS
from quantimate.data import pit

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.dta"
GPR_LAG_CDAYS = 2      # konzervatív elérhetőségi lag NAPTÁRI napban
EPU_PUB_DELAY_CDAYS = 7  # hóvége utáni publikációs puffer

GPR_COLS = ("gpr_level", "gpr_chg5", "gpr_act", "gpr_threat")
EPUTRADE_COLS = ("eputrade_level", "eputrade_chg12")
GEO_FEATURES = GPR_COLS + EPUTRADE_COLS

pit.register(r"gpr_(level|chg5|act|threat)",
             source="GPR napi index (matteoiacoviello.com, .dta)", lag_days=2,
             asof_rule="shift(+2 naptári nap) a forrás saját napi naptárán, "
                       "majd illesztés a kereskedési naptárra (ffill limit 5)",
             notes="a napi index revideálódhat — as-of letöltési dátum a "
                   "sidecar-metaadatban naplózva")
pit.register(r"eputrade_(level|chg12)",
             source="FRED EPUTRADE (havi EPU-trade kategória)", lag_days=25,
             asof_rule="a hó M értéke hóvége+7 naptári naptól él (lépcső, "
                       "publikációs lag) — kb. 25 kereskedési nap",
             notes="a havi EPU materiálisan nem revideálódik (macro.py-elv)")


def fetch_gpr_daily(cache_dir: Path | None = None) -> pd.DataFrame:
    """GPR napi tábla (date-indexelt GPRD/ACT/THREAT) cache-sel és as-of
    naplóval (data/gpr_daily.parquet + .meta.json)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR
    cache_file = cache_dir / "gpr_daily.parquet"
    meta_file = cache_dir / "gpr_daily.meta.json"

    if cache_file.exists():
        mtime = dt.date.fromtimestamp(cache_file.stat().st_mtime)
        if mtime == dt.date.today():
            return pd.read_parquet(cache_file)

    last_exc = None
    for delay in (0,) + RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            r = requests.get(GPR_URL, headers=HEADERS, timeout=60)
            r.raise_for_status()
            df = pd.read_stata(io.BytesIO(r.content))
            break
        except Exception as exc:
            last_exc = exc
    else:
        if cache_file.exists():
            warnings.warn(f"GPR: hálózati hiba — régebbi cache-ből ({last_exc}).")
            return pd.read_parquet(cache_file)
        raise ConnectionError(f"GPR elérhetetlen és nincs cache ({last_exc})")

    out = (df[["date", "GPRD", "GPRD_ACT", "GPRD_THREAT"]]
           .assign(date=pd.to_datetime(df["date"]))
           .set_index("date").astype(float).sort_index())
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_file)
    meta_file.write_text(json.dumps({
        "asof_download": dt.date.today().isoformat(),
        "last_obs": str(out.index.max().date()), "n_obs": len(out),
        "url": GPR_URL}), encoding="utf-8")
    return out


def gpr_features(daily_index: pd.DatetimeIndex, gpr: pd.DataFrame,
                 lag_cdays: int = GPR_LAG_CDAYS) -> pd.DataFrame:
    """GPR-jellemzők a kereskedési naptárra, elérhetőségi laggal.

    A shiftet a forrás SAJÁT (teljes naptári) során végezzük — így a t napi
    érték a t−lag napi index, kereskedési-naptári sorvesztés nélkül."""
    lagged = gpr.shift(lag_cdays)              # naptári napos forrás-sor
    on_idx = lagged.reindex(daily_index).ffill(limit=5)
    out = pd.DataFrame(index=daily_index)
    out["gpr_level"] = on_idx["GPRD"]
    out["gpr_chg5"] = on_idx["GPRD"].diff(5)
    out["gpr_act"] = on_idx["GPRD_ACT"]
    out["gpr_threat"] = on_idx["GPRD_THREAT"]
    return out


def eputrade_features(daily_index: pd.DatetimeIndex,
                      cache_dir: Path | None = None) -> pd.DataFrame:
    """EPU-trade (havi) lépcsős as-of illesztése: a hó M értéke csak
    hóvége + EPU_PUB_DELAY_CDAYS naptól él."""
    from quantimate.features.macro import fetch_fred_series
    s = fetch_fred_series("EPUTRADE", cache_dir, lag_days=0)  # nyers havi sor
    s = s.sort_index()
    chg12 = s.diff(12)
    avail = (s.index + pd.offsets.MonthEnd(0)
             + pd.Timedelta(days=EPU_PUB_DELAY_CDAYS))
    out = pd.DataFrame(index=daily_index,
                       columns=list(EPUTRADE_COLS), dtype=float)
    pos = np.searchsorted(avail.values, daily_index.values, side="right") - 1
    valid = pos >= 0
    out.loc[valid, "eputrade_level"] = s.to_numpy()[pos[valid]]
    out.loc[valid, "eputrade_chg12"] = chg12.to_numpy()[pos[valid]]
    return out


def build_geo_features(daily_index: pd.DatetimeIndex, use_gpr: bool = True,
                       use_tpu: bool = True,
                       cache_dir: Path | None = None) -> pd.DataFrame:
    """Geo-jellemzők a jellemző-index naptárára (a macro.py mintája: az
    elérhetetlen forrás kihagyva, a fő lánc nem törik)."""
    frames = []
    if use_gpr:
        try:
            frames.append(gpr_features(daily_index,
                                       fetch_gpr_daily(cache_dir)))
        except Exception as exc:
            warnings.warn(f"GPR skip: {exc}")
    if use_tpu:
        try:
            frames.append(eputrade_features(daily_index, cache_dir))
        except Exception as exc:
            warnings.warn(f"EPU-trade skip: {exc}")
    if not frames:
        return pd.DataFrame(index=daily_index)
    return pd.concat(frames, axis=1)
