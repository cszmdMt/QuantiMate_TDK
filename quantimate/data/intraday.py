"""Intraday-aggregátum réteg (V8-1): Yahoo 60m barok → NAPI jellemzők.

Rögzített szakmai döntés: a napi gerinc
marad; órás barokon való közvetlen tréning TILOS (Merton 1980: a μ-becslés a
lefedett időtartamtól függ, nem a mintavételi sűrűségtől — az intraday adat a
σ-oldalt és a kalibrációt javíthatja). Az órás adat kizárólag napon belüli
aggregátumként lép be, a t nap jellemzője KIZÁRÓLAG a t nap RTH-barjaiból áll
elő → a t-napi záráskor ismert (PIT lag=0), és egy lezárt nap értéke későbbi
barok érkezésével nem változik (prefix-invariancia konstrukcióból).

Elérhetőség: a Yahoo 60m ~730 napra ad adatot → a jellemzők csak ott léteznek,
ahol van bar (előtte NaN) — a GBDT-ág natívan kezeli, ezért az A/B először
GBDT-n fut (a jellemzők az extras-ban utaznak, nem az X_scaled-ben).

Megjegyzés: a 60m lánc KORRIGÁLATLAN árakat ad; minden jellemző napon belüli
ARÁNY (hozam, R², VWAP-távolság, volumen-részarány), így az osztalék-korrekció
nem érinti — a napi zárás vs. utolsó bar konzisztencia-őr toleranciája ezt
figyelembe veszi.
"""
from __future__ import annotations

import datetime as dt
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quantimate.config import DATA_DIR
from quantimate.data.data import HEADERS, RETRY_DELAYS, _safe_name
from quantimate.data import pit

INTRADAY_RANGE_DAYS = 730         # Yahoo 60m maximum
RTH_START = dt.time(9, 30)        # New York-i kereskedési idő
RTH_END = dt.time(16, 0)

INTRADAY_FEATURES = ("rv_1h", "bpv", "jump", "r_first_hour", "r_last_hour",
                     "intraday_trend_r2", "vwap_gap", "volume_share_last_hour",
                     "n_bars")

# PIT-regisztráció (V8-10): minden oszlop a t nap RTH-barjaiból, záráskor ismert.
pit.register(r"(rv_1h|bpv|jump|r_first_hour|r_last_hour|intraday_trend_r2"
             r"|vwap_gap|volume_share_last_hour|n_bars)",
             source="yahoo_60m (RTH, korrigálatlan)", lag_days=0,
             asof_rule="a t nap 9:30–16:00 ET barjaiból aggregálva — lezárt "
                       "nap értéke későbbi adattól nem változik",
             notes="~730 nap elérhetőség; előtte NaN (availability-mask), "
                   "GBDT-ág kezeli natívan")


def fetch_intraday_60m(ticker: str = "NVDA", cache_dir: Path | None = None,
                       force_refresh: bool = False) -> pd.DataFrame:
    """60 perces OHLCV-barok a Yahoo chart API-ról, ET-időbélyeggel (tz-aware).

    Cache: data/{TICKER}_60m.parquet — aznapi fájlból hálózat nélkül; hálózati
    hiba + régi cache → figyelmeztetés mellett a cache (data.py-minta)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_safe_name(ticker)}_60m.parquet"

    if cache_file.exists() and not force_refresh:
        mtime = dt.date.fromtimestamp(cache_file.stat().st_mtime)
        if mtime == dt.date.today():
            return pd.read_parquet(cache_file)

    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?range={INTRADAY_RANGE_DAYS}d&interval=60m")
    last_exc = None
    for delay in (0,) + RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            break
        except Exception as exc:
            last_exc = exc
    else:
        if cache_file.exists():
            warnings.warn(f"{ticker} 60m: hálózati hiba — régebbi cache-ből "
                          f"({last_exc}).")
            return pd.read_parquet(cache_file)
        raise ConnectionError(f"{ticker} 60m elérhetetlen és nincs cache "
                              f"({last_exc})")

    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True)
    idx = idx.tz_convert("America/New_York")   # tz-AWARE marad (napon belüli idő!)
    bars = pd.DataFrame({
        "Open": np.asarray(q["open"], dtype=float),
        "High": np.asarray(q["high"], dtype=float),
        "Low": np.asarray(q["low"], dtype=float),
        "Close": np.asarray(q["close"], dtype=float),
        "Volume": np.asarray(q["volume"], dtype=float),
    }, index=idx)
    bars = bars[~bars.index.duplicated(keep="last")]
    bars = bars[bars[["Open", "High", "Low", "Close"]].notna().all(axis=1)]
    bars.index.name = "Datetime"
    bars.to_parquet(cache_file)
    return bars


def filter_rth(bars: pd.DataFrame) -> pd.DataFrame:
    """Csak a szabályos kereskedési idő (9:30 ≤ t < 16:00 ET) barjai —
    pre/post-market kizárva. tz-naiv indexnél ET-ként értelmezzük (teszt-út)."""
    t = bars.index.time
    return bars[(t >= RTH_START) & (t < RTH_END)]


def _day_features(day: pd.DataFrame) -> dict:
    """Egy nap RTH-barjaiból a 9 aggregátum. Minden mennyiség napon belüli."""
    c = day["Close"].to_numpy()
    o = float(day["Open"].iloc[0])
    v = day["Volume"].to_numpy()
    # Napon belüli hozamok: első bar nyitó→záró, majd bar záró→záró.
    r = np.diff(np.log(np.concatenate([[o], c])))
    n = len(r)

    rv = float(np.sum(r ** 2))
    bpv = float((np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))) if n >= 2 else np.nan
    jump = max(rv - bpv, 0.0) if np.isfinite(bpv) else np.nan

    # Napon belüli trend-R²: log-záróárak lineáris illesztése a bar-indexre.
    if n >= 3:
        y = np.log(c)
        x = np.arange(len(y), dtype=float)
        ssr = np.sum((y - np.poly1d(np.polyfit(x, y, 1))(x)) ** 2)
        sst = np.sum((y - y.mean()) ** 2)
        trend_r2 = float(1.0 - ssr / sst) if sst > 0 else np.nan
    else:
        trend_r2 = np.nan

    typical = (day["High"] + day["Low"] + day["Close"]).to_numpy() / 3.0
    vol_sum = float(v.sum())
    vwap = float((typical * v).sum() / vol_sum) if vol_sum > 0 else np.nan
    vwap_gap = float(c[-1] / vwap - 1.0) if np.isfinite(vwap) else np.nan

    return {
        "rv_1h": rv, "bpv": bpv, "jump": jump,
        "r_first_hour": float(r[0]), "r_last_hour": float(r[-1]),
        "intraday_trend_r2": trend_r2, "vwap_gap": vwap_gap,
        "volume_share_last_hour": (float(v[-1] / vol_sum) if vol_sum > 0 else np.nan),
        "n_bars": float(len(day)),
    }


def build_intraday_daily(bars: pd.DataFrame) -> pd.DataFrame:
    """60m barok → napi jellemző-tábla (tz-naiv napi DatetimeIndex).

    Félnapok: kevesebb bar (n_bars minőségjelző), a jellemzők ettől még
    értelmezettek. Egy-baros nap: bpv/jump/trend_r2 NaN (nincs bar-pár)."""
    rth = filter_rth(bars)
    if len(rth) == 0:
        return pd.DataFrame(columns=list(INTRADAY_FEATURES))
    dates = pd.DatetimeIndex(rth.index.tz_localize(None)
                             if rth.index.tz is not None else rth.index).normalize()
    rows, out_idx = [], []
    for d, day in rth.groupby(dates):
        rows.append(_day_features(day))
        out_idx.append(d)
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(out_idx, name="Date"))
    return out.sort_index()


def build_intraday_features(daily_index: pd.DatetimeIndex,
                            ticker: str = "NVDA",
                            cache_dir: Path | None = None) -> pd.DataFrame:
    """A napi pipeline-indexre igazított intraday-jellemzők (availability-mask:
    a 60m lefedettség előtti napokon NaN — SOHA nem töltjük fel)."""
    bars = fetch_intraday_60m(ticker, cache_dir=cache_dir)
    daily = build_intraday_daily(bars)
    return daily.reindex(daily_index)


def consistency_report(daily_close: pd.Series, bars: pd.DataFrame,
                       tol: float = 0.01) -> pd.DataFrame:
    """Split-őr: a napi (korrigált) zárás ≈ az utolsó RTH-bar zárója.

    A 60m lánc korrigálatlan → osztalék-napokon apró rés jogos; tol feletti
    eltérés split-/adathiba gyanú. Visszaadja az átfedés eltérés-tábláját."""
    rth = filter_rth(bars)
    dates = pd.DatetimeIndex(rth.index.tz_localize(None)
                             if rth.index.tz is not None else rth.index).normalize()
    last_close = rth["Close"].groupby(dates).last()
    both = pd.DataFrame({"daily": daily_close, "bar": last_close}).dropna()
    both["rel_diff"] = (both["bar"] / both["daily"] - 1.0).abs()
    both["ok"] = both["rel_diff"] <= tol
    return both
