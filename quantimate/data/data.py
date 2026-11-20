"""Adatréteg: Yahoo v8 chart API letöltés osztalék-korrekcióval,
retry+cache-sel és időzóna-audittal.

Kulcselvek:
- Az adjclose-t NEM önmagában tesszük a Close helyére: korrekciós faktort
  számolunk (f_t = adjclose_t / close_t), és azzal szorozzuk az
  Open/High/Low/Close mindegyikét — a Volume-ot nem. Így az ATR/Stochastic/
  Bollinger konzisztens marad.
- Null OHLC-sorok auditált eldobása, duplikált timestampek keep-last.
- tz: UTC → America/New_York → normalize (a korábbi tz_localize(None) helyett).
- Cache: data/*.parquet; aznapi fájlból hálózat nélkül töltünk.
"""
from __future__ import annotations

import time
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quantimate.config import DATA_DIR

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
}

RETRY_DELAYS = (2, 4, 8)


def _safe_name(ticker: str) -> str:
    """Fájlnévbiztos ticker-név: '^VIX' → 'VIX'."""
    return ticker.replace("^", "").replace("=", "_")


def _fetch_chart_json(ticker: str, years: int) -> dict:
    """Nyers chart-JSON letöltése exponenciális backoff-os újrapróbálkozással."""
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?range={years}y&interval=1d")
    last_exc: Exception | None = None
    for attempt, delay in enumerate((0,) + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()["chart"]["result"][0]
        except Exception as exc:  # hálózati hiba, 429, JSON-séma hiba
            last_exc = exc
    raise ConnectionError(f"{ticker}: a Yahoo chart API {1 + len(RETRY_DELAYS)} "
                          f"kísérlet után sem elérhető ({last_exc})")


def parse_chart_result(res: dict, ticker: str) -> pd.DataFrame:
    """Chart-JSON → nyers (korrigálatlan) OHLCV+AdjClose DataFrame, tz-audittal.

    A timestampek UTC-ben érkeznek; New York-i idővé konvertálva, majd napra
    normálva kapjuk a helyes kereskedési dátumot (pre/post-market anomáliák
    napcsúszása ellen).
    """
    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True)
    idx = idx.tz_convert("America/New_York").normalize().tz_localize(None)

    df = pd.DataFrame({
        "Open": np.asarray(q["open"], dtype=float),
        "High": np.asarray(q["high"], dtype=float),
        "Low": np.asarray(q["low"], dtype=float),
        "Close": np.asarray(q["close"], dtype=float),
        "Volume": np.asarray(q["volume"], dtype=float),
    }, index=idx)

    adj = res["indicators"].get("adjclose")
    if adj and adj[0].get("adjclose") is not None:
        df["AdjClose"] = np.asarray(adj[0]["adjclose"], dtype=float)
    else:
        df["AdjClose"] = df["Close"]  # indexeknél (VIX/TNX) nincs külön adjclose

    # Duplikált timestampek: az utolsót tartjuk meg.
    df = df[~df.index.duplicated(keep="last")]
    return df


def drop_null_ohlc_rows(df: pd.DataFrame, ticker: str = "?") -> pd.DataFrame:
    """Sor eldobása, ha BÁRMELY OHLC mező null — auditált (kiírt) darabszámmal."""
    mask = df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    n_drop = int((~mask).sum())
    if n_drop:
        print(f"  FIGYELEM {ticker}: {n_drop} null-OHLC sor eldobva")
    return df[mask]


def apply_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Osztalék/split-korrekció a teljes OHLC-re a faktor-módszerrel.

    f_t = AdjClose_t / Close_t; O/H/L/C ← f_t·O/H/L/C, Volume változatlan.
    Érvényességi őr: f_t ∈ (0, 1.05] — ezen kívüli faktor adathibát jelez.
    """
    out = df.copy()
    f = out["AdjClose"] / out["Close"]
    bad = ~((f > 0) & (f <= 1.05))
    if bad.any():
        raise ValueError(f"Érvénytelen korrekciós faktor {int(bad.sum())} napon "
                         f"(min={f.min():.4f}, max={f.max():.4f}) — adathiba gyanú.")
    for col in ("Open", "High", "Low", "Close"):
        out[col] = out[col] * f
    return out.drop(columns=["AdjClose"])


def fetch_ohlcv(ticker: str, years: int = 10, cache_dir: Path | None = None,
                force_refresh: bool = False) -> pd.DataFrame:
    """Egy ticker korrigált OHLCV-je '{TICKER}_Open' stb. oszlopnevekkel.

    Cache-viselkedés: aznapi parquet → hálózat nélkül; friss letöltés után
    parquet-írás; hálózati hiba + bármilyen (akár régi) cache → figyelmeztetés
    mellett a cache; hálózati hiba + nincs cache → érthető hibaüzenet.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_safe_name(ticker)}.parquet"

    if cache_file.exists() and not force_refresh:
        mtime = dt.date.fromtimestamp(cache_file.stat().st_mtime)
        if mtime == dt.date.today():
            return pd.read_parquet(cache_file)

    try:
        res = _fetch_chart_json(ticker, years)
    except ConnectionError:
        if cache_file.exists():
            warnings.warn(f"{ticker}: hálózati hiba — RÉGEBBI cache-ből töltve "
                          f"({cache_file.name}).")
            return pd.read_parquet(cache_file)
        raise ConnectionError(
            f"{ticker}: a hálózat elérhetetlen és nincs cache ({cache_file}). "
            f"Szintetikus adatot nem generálunk — futtasd újra működő hálózattal."
        ) from None

    raw = parse_chart_result(res, ticker)
    raw = drop_null_ohlc_rows(raw, ticker)
    adj = apply_adjustment(raw)
    adj.columns = [f"{_safe_name(ticker)}_{c}" for c in adj.columns]
    adj.index.name = "Date"

    adj.to_parquet(cache_file)
    return adj


def assemble_universe(ohlcv_frames: list[pd.DataFrame],
                      light_frames: list[pd.DataFrame],
                      ohlcv_tickers) -> pd.DataFrame:
    """Összefűzés + naptár-illesztés — a letöltéstől FÜGGETLEN, tiszta logika.

    A golden-run teszt is ezt hívja fagyasztott parquet-ekből, így a
    letöltő- és az összefűző-út bitre azonos eredményt ad azonos bemenetre.
    """
    light = [f.ffill(limit=3) for f in light_frames]
    df = pd.concat(list(ohlcv_frames) + light, axis=1)
    ohlcv_cols = [c for c in df.columns
                  if c.split("_")[0] in {_safe_name(t) for t in ohlcv_tickers}]
    df = df.dropna(subset=ohlcv_cols)
    # A light oszlopok maradék NaN-jait (pl. eltérő kereskedési naptár) ffilllel
    # pótoljuk — kauzális, csak múltbeli értéket használ.
    df = df.ffill(limit=3)
    return df


def load_universe(ohlcv_tickers, light_tickers=(), years: int = 10,
                  cache_dir: Path | None = None) -> pd.DataFrame:
    """A teljes adat-univerzum egyetlen, dátum-illesztett DataFrame-ben.

    A light (Volume nélküli) tickereknél csak a Close-t tartjuk meg
    '{TICKER}_Close' néven; a NaN-okat legfeljebb 3 napig visszatöltjük (ffill).
    Az OHLCV-tickerek metszet-naptárára szűkítünk (dropna az OHLCV-oszlopokon).
    """
    ohlcv_frames, light_frames = [], []
    for t in ohlcv_tickers:
        d = fetch_ohlcv(t, years=years, cache_dir=cache_dir)
        ohlcv_frames.append(d)
        print(f"  OK {t}: {len(d)} nap")
    for t in light_tickers:
        d = fetch_ohlcv(t, years=years, cache_dir=cache_dir)
        light_frames.append(d[[f"{_safe_name(t)}_Close"]])
        print(f"  OK {t} (light): {len(d)} nap")
    return assemble_universe(ohlcv_frames, light_frames, ohlcv_tickers)


def load_universe_from_dir(src_dir: Path, ohlcv_tickers, light_tickers=(),
                           end_date=None) -> pd.DataFrame:
    """Univerzum FAGYASZTOTT parquet-könyvtárból — hálózat és cache-frissítés
    nélkül (golden-run teszt). Opcionális end_date-re szeletel."""
    src_dir = Path(src_dir)
    ohlcv_frames, light_frames = [], []
    for t in ohlcv_tickers:
        d = pd.read_parquet(src_dir / f"{_safe_name(t)}.parquet")
        ohlcv_frames.append(d)
    for t in light_tickers:
        d = pd.read_parquet(src_dir / f"{_safe_name(t)}.parquet")
        light_frames.append(d[[f"{_safe_name(t)}_Close"]])
    df = assemble_universe(ohlcv_frames, light_frames, ohlcv_tickers)
    if end_date is not None:
        df = df.loc[:pd.Timestamp(end_date)]
    return df
