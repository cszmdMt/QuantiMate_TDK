"""Historikus hír-szentiment (V7-4): FNSPID + FinBERT — a két alvó modul élesítése.

A V6 fúziós hálójának szentiment-ága nulla-vektoron "tanult", a Bayes-k
kalibrálatlan volt. A hiányzó láncszem egy historikus, időbélyeges hír-adatbázis:
az **FNSPID** (Dong–Fan–Peng 2024, KDD; HuggingFace: Zihan1004/FNSPID). A hírek
pontozására a pénzügyi **FinBERT** (Huang–Wang–Yang 2023) a standard.

**Méret:** az FNSPID `All_external.csv` 5,7 GB, Stock_symbol szerint
lexikálisan rendezve. NEM töltjük le egészben: **byte-offset bináris kereséssel**
megkeressük a NVDA-blokk határait, és CSAK azt a néhány MB-ot húzzuk le (a nyers
dump SOHA nem kerül a repóba; a szűrt kivonat is csak ha < 5 MB). Cache: parquet.

**Időzóna:** a hír-időbélyeg UTC; a piaczárás (New York 16:00) UTÁNI hír a
t+1 kereskedési naphoz tartozik — erre külön igazítás.

**Opcionális modell:** a FinBERT (torch/transformers) opcionális; hiányában
a modul őszinte "nem futott" jelzést ad, a fő lánc független marad.
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FNSPID_URL = ("https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
              "Stock_news/All_external.csv")
SYMBOL_COL = 2   # Date, Article_title, Stock_symbol, ...
HEADERS = {"User-Agent": "Mozilla/5.0"}


def finbert_available() -> bool:
    import importlib.util as u
    return u.find_spec("torch") is not None and u.find_spec("transformers") is not None


def _range_get(url: str, start: int, end: int) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={**HEADERS,
                                               "Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req, timeout=30).read()


def _symbol_at_offset(url: str, offset: int) -> tuple[str, int]:
    """A `offset` utáni ELSŐ teljes sor Stock_symbol-ja és a sor kezdő-offszetje.
    (Az All_external soronként egy rekord — üres Article, nincs beágyazott újsor.)"""
    chunk = _range_get(url, offset, offset + 4000).decode("utf-8", errors="replace")
    nl = chunk.find("\n")
    line = chunk[nl + 1:].split("\n", 1)[0] if nl >= 0 else chunk
    line_start = offset + nl + 1
    parts = line.split(",")
    sym = parts[SYMBOL_COL] if len(parts) > SYMBOL_COL else ""
    return sym.strip(), line_start


def find_symbol_block(url: str, symbol: str, file_size: int) -> tuple[int, int]:
    """A `symbol`-hoz tartozó [start, end] byte-tartomány bináris kereséssel.
    A fájl Stock_symbol szerint lexikálisan rendezett; ASCII-összehasonlítás."""
    def first_ge(target: str) -> int:
        lo, hi = 0, file_size
        while lo < hi:
            mid = (lo + hi) // 2
            sym, ls = _symbol_at_offset(url, mid)
            if sym and sym < target:
                lo = ls + len(sym) + 1
                lo = max(lo, mid + 1)
            else:
                hi = mid
        return lo

    start = first_ge(symbol)
    # a symbol utáni következő szimbólum kezdete = a blokk vége
    end = first_ge(symbol + "￿")
    return start, end


def fetch_fnspid_symbol(symbol: str = "NVDA", cache_dir: Path | None = None,
                        max_mb: float = 20.0) -> pd.DataFrame:
    """A `symbol` hírblokkja (Date, Article_title) — byte-range letöltéssel,
    parquet-cache-sel. A nyers CSV-blokk NEM kerül a repóba."""
    from quantimate.config import DATA_DIR
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"fnspid_{symbol}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    import urllib.request
    head = urllib.request.Request(FNSPID_URL, method="HEAD", headers=HEADERS)
    file_size = int(urllib.request.urlopen(head, timeout=30).headers["Content-Length"])

    start, end = find_symbol_block(FNSPID_URL, symbol, file_size)
    if end - start > max_mb * 1e6:
        end = start + int(max_mb * 1e6)   # biztonsági plafon
    raw = _range_get(FNSPID_URL, start, end).decode("utf-8", errors="replace")
    # az első és utolsó (csonka) sort eldobjuk, csak a symbol-egyezőket tartjuk
    header = "Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,Lsa_summary,Luhn_summary,Textrank_summary,Lexrank_summary"
    df = pd.read_csv(io.StringIO(header + "\n" + raw), on_bad_lines="skip",
                     engine="python")
    df = df[df["Stock_symbol"].astype(str).str.strip() == symbol]
    df = df[["Date", "Article_title"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df.to_parquet(cache)
    return df


def _market_adjusted_trading_date(ts_utc: pd.Series) -> pd.Series:
    """A New York-i piaczárás (16:00 ET) UTÁNI hír a KÖVETKEZŐ kereskedési
    naphoz tartozik. UTC→ET; ha az idő >= 16:00 ET, +1 nap; majd normalize."""
    et = ts_utc.dt.tz_convert("America/New_York")
    after_close = et.dt.hour >= 16
    d = et.dt.normalize()
    d = d.where(~after_close, d + pd.Timedelta(days=1))
    return d.dt.tz_localize(None)


def finbert_daily_sentiment(news: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    """FinBERT-pontozás a címeken → napi aggregált szentiment (átlag, szórás,
    hírszám). A pontszám ∈ [−1, +1]: P(pozitív) − P(negatív). Piaczárás-igazított
    kereskedési nap. torch/transformers hiányában RuntimeError."""
    if not finbert_available():
        raise RuntimeError("torch/transformers nem elérhető — a szentiment-szakasz kihagyva")
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    # ProsusAI/finbert címkék: 0=positive, 1=negative, 2=neutral
    titles = news["Article_title"].astype(str).tolist()
    scores = np.empty(len(titles))
    with torch.no_grad():
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=64)
            probs = torch.softmax(model(**enc).logits, dim=1).numpy()
            scores[i:i + batch_size] = probs[:, 0] - probs[:, 1]   # pos − neg

    out = news.copy()
    out["sentiment"] = scores
    out["trade_date"] = _market_adjusted_trading_date(out["Date"])
    daily = out.groupby("trade_date").agg(
        senti_mean=("sentiment", "mean"),
        senti_std=("sentiment", "std"),
        news_count=("sentiment", "size")).reset_index()
    daily["senti_std"] = daily["senti_std"].fillna(0.0)
    return daily.set_index("trade_date")


def nvda_daily_sentiment(cache_dir: Path | None = None) -> pd.DataFrame:
    """A teljes NVDA napi FinBERT-szentiment, parquet-cache-sel (a FinBERT-
    pontozás drága → egyszer fut). torch/transformers hiányában RuntimeError."""
    from quantimate.config import DATA_DIR
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR
    cache = cache_dir / "nvda_daily_sentiment.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    news = fetch_fnspid_symbol("NVDA", cache_dir=cache_dir)
    daily = finbert_daily_sentiment(news)
    daily.to_parquet(cache)
    return daily


def calibrate_bayes_k(daily_senti: pd.Series, next_dir: pd.Series,
                      k_grid=None) -> dict:
    """A Bayes-réteg k-ját historikus hír-adaton kalibráljuk: a k, amely
    minimalizálja a log-loss-t a (szentiment → másnapi irány) leképezésen.
    KIZÁRÓLAG a train-szakaszon (a hívó szűri). Ezzel az 'illusztratív' címke
    levehető (a k adatból jön, nem heurisztika)."""
    from sklearn.linear_model import LogisticRegression
    s = np.asarray(daily_senti, dtype=float).reshape(-1, 1)
    y = np.asarray(next_dir, dtype=int)
    lr = LogisticRegression(max_iter=1000).fit(s, y)
    # a logisztikus meredekség ~ a k·s kitevő skálája; visszaadjuk mint kalibrált k
    return {"k": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]),
            "train_acc": float(lr.score(s, y))}
