"""Explicit snapshot creation and offline, checksummed reads. No network at import."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import urllib.parse

import numpy as np
import pandas as pd

from quantimate.data.data import _safe_name, parse_chart_result, drop_null_ohlc_rows, apply_adjustment
from quantimate.experiment_protocol import (RAW_START, DATE_PAIRS, HORIZONS, atomic_json, digest,
    file_hash, read_json)
from quantimate.features.pooled import POOLED_UNIVERSE

CORE = ("NVDA", "SPY", "QQQ", "AMD", "SMH")
LIGHT = ("^VIX", "^VXN", "^TNX")
ALL_TICKERS = tuple(sorted(set(CORE + LIGHT + POOLED_UNIVERSE)))
OPTIONAL = ("nvda_earnings_verified.csv", "NVDA_60m.parquet",
            "gpr_daily.parquet", "gpr_daily.meta.json", "fred_EPUTRADE.parquet",
            "finra_si_NVDA.parquet", "regsho_NVDA.parquet", "edgar_cache/form4_nvda.parquet")
SNAPSHOT_SCHEMA = 2
# Calendar days kept after the last cutoff so the longest horizon has an observed target.
TARGET_MARGIN_DAYS = 2 * max(HORIZONS) + 10


def validate_daily(frame, ticker):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None:
        raise ValueError(f"{ticker}: expected timezone-naive daily index")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique or frame.empty:
        raise ValueError(f"{ticker}: empty, unsorted or duplicate dates")
    if not frame.index.equals(frame.index.normalize()):
        raise ValueError(f"{ticker}: not daily bars")
    prefix = _safe_name(ticker)
    required = [f"{prefix}_{c}" for c in ("Open", "High", "Low", "Close", "Volume")]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{ticker}: missing OHLCV columns")
    prices = frame[required[:4]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"{ticker}: invalid adjusted OHLC prices")


def download_daily(ticker, end):
    import requests
    start_ts = int(pd.Timestamp(RAW_START, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + urllib.parse.quote(ticker, safe="")
    response = requests.get(url, params=dict(period1=start_ts, period2=end_ts,
        interval="1d", events="div,splits", includeAdjustedClose="true"),
        headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
    response.raise_for_status()
    chart = response.json()["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Yahoo daily download failed for {ticker}: {chart.get('error')}")
    result = chart["result"][0]
    if result.get("meta", {}).get("dataGranularity") != "1d":
        raise ValueError(f"{ticker}: Yahoo did not return daily data")
    if ticker not in LIGHT and not result.get("indicators", {}).get("adjclose"):
        raise ValueError(f"{ticker}: Yahoo omitted adjusted close; raw-price fallback is forbidden")
    raw = drop_null_ohlc_rows(parse_chart_result(result, ticker), ticker)
    if "AdjClose" not in raw or raw.AdjClose.isna().any():
        raise ValueError(f"{ticker}: adjusted close unavailable")
    adjusted = apply_adjustment(raw).add_prefix(_safe_name(ticker) + "_")
    adjusted = adjusted.loc[RAW_START:]
    return adjusted, result.get("meta", {}).get("firstTradeDate")


def validate_pairs(nvda):
    for cutoff, forecast in DATE_PAIRS:
        c, f = pd.Timestamp(cutoff), pd.Timestamp(forecast)
        if c not in nvda.index or f not in nvda.index:
            raise ValueError(f"Missing NVDA dates: {cutoff} / {forecast}")
        if nvda.index.get_loc(f) != nvda.index.get_loc(c) + 1:
            raise ValueError(f"Not consecutive observed trading sessions: {cutoff} / {forecast}")
        session_after(nvda.index, cutoff, max(HORIZONS))


def session_after(index, cutoff, horizon):
    """The trading session `horizon` observed sessions after the cutoff."""
    position = index.get_loc(pd.Timestamp(cutoff)) + horizon
    if position >= len(index):
        raise ValueError(f"No observed {horizon}-session target after {cutoff}")
    return str(index[position].date())


def check_optional(cache_dir, allow_missing=()):
    """Optional sources must be present unless their absence is explicitly accepted.

    A source that is silently missing removes its data layer from every date pair
    (the saved schema-1 run lost all V8 layers this way).
    """
    unknown = sorted(set(allow_missing) - set(OPTIONAL))
    if unknown:
        raise ValueError("Unknown optional sources: " + ", ".join(unknown))
    missing = [name for name in OPTIONAL if not (Path(cache_dir) / name).exists()]
    refused = [name for name in missing if name not in allow_missing]
    if refused:
        raise FileNotFoundError(
            "Optional sources missing from " + str(cache_dir) + ": " + ", ".join(refused)
            + ". Their data layers would not take part in the experiment. Provide the files,"
              " or accept each absence explicitly with --allow-missing. Nothing was trained.")
    return missing


def create_snapshot(destination, cache_dir, *, download=False, allow_missing=()):
    """Explicit user command only. Existing snapshots are never overwritten."""
    destination, cache_dir = Path(destination).resolve(), Path(cache_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Snapshot already exists: {destination}")
    # Before any download: a missing data layer is an error, not a footnote.
    accepted_missing = check_optional(cache_dir, allow_missing)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    end = (pd.Timestamp(max(c for c, _ in DATE_PAIRS))
           + pd.Timedelta(days=TARGET_MARGIN_DAYS)).date().isoformat()
    manifest = dict(schema=SNAPSHOT_SCHEMA, raw_start=RAW_START, end_exclusive=end,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    adjusted_prices=True, source="Yahoo daily chart" if download else "existing adjusted parquet cache",
                    files={}, coverage={}, missing_optional=accepted_missing,
                    caveats=["Frozen current-vintage adjusted prices; not historical Yahoo vintages.",
                             "Pooled universe has survivorship bias.",
                             "Optional sources can have shorter coverage and revised historical values."])
    try:
        frames = {}
        for ticker in ALL_TICKERS:
            name = _safe_name(ticker) + ".parquet"
            first_trade = None
            if download:
                frame, first_trade = download_daily(ticker, end)
            else:
                source = cache_dir / name
                if not source.exists():
                    raise FileNotFoundError(f"Missing daily source {source}; use explicit snapshot --download")
                frame = pd.read_parquet(source).loc[RAW_START:]
                frame = frame.loc[frame.index < pd.Timestamp(end)]
            validate_daily(frame, ticker)
            if ticker in CORE + LIGHT and frame.index[0] > pd.Timestamp(RAW_START) + pd.Timedelta(days=7):
                raise ValueError(f"{ticker}: cache begins {frame.index[0].date()}, required start {RAW_START}. "
                                 "Create a new snapshot with --download; nothing was trained.")
            if first_trade is not None:
                expected_start = max(pd.Timestamp(RAW_START), pd.Timestamp(first_trade, unit="s", tz="UTC").tz_localize(None).normalize())
                if frame.index[0] > expected_start + pd.Timedelta(days=10):
                    raise ValueError(f"{ticker}: truncated downloaded history")
            frame.to_parquet(stage / name)
            frames[ticker] = frame
            manifest["coverage"][ticker] = dict(first=str(frame.index[0].date()),
                last=str(frame.index[-1].date()), rows=len(frame), first_trade_unix=first_trade)
        validate_pairs(frames["NVDA"])
        sessions = frames["NVDA"].index
        for ticker in CORE:
            missing = sessions.symmetric_difference(frames[ticker].index)
            if len(missing):
                raise ValueError(f"{ticker}: {len(missing)} missing NVDA trading sessions")
        for ticker in POOLED_UNIVERSE:
            frame = frames[ticker]
            expected = sessions[(sessions >= frame.index[0]) & (sessions <= frame.index[-1])]
            if len(expected.difference(frame.index)):
                raise ValueError(f"{ticker}: missing sessions inside pooled history")
            if frame.index[-1] < pd.Timestamp(max(f for _, f in DATE_PAIRS)):
                raise ValueError(f"{ticker}: pooled history ends before the last forecast")
        for relative in OPTIONAL:
            if relative in accepted_missing:
                continue
            source, target = cache_dir / relative, stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            # Parse now, fail rather than silently omit a damaged optional file.
            if source.suffix == ".parquet":
                pd.read_parquet(target)
            elif source.suffix == ".csv":
                pd.read_csv(target)
            else:
                read_json(target)
        for file in sorted(stage.rglob("*")):
            if file.is_file():
                manifest["files"][file.relative_to(stage).as_posix()] = file_hash(file)
        manifest["snapshot_id"] = digest(manifest)
        atomic_json(stage / "snapshot.json", manifest)
        stage.rename(destination)
    except BaseException:
        # Only the temporary directory created by this call is removed.
        shutil.rmtree(stage)
        raise
    return manifest


def verify_snapshot(path):
    path = Path(path).resolve()
    m = read_json(path / "snapshot.json")
    if digest({k: v for k, v in m.items() if k != "snapshot_id"}) != m["snapshot_id"]:
        raise ValueError("Snapshot manifest checksum mismatch")
    for relative, expected in m["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path) or file_hash(file) != expected:
            raise ValueError(f"Snapshot changed or invalid path: {relative}")
    return m


class Snapshot:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = verify_snapshot(self.path)

    def daily(self, ticker, cutoff=None):
        frame = pd.read_parquet(self.path / (_safe_name(ticker) + ".parquet"))
        return frame.loc[:cutoff].copy() if cutoff is not None else frame

    def universe(self, cutoff, ticker="NVDA", generic=False):
        own = self.daily(ticker, cutoff)
        index = own.index
        names = (ticker, "SPY", "QQQ", "SMH") if generic else CORE
        parts = [self.daily(t, cutoff).reindex(index) for t in dict.fromkeys(names)]
        for t in LIGHT:
            series = self.daily(t, cutoff)[[_safe_name(t) + "_Close"]]
            parts.append(series.reindex(index).ffill(limit=3))
        frame = pd.concat(parts, axis=1)
        if frame.loc[pd.Timestamp(cutoff)-pd.DateOffset(years=10):].isna().any().any():
            raise ValueError(f"Missing daily input in {ticker}'s 10-year window")
        return frame

    def target_date(self, cutoff, horizon):
        """Calendar lookup only; no price column is read."""
        index = pd.read_parquet(self.path / "NVDA.parquet", columns=[]).index
        return session_after(index, cutoff, horizon)

    def optional(self, name):
        if name not in self.manifest["files"]:
            return None
        path = self.path / name
        return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
