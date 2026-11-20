"""Jellemző-képzés (V5 indikátorok modulba emelve + B3 új jellemzők).

Anti-leakage szabály: ebben a fájlban shift csak POZITÍV lag-gel
fordulhat elő (múltra nézés); erre statikus grep-teszt fut a tests/ alatt.
Minden rolling/EWM ablak kizárólag múltbeli adatot használ.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def wilder_ema(s: pd.Series, n: int) -> pd.Series:
    """Wilder-féle simítás (RSI/ATR/ADX eredeti képlete): alpha = 1/n."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def ewma_volatility(log_ret: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA-volatilitás — kauzális (csak múltbeli hozamokból)."""
    return np.sqrt((log_ret ** 2).ewm(alpha=1 - lam, adjust=False).mean())


def compute_indicators(d: pd.DataFrame, t: str, full: bool = True) -> pd.DataFrame:
    """Egy OHLCV-ticker indikátorai. full=False → könnyű makro-készlet (5 oszlop)."""
    o = pd.DataFrame(index=d.index)
    C, H, L, V = d[f"{t}_Close"], d[f"{t}_High"], d[f"{t}_Low"], d[f"{t}_Volume"]

    r = np.log(C / C.shift(1))
    o[f"{t}_log_ret"] = r
    o[f"{t}_ewma_vol"] = ewma_volatility(r)

    sma20, sma50 = C.rolling(20).mean(), C.rolling(50).mean()
    o[f"{t}_dist_sma20"] = (C - sma20) / sma20
    o[f"{t}_dist_sma50"] = (C - sma50) / sma50

    delta = C.diff()
    gain = wilder_ema(delta.clip(lower=0), 14)
    loss = wilder_ema((-delta).clip(lower=0), 14)
    o[f"{t}_rsi14"] = 100 - 100 / (1 + gain / loss)

    if not full:
        return o

    ema12, ema26 = C.ewm(span=12, adjust=False).mean(), C.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    o[f"{t}_macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / C

    std20 = C.rolling(20).std()
    upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
    o[f"{t}_boll_pctB"] = (C - lower) / (upper - lower)
    o[f"{t}_boll_bw"] = (upper - lower) / sma20

    tr = pd.concat([H - L, (H - C.shift(1)).abs(), (L - C.shift(1)).abs()], axis=1).max(axis=1)
    o[f"{t}_atr_norm"] = wilder_ema(tr, 14) / C

    up_move, down_move = H.diff(), -L.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=d.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=d.index)
    atr_raw = wilder_ema(tr, 14)
    plus_di = 100 * wilder_ema(plus_dm, 14) / atr_raw
    minus_di = 100 * wilder_ema(minus_dm, 14) / atr_raw
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    o[f"{t}_adx14"] = wilder_ema(dx, 14)
    o[f"{t}_di_spread"] = (plus_di - minus_di) / 100

    l14, h14 = L.rolling(14).min(), H.rolling(14).max()
    k = 100 * (C - l14) / (h14 - l14)
    o[f"{t}_stoch_k"] = k
    o[f"{t}_stoch_d"] = k.rolling(3).mean()

    o[f"{t}_roc10"] = 100 * (C - C.shift(10)) / C.shift(10)

    obv = (np.sign(C.diff()).fillna(0) * V).cumsum()
    o[f"{t}_obv_z"] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std()
    o[f"{t}_vol_z"] = (V - V.rolling(20).mean()) / V.rolling(20).std()

    return o


def compute_index_features(d: pd.DataFrame, t: str) -> pd.DataFrame:
    """Volume nélküli index (VIX/VXN/TNX) könnyű készlete: szint, 5 napos
    változás, dist_sma20 (B3a/B3c). A közös compute_indicators KeyError-t dobna
    a hiányzó Volume miatt — ezért külön függvény (hibakatalógus 6.)."""
    o = pd.DataFrame(index=d.index)
    level = d[f"{t}_Close"]
    o[f"{t}_level"] = level
    o[f"{t}_chg5"] = level.diff(5)
    sma20 = level.rolling(20).mean()
    o[f"{t}_dist_sma20"] = (level - sma20) / sma20
    return o


def overnight_intraday(d: pd.DataFrame, t: str) -> pd.DataFrame:
    """B3e: overnight és intraday hozam szétbontása.
    r_on = log(Open_t / Close_{t-1}) — kauzális (a t-napi nyitó ár t-napon ismert);
    r_id = log(Close_t / Open_t)."""
    o = pd.DataFrame(index=d.index)
    o[f"{t}_r_overnight"] = np.log(d[f"{t}_Open"] / d[f"{t}_Close"].shift(1))
    o[f"{t}_r_intraday"] = np.log(d[f"{t}_Close"] / d[f"{t}_Open"])
    return o


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """B3d: naptári jellemzők — hét napja (sin/cos), kereskedési hónap-vége
    dummy, opciós lejárati (3. péntek) dummy. Determinisztikus naptári tények,
    előre ismertek → nem szivárognak."""
    o = pd.DataFrame(index=index)
    dow = index.dayofweek.values.astype(float)
    o["cal_dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    o["cal_dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    ym = pd.Series(index.to_period("M"), index=index)
    o["cal_month_end"] = (~ym.duplicated(keep="last")).astype(float).values
    o["cal_opex_friday"] = ((index.dayofweek == 4)
                            & (index.day >= 15) & (index.day <= 21)).astype(float)
    return o


def generate_earnings_candidates(close: pd.Series, threshold_sigma: float = 4.0,
                                 months=(2, 5, 8, 11)) -> pd.DatetimeIndex:
    """Earnings-nap JELÖLTEK a historikus árfolyamból (B3b tartalék-út).

    |napi log-hozam| > threshold_sigma × gördülő σ(252) napok, az NVDA
    jelentési hónapjaira (febr/máj/aug/nov) szűrve, negyedévenként a
    legnagyobb mozgású nap. FELÜLVIZSGÁLATRA SZORUL: a jelölt a jelentést
    KÖVETŐ nagy mozgás napja, nem maga a bejelentés napja.
    """
    r = np.log(close / close.shift(1))
    sigma = r.rolling(252, min_periods=20).std()
    hits = r.index[(r.abs() > threshold_sigma * sigma) & r.index.month.isin(months)]
    if len(hits) == 0:
        return pd.DatetimeIndex([])
    # Negyedévenként (év, hónap) a legnagyobb |mozgás|
    hit_df = pd.DataFrame({"absr": r.loc[hits].abs()}, index=hits)
    grouped = hit_df.groupby([hit_df.index.year, hit_df.index.month])["absr"].idxmax()
    return pd.DatetimeIndex(sorted(grouped.values))


def earnings_features(index: pd.DatetimeIndex,
                      earnings_dates: pd.DatetimeIndex | None,
                      clip_days: int = 30) -> pd.DataFrame:
    """B3b: napok a következő earnings-ig / az előző óta (clip 30) + earnings-hét dummy.

    Feature-flag: earnings_dates=None vagy üres → csupa nulla oszlopok +
    figyelmeztetés (hibakatalógus 15.), a pipeline nem törik el.

    Megjegyzés (leakage-audit): a valós earnings-naptár hetekkel előre nyilvános
    információ, ezért a "napok a következőig" jellemző kereskedésileg jogos.
    A historikus árfolyamból GENERÁLT jelöltnaptár esetén ez a jövőbeli nagy
    mozgás napját használja proxyként — ezt a notebook explicit dokumentálja.
    """
    o = pd.DataFrame(index=index)
    if earnings_dates is None or len(earnings_dates) == 0:
        warnings.warn("Nincs earnings-naptár — az earnings-jellemzők nullák.")
        o["earn_days_to_next"] = 0.0
        o["earn_days_since_prev"] = 0.0
        o["earn_week"] = 0.0
        return o

    # Az earnings-dátumokat a kereskedési naptár pozícióira képezzük.
    pos_of_date = np.searchsorted(index.values, pd.DatetimeIndex(earnings_dates).values)
    pos_of_date = np.unique(np.clip(pos_of_date, 0, len(index) - 1))
    all_pos = np.arange(len(index))

    nxt = np.searchsorted(pos_of_date, all_pos, side="left")
    days_to_next = np.where(nxt < len(pos_of_date),
                            pos_of_date[np.minimum(nxt, len(pos_of_date) - 1)] - all_pos,
                            clip_days)
    prv = np.searchsorted(pos_of_date, all_pos, side="right") - 1
    days_since_prev = np.where(prv >= 0, all_pos - pos_of_date[np.maximum(prv, 0)],
                               clip_days)

    o["earn_days_to_next"] = np.clip(days_to_next, 0, clip_days).astype(float)
    o["earn_days_since_prev"] = np.clip(days_since_prev, 0, clip_days).astype(float)
    o["earn_week"] = ((np.clip(days_to_next, 0, clip_days) <= 2)
                      | (np.clip(days_since_prev, 0, clip_days) <= 2)).astype(float)
    return o


def build_features(df: pd.DataFrame, target: str, ohlcv_tickers, light_tickers,
                   earnings_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """A teljes jellemző-mátrix a TELJES idősoron (target nélkül).

    NVDA: teljes készlet; többi OHLCV-ticker: könnyű készlet; VIX/VXN/TNX:
    index-készlet; + earnings + naptár + overnight/intraday.
    """
    from quantimate.data.data import _safe_name

    frames = [compute_indicators(df, target, full=True)]
    for t in ohlcv_tickers:
        if t != target:
            frames.append(compute_indicators(df, _safe_name(t), full=False))
    for t in light_tickers:
        frames.append(compute_index_features(df, _safe_name(t)))
    frames.append(overnight_intraday(df, target))
    frames.append(calendar_features(df.index))
    frames.append(earnings_features(df.index, earnings_dates))
    return pd.concat(frames, axis=1)
