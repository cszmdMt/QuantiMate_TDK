"""V7-1 tesztek: Yang–Zhang kézi ellenőrzés + off-by-one, HAR train-only
és kauzalitás (prefix-invariancia)."""
import numpy as np
import pandas as pd
import pytest

from quantimate.features.volatility import har_rv_forecast, yang_zhang_sigma


def _ohlc_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0.001, 0.005, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"NVDA_Open": open_, "NVDA_High": high,
                         "NVDA_Low": low, "NVDA_Close": close}, index=idx)


def test_yang_zhang_hand_computed():
    """A t-napi YZ bitre egyezik a képlet kézi (numpy) kiszámításával."""
    df = _ohlc_frame()
    w = 21
    yz = yang_zhang_sigma(df, "NVDA", window=w)

    t = 40
    sl = slice(t - w + 1, t + 1)
    O, H = df["NVDA_Open"].to_numpy(), df["NVDA_High"].to_numpy()
    L, C = df["NVDA_Low"].to_numpy(), df["NVDA_Close"].to_numpy()
    r_o = np.log(O[1:] / C[:-1])
    r_o = np.concatenate([[np.nan], r_o])
    r_c = np.log(C / O)
    rs = np.log(H / O) * np.log(H / C) + np.log(L / O) * np.log(L / C)

    var_o = np.var(r_o[sl], ddof=1)
    var_c = np.var(r_c[sl], ddof=1)
    var_rs = np.mean(rs[sl])
    k = 0.34 / (1.34 + (w + 1) / (w - 1))
    expected = np.sqrt(var_o + k * var_c + (1 - k) * var_rs)
    assert yz.iloc[t] == pytest.approx(expected, rel=1e-12)


def test_yang_zhang_off_by_one_causality():
    """A t-napi YZ a t-napi OHLC-t HASZNÁLJA (záráskori becslő), a t+1 napit NEM."""
    df = _ohlc_frame()
    yz_base = yang_zhang_sigma(df, "NVDA")

    t = 40
    df_future = df.copy()
    df_future.iloc[t + 1, :] = df_future.iloc[t + 1, :] * 1.5   # jövő elrontása
    yz_future = yang_zhang_sigma(df_future, "NVDA")
    assert yz_future.iloc[t] == yz_base.iloc[t], "A t-napi YZ a jövőtől függ!"

    df_today = df.copy()
    df_today.loc[df.index[t], "NVDA_High"] *= 1.2               # ma elrontása
    yz_today = yang_zhang_sigma(df_today, "NVDA")
    assert yz_today.iloc[t] != yz_base.iloc[t], \
        "A t-napi YZ-nek a t-napi range-et használnia kell"


def test_yang_zhang_recovers_true_vol():
    """Konstans-vol GBM-en a YZ a valódi σ közelébe becsül (szendvics-teszt)."""
    rng = np.random.default_rng(7)
    n, true_sigma = 2000, 0.02
    r = rng.normal(0, true_sigma, n)
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2018-01-01", periods=n)
    df = pd.DataFrame({
        "NVDA_Open": np.concatenate([[100], close[:-1]]),
        "NVDA_Close": close,
    }, index=idx)
    df["NVDA_High"] = np.maximum(df["NVDA_Open"], df["NVDA_Close"]) * np.exp(true_sigma * 0.4)
    df["NVDA_Low"] = np.minimum(df["NVDA_Open"], df["NVDA_Close"]) * np.exp(-true_sigma * 0.4)
    yz = yang_zhang_sigma(df, "NVDA").dropna()
    assert yz.mean() == pytest.approx(true_sigma, rel=0.35)


def _sigma_series(n=500, seed=1):
    rng = np.random.default_rng(seed)
    log_sig = -3.9 + np.cumsum(rng.normal(0, 0.05, n)) * 0.1
    return pd.Series(np.exp(log_sig), index=pd.bdate_range("2020-01-01", periods=n))


def test_har_train_only_fit():
    """A train utáni σ-értékek elrontása nem változtatja sem az
    együtthatókat, sem a train-szakaszbeli előrejelzést."""
    sig = _sigma_series()
    train_end = sig.index[350]
    base = har_rv_forecast(sig, train_end)

    mutated = sig.copy()
    mutated.iloc[360:] = mutated.iloc[360:] * 5.0
    mut = har_rv_forecast(mutated, train_end)

    np.testing.assert_array_equal(base.iloc[:355].to_numpy(),
                                  mut.iloc[:355].to_numpy())


def test_har_prefix_invariance():
    """Kauzalitás: a t-ig tartó prefixen számolt előrejelzés bitre azonos a
    teljes soron számolt t-edikkel (a fit train-only, a prefix ⊇ train)."""
    sig = _sigma_series()
    train_end = sig.index[300]
    full = har_rv_forecast(sig, train_end)
    for t in (320, 400, 499):
        prefix = har_rv_forecast(sig.iloc[:t + 1], train_end)
        assert prefix.iloc[-1] == full.iloc[t], f"t={t}: a HAR nem kauzális!"


def test_har_tracks_volatility_level():
    """Sanity: a HAR-előrejelzés korrelál a következő napi σ-val."""
    sig = _sigma_series(n=800, seed=3)
    har = har_rv_forecast(sig, sig.index[500])
    realized_next = sig.to_numpy()[1:]
    pred = har.to_numpy()[:-1]
    corr = np.corrcoef(pred[550:], realized_next[550:])[0, 1]
    assert corr > 0.8
