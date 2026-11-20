"""V7-4 tesztek: piaczárás-igazítás, Bayes-k kalibráció, FinBERT-guard.
A hálózati/modell-függő részek külön (opcionális) jelöléssel."""
import numpy as np
import pandas as pd
import pytest

from quantimate.features.news_history import (_market_adjusted_trading_date,
                                     calibrate_bayes_k, finbert_available)


def test_market_close_adjustment():
    """A 16:00 ET UTÁNI hír a KÖVETKEZŐ napra esik; előtte az aznapira."""
    ts = pd.Series(pd.to_datetime([
        "2024-03-15 13:00:00",   # 09:00 ET (nyitás előtt/alatt) → aznap
        "2024-03-15 21:30:00",   # 17:30 ET (zárás után) → másnap
        "2024-03-15 19:59:00",   # 15:59 ET (zárás előtt) → aznap
    ], utc=True))
    d = _market_adjusted_trading_date(ts)
    assert d.iloc[0] == pd.Timestamp("2024-03-15")
    assert d.iloc[1] == pd.Timestamp("2024-03-16")
    assert d.iloc[2] == pd.Timestamp("2024-03-15")


def test_bayes_k_calibration_recovers_sign():
    """Ha a pozitív szentiment tényleg emelkedést jósol, a kalibrált k pozitív."""
    rng = np.random.default_rng(0)
    s = rng.normal(0, 0.3, 500)
    y = (s + rng.normal(0, 0.2, 500) > 0).astype(int)   # szentiment → irány
    res = calibrate_bayes_k(pd.Series(s), pd.Series(y))
    assert res["k"] > 0
    assert res["train_acc"] > 0.6


def test_bayes_k_zero_when_no_relation():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 0.3, 500)
    y = rng.integers(0, 2, 500)                          # független
    res = calibrate_bayes_k(pd.Series(s), pd.Series(y))
    assert abs(res["k"]) < 1.0                           # nincs erős kapcsolat


@pytest.mark.skipif(finbert_available(),
                    reason="torch/transformers elérhető — a guard-ág nem tesztelhető")
def test_finbert_guard_when_unavailable():
    from quantimate.features.news_history import finbert_daily_sentiment
    with pytest.raises(RuntimeError, match="nem elérhető"):
        finbert_daily_sentiment(pd.DataFrame({"Article_title": ["x"],
                                              "Date": [pd.Timestamp.now(tz="UTC")]}))


@pytest.mark.skipif(not finbert_available(),
                    reason="torch/transformers nem elérhető")
def test_finbert_scores_are_bounded():
    """FinBERT elérhető: a pontszám ∈ [−1, 1] (P(pos)−P(neg))."""
    from quantimate.features.news_history import finbert_daily_sentiment
    news = pd.DataFrame({
        "Article_title": ["Nvidia beats earnings, stock surges",
                          "Nvidia plunges on weak guidance",
                          "Market flat ahead of data"],
        "Date": pd.to_datetime(["2024-01-02 12:00", "2024-01-02 12:00",
                                "2024-01-03 12:00"], utc=True)})
    daily = finbert_daily_sentiment(news)
    assert (daily["senti_mean"].abs() <= 1.0).all()
    assert len(daily) == 2
