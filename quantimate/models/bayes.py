"""Bayes-réteg fegyelmezése: jel-clip, LR-plafon, risk-clip.

A réteg ILLUSZTRATÍV: a k paraméter historikus hír-adaton nincs kalibrálva,
ezért a backtest Gemini-MENTES valószínűségeken fut; a Bayes-frissítés csak
az élő döntés-demonstrációban jelenik meg.
"""
from __future__ import annotations

import math

import numpy as np


def news_signal(analysis: dict) -> float:
    """Hír-jel a Gemini-pontszámokból; a geopolitical_risk szélsőértékei
    (0.0/1.0) [0.05, 0.95] közé vágva (A8d)."""
    risk = float(np.clip(analysis.get("geopolitical_risk", 0.5), 0.05, 0.95))
    conf = float(analysis.get("confidence", 0.0))
    return (analysis.get("sentiment", 0.0) * conf
            + 0.5 * analysis.get("fundamental_score", 0.0) * conf
            - 0.4 * (risk - 0.5) * max(conf, 0.1))


def bayes_update(prior_p: float, analysis: dict, k: float = 0.6,
                 lr_cap: float = 1.5, s_clip: float = 3.0) -> float:
    """Odds-formájú Bayes-frissítés plafonnal.

    s clipelve [−3, 3] közé az exp-overflow ellen (hibakatalógus 16.);
    LR = clip(exp(k·s), 1/lr_cap, lr_cap) — egyetlen nap hírei legfeljebb
    1.5× szorzót kaphatnak (A8a).
    """
    s = float(np.clip(news_signal(analysis), -s_clip, s_clip))
    lr = float(np.clip(math.exp(k * s), 1.0 / lr_cap, lr_cap))
    prior_p = float(np.clip(prior_p, 1e-6, 1 - 1e-6))
    post_odds = prior_p / (1 - prior_p) * lr
    return post_odds / (1 + post_odds)
