"""Valószínűség-kalibráció: Platt-skálázás az izotonikus regresszió helyett.

Az izotonikus regresszió ~1000 minta alatt túltanul (Niculescu-Mizil &
Caruana 2005) — a V5 369 val-pontja messze ez alatt van. A Platt-skálázás
(logisztikus regresszió a logit-transzformált nyers valószínűségen) 2
paraméteres, kis mintán is stabil.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """p_cal = sigmoid(a·logit(p_raw) + b), a val-halmazon illesztve."""

    def __init__(self):
        self._lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(_logit(p_raw)[:, None], np.asarray(y, dtype=int))
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(_logit(p_raw)[:, None])[:, 1]


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10,
                      min_count: int = 5):
    """Megbízhatósági diagram pontjai: (átlagos jósolt p, valós gyakoriság, darab)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        mask = ids == b
        if mask.sum() >= min_count:
            xs.append(float(p[mask].mean()))
            ys.append(float(y[mask].mean()))
            ns.append(int(mask.sum()))
    return np.array(xs), np.array(ys), np.array(ns)
