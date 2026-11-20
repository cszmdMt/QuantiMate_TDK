"""Magyarázhatóság (V7-17): purged permutációs fontosság + TreeSHAP.

A naiv feature importance idősoron félrevezet (átfedő címkék, korrelált
jellemzők) — az AFML 8. fejezete purged CV-be ágyazott MDA-t ír elő, korrelált
jellemzőkre klaszterezett fontosságot (López de Prado 2018/2020). A pinball-
loss-ra permutálunk (nem accuracy-re — konzisztens a rangsorolási elvvel).
A GBDT-ághoz egzakt TreeSHAP (a LightGBM beépített pred_contrib-ja).

Ez RIPORT-réteg: a fontosság-számítás val-adaton fut, a teszt-fegyelmet nem
érinti.
"""
from __future__ import annotations

import numpy as np

from quantimate.models.models import pinball_loss_np


def permutation_importance_pinball(predict_fn, X_tab: np.ndarray, Y: np.ndarray,
                                   quantiles, feature_names=None, *,
                                   n_repeats: int = 5, seed: int = 42) -> list[dict]:
    """MDA a pinball-loss-on: minden jellemző oszlopát permutálva mérjük a
    val-pinball romlását. Determinisztikus (seedelt).

    predict_fn(X) → (B, H, Q) kvantilis-predikció. Nagyobb ΔLoss = fontosabb.
    """
    rng = np.random.default_rng(seed)
    X_tab = np.asarray(X_tab, dtype=float)
    n_feat = X_tab.shape[1]
    names = feature_names or [f"f{j}" for j in range(n_feat)]

    base = pinball_loss_np(Y, predict_fn(X_tab), quantiles=quantiles)
    rows = []
    for j in range(n_feat):
        deltas = []
        for _ in range(n_repeats):
            Xp = X_tab.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(pinball_loss_np(Y, predict_fn(Xp), quantiles=quantiles) - base)
        rows.append({"jellemző": names[j], "MDA": float(np.mean(deltas)),
                     "MDA_szórás": float(np.std(deltas))})
    rows.sort(key=lambda r: r["MDA"], reverse=True)
    return rows


def clustered_importance_pinball(predict_fn, X_tab, Y, quantiles, clusters,
                                 feature_names=None, *, n_repeats: int = 5,
                                 seed: int = 42) -> list[dict]:
    """Csoport-szintű MDA: korrelált jellemzőket EGYÜTT permutálunk (López de
    Prado clustered MDA) — így a korreláció nem hígítja szét a fontosságot.
    `clusters`: {klaszter_név: [oszlop-indexek]}."""
    rng = np.random.default_rng(seed)
    X_tab = np.asarray(X_tab, dtype=float)
    base = pinball_loss_np(Y, predict_fn(X_tab), quantiles=quantiles)
    rows = []
    for cname, cols in clusters.items():
        deltas = []
        for _ in range(n_repeats):
            Xp = X_tab.copy()
            perm = rng.permutation(len(Xp))
            for j in cols:
                Xp[:, j] = Xp[perm, j]          # KÖZÖS permutáció → korreláció megtartva
            deltas.append(pinball_loss_np(Y, predict_fn(Xp), quantiles=quantiles) - base)
        rows.append({"klaszter": cname, "MDA": float(np.mean(deltas)),
                     "MDA_szórás": float(np.std(deltas))})
    rows.sort(key=lambda r: r["MDA"], reverse=True)
    return rows


def correlation_clusters(X_tab: np.ndarray, feature_names,
                         threshold: float = 0.7) -> dict:
    """Egyszerű korreláció-alapú klaszterezés: |ρ| ≥ threshold jellemzők egy
    csoportba (union-find). Determinisztikus."""
    X = np.asarray(X_tab, dtype=float)
    n = X.shape[1]
    corr = np.corrcoef(X, rowvar=False)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= threshold:
                parent[find(i)] = find(j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return {f"cl{k}_" + "+".join(feature_names[c] for c in cols[:2])
            + ("…" if len(cols) > 2 else ""): cols
            for k, cols in enumerate(groups.values())}
