"""Kvantilis-ensemble (V7-7): győztes-választás helyett kombináció.

A „forecast combination puzzle" (Timmermann 2006): az egyszerű átlagolás
tipikusan veri az egyedi modelleket. Kvantilis-előrejelzésnél a helyes művelet
a **kvantilis-átlagolás** (Vincentizálás; Lichtendahl et al. 2013): élesebb
eredő eloszlást ad, mint a valószínűség-átlagolás. Dinamikus súlyozásként a
CRPS-alapú exponenciális súlyozás az online CRPS-learning (Berrisch–Ziel 2023)
egyszerűsített, dokumentált változata.

A tag-választás (k) és a súly-hőmérséklet (η) KIZÁRÓLAG val-adatból;
a CQR/ACI változatlanul az ensemble-kimenetre ül — a konformális garancia
modell-agnosztikus.
"""
from __future__ import annotations

import numpy as np

from quantimate.models.models import pinball_loss_np


def vincentile_average(q_arrays, weights=None) -> np.ndarray:
    """(B,H,Q) kvantilis-tömbök súlyozott, kvantilisenkénti átlaga,
    a Q-tengelyen újrarendezve."""
    q = np.stack([np.asarray(a, dtype=float) for a in q_arrays])   # (M,B,H,Q)
    if weights is None:
        weights = np.full(len(q), 1.0 / len(q))
    w = np.asarray(weights, dtype=float)
    assert len(w) == len(q) and np.isclose(w.sum(), 1.0)
    out = np.tensordot(w, q, axes=1)                               # (B,H,Q)
    return np.sort(out, axis=-1)


def crps_based_weights(crps_values, eta: float) -> np.ndarray:
    """w_m ∝ exp(−η·(CRPS_m − min)/scale) — η=0 → egyenletes átlag."""
    c = np.asarray(crps_values, dtype=float)
    scale = c.std() if c.std() > 0 else 1.0
    w = np.exp(-eta * (c - c.min()) / scale)
    return w / w.sum()


def choose_ensemble(member_q_va: dict, Y_va: np.ndarray, quantiles,
                    k_grid=(2, 3, 5), eta_grid=(0.0, 1.0, 3.0)) -> dict:
    """Tagok, k és η kiválasztása a VAL-on (minden kombináció olcsó — nincs
    újratanítás). Determinisztikus: a tagok egyedi val-pinball szerint
    rangsorolódnak, holtversenynél kulcs-név szerint.

    Visszaad: {"members", "weights", "k", "eta", "val_pinball", "grid"}.
    """
    keys = sorted(member_q_va,
                  key=lambda k: (pinball_loss_np(Y_va, member_q_va[k],
                                                 quantiles=quantiles), k))
    member_crps = {k: 2.0 * pinball_loss_np(Y_va, member_q_va[k],
                                            quantiles=quantiles) for k in keys}

    best = None
    grid_rows = []
    for k in k_grid:
        if k > len(keys):
            continue
        top = keys[:k]
        for eta in eta_grid:
            w = crps_based_weights([member_crps[m] for m in top], eta)
            q_ens = vincentile_average([member_q_va[m] for m in top], w)
            pin = float(pinball_loss_np(Y_va, q_ens, quantiles=quantiles))
            grid_rows.append({"k": k, "eta": eta, "val_pinball": pin})
            if best is None or pin < best["val_pinball"]:
                best = {"members": list(top), "weights": w, "k": k,
                        "eta": float(eta), "val_pinball": pin}
    best["grid"] = grid_rows
    return best
