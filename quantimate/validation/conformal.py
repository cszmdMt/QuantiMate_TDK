"""Konformális predikció (B2): CQR + ACI — a 80–85% cél operacionalizálása.

CQR (Romano et al. 2019): a val-halmaz konformitás-pontszámaiból
(E_i = max(q_lo−y, y−q_hi)) számolt korrekció szimmetrikusan tágítja/szűkíti
a kvantilis-sávot úgy, hogy a névleges lefedettség véges mintán garantált.

ACI (Gibbs–Candès 2021): a teszten online frissítjük az α_t szintet
(α_{t+1} = α_t + γ(α_cél − err_t)) — kizárólag MÚLTBELI hibákból.
γ=0.01; α_t clipelve az oszcilláció ellen.
"""
from __future__ import annotations

import numpy as np


def conformity_scores(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    """CQR konformitás-pontszám: E_i = max(q_lo−y_i, y_i−q_hi).
    Pozitív, ha a pont a sávon kívül van; negatív, ha bent."""
    y = np.asarray(y, dtype=float)
    return np.maximum(np.asarray(q_lo) - y, y - np.asarray(q_hi))


def cqr_delta(scores: np.ndarray, alpha: float) -> float:
    """Véges mintás korrekció: a pontszámok ⌈(n+1)(1−α)⌉/n kvantilise."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(scores[k - 1])


def apply_band(q_lo: np.ndarray, q_hi: np.ndarray, delta: float):
    """Sáv-korrekció: [q_lo − δ, q_hi + δ]."""
    return np.asarray(q_lo) - delta, np.asarray(q_hi) + delta


def aci_bands(y_test: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray,
              cal_scores: np.ndarray, alpha_target: float,
              gamma: float = 0.01, alpha_clip: tuple = (0.01, 0.3)):
    """Adaptive Conformal Inference a teszt-szakaszon.

    Minden t-re: δ_t a (val-pontszámok + t ELŐTTI teszt-pontszámok) alapján az
    aktuális α_t szinten; utána err_t-ből α frissül. A t-napi sáv tehát csak
    múltbeli információból épül.

    Visszaad: (lo, hi, alphas, errs) tömbök.
    """
    y_test = np.asarray(y_test, dtype=float)
    scores = list(np.asarray(cal_scores, dtype=float))
    alpha_t = alpha_target
    lo_out = np.empty(len(y_test))
    hi_out = np.empty(len(y_test))
    alphas = np.empty(len(y_test))
    errs = np.empty(len(y_test))

    for t in range(len(y_test)):
        alphas[t] = alpha_t
        delta = cqr_delta(np.asarray(scores), alpha_t)
        lo_out[t] = q_lo[t] - delta
        hi_out[t] = q_hi[t] + delta
        err = float(not (lo_out[t] <= y_test[t] <= hi_out[t]))
        errs[t] = err
        alpha_t = float(np.clip(alpha_t + gamma * (alpha_target - err),
                                alpha_clip[0], alpha_clip[1]))
        scores.append(float(conformity_scores(y_test[t:t + 1],
                                              q_lo[t:t + 1], q_hi[t:t + 1])[0]))
    return lo_out, hi_out, alphas, errs


def empirical_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(((np.asarray(lo) <= y) & (y <= np.asarray(hi))).mean())


# ── V7-9: Rezsim-kondicionált (Mondrian) lefedettség ─────────────────────────
# A CQR/ACI csak MARGINÁLIS garanciát ad: átlagosan 85%, de lehet, hogy nyugodt
# rezsimben 92% és turbulensben 70% — pont ott bukik, ahol a legtöbbet érne.
# A Mondrian-eljárás (Boström–Johansson 2020) kategóriánként kalibrál →
# kategória-feltételes lefedettség. Kis kategória-mintánál (n < min_n) csak
# diagnosztika készül, feltételes kalibráció nélkül.

def coverage_by_category(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                         categories: np.ndarray,
                         category_names: dict | None = None) -> list[dict]:
    """Lefedettség × kategória diagnosztika Wilson 95% CI-vel."""
    from statsmodels.stats.proportion import proportion_confint

    y = np.asarray(y, dtype=float)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    cats = np.asarray(categories)
    rows = []
    for c in np.unique(cats):
        m = cats == c
        n = int(m.sum())
        inside = int(((lo[m] <= y[m]) & (y[m] <= hi[m])).sum())
        ci_lo, ci_hi = proportion_confint(inside, n, alpha=0.05, method="wilson")
        rows.append({
            "kategória": (category_names or {}).get(c, str(c)),
            "n": n, "empirikus": inside / n if n else float("nan"),
            "Wilson alsó": float(ci_lo), "Wilson felső": float(ci_hi),
        })
    return rows


def mondrian_cqr_delta(scores: np.ndarray, categories: np.ndarray,
                       alpha: float, min_n: int = 50) -> dict:
    """Kategóriánkénti CQR-korrekció; n < min_n kategóriánál a GLOBÁLIS
    korrekcióra esik vissza, a visszaesés jelölve.

    Visszaad: {kategória: {"delta", "n", "fallback"}} + {"__global__": δ_global}.
    """
    scores = np.asarray(scores, dtype=float)
    cats = np.asarray(categories)
    global_delta = cqr_delta(scores, alpha)
    out = {"__global__": global_delta}
    for c in np.unique(cats):
        m = cats == c
        n = int(m.sum())
        if n >= min_n:
            out[c] = {"delta": cqr_delta(scores[m], alpha), "n": n,
                      "fallback": False}
        else:
            out[c] = {"delta": global_delta, "n": n, "fallback": True}
    return out


def apply_mondrian_band(q_lo: np.ndarray, q_hi: np.ndarray,
                        categories: np.ndarray, deltas: dict):
    """Kategória-függő sáv-korrekció; ismeretlen kategória → globális δ."""
    q_lo, q_hi = np.asarray(q_lo, dtype=float), np.asarray(q_hi, dtype=float)
    cats = np.asarray(categories)
    d = np.array([deltas.get(c, {"delta": deltas["__global__"]})["delta"]
                  if isinstance(deltas.get(c), dict)
                  else deltas["__global__"] for c in cats])
    return q_lo - d, q_hi + d


# ── V7-10: Nem-cserélhető konformális — NexCP-súlyozás + AgACI ────────────────
# Az ACI két gyengéje: a γ=0.01 kézi választás, és a kalibrációs pontok egyenlő
# súlya (a 2020-as pontok ma kevésbé relevánsak). A NexCP (Barber et al. 2023)
# időben lecsengő súllyal kvantilizál; az AgACI (Zaffran et al. 2022) a γ-t
# online expert-aggregációval tanulja.

def weighted_quantile(values: np.ndarray, weights: np.ndarray,
                      q: float) -> float:
    """Súlyozott kvantilis (a NexCP korrekció alapja).

    Nulla-összegű súlynál (elvben nem fordulhat elő, mert a NexCP-súlyok
    ρ^k > 0) egyenletes súlyra esünk vissza, hogy ne osszunk nullával."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        weights = np.ones_like(weights)
        total = weights.sum()
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) / total
    idx = int(np.searchsorted(cw, q))
    return float(v[min(idx, len(v) - 1)])


def nexcp_delta(scores: np.ndarray, alpha: float, rho: float = 0.99) -> float:
    """NexCP korrekció: a konformitás-pontszámok ρ^(n-1-i) súlyú (1-α)
    kvantilise + a "végtelen-pont" (a jelen) súlya (Barber et al. 2023).
    ρ=1 → visszaadja a standard (súlyozatlan) kvantilist."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    w = rho ** np.arange(n - 1, -1, -1)       # legrégebbi kicsi, legújabb ~1
    # A tesztpont (végtelen score) súlya = ρ^0 = 1; a kvantilis-szintet a
    # normált súlyokkal a (súly + 1) tömegen számoljuk.
    w_ext = np.append(w, 1.0)
    v_ext = np.append(scores, np.inf)
    return weighted_quantile(v_ext, w_ext, 1 - alpha)


def agaci_bands(y_test, q_lo, q_hi, cal_scores, alpha_target,
                gamma_grid=(0.005, 0.01, 0.02, 0.05, 0.1), eta: float = 10.0,
                alpha_clip=(0.01, 0.3)):
    """AgACI: több γ-szakértő ACI-ja, exponenciális súlyozású online
    aggregációval (a pinball-veszteségük alapján). Paraméter-mentes a γ-ban.

    Visszaad: (lo, hi, alphas_eff, errs).
    """
    y = np.asarray(y_test, dtype=float)
    q_lo, q_hi = np.asarray(q_lo, dtype=float), np.asarray(q_hi, dtype=float)
    G = len(gamma_grid)
    experts_alpha = np.full(G, alpha_target)
    scores_hist = [list(np.asarray(cal_scores, dtype=float)) for _ in range(G)]
    log_w = np.zeros(G)

    lo_out = np.empty(len(y))
    hi_out = np.empty(len(y))
    alphas_eff = np.empty(len(y))
    errs = np.empty(len(y))

    for t in range(len(y)):
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        # Szakértőnkénti δ és sáv
        deltas = np.array([cqr_delta(np.asarray(scores_hist[g]),
                                     experts_alpha[g]) for g in range(G)])
        alpha_eff = float(np.clip(np.dot(w, experts_alpha),
                                  alpha_clip[0], alpha_clip[1]))
        alphas_eff[t] = alpha_eff
        delta_agg = float(np.dot(w, deltas))
        lo_out[t], hi_out[t] = q_lo[t] - delta_agg, q_hi[t] + delta_agg
        covered = lo_out[t] <= y[t] <= hi_out[t]
        errs[t] = float(not covered)

        # Szakértők frissítése (ACI) + súlyozás a pinball-veszteségükkel
        sc = conformity_scores(y[t:t + 1], q_lo[t:t + 1], q_hi[t:t + 1])[0]
        for g, gm in enumerate(gamma_grid):
            d_g = cqr_delta(np.asarray(scores_hist[g]), experts_alpha[g])
            err_g = float(not (q_lo[t] - d_g <= y[t] <= q_hi[t] + d_g))
            experts_alpha[g] = float(np.clip(
                experts_alpha[g] + gm * (alpha_target - err_g),
                alpha_clip[0], alpha_clip[1]))
            # interval-score (Gneiting–Raftery) mint szakértő-veszteség
            width = (q_hi[t] + d_g) - (q_lo[t] - d_g)
            penalty = (2 / alpha_target) * (
                max(0, (q_lo[t] - d_g) - y[t]) + max(0, y[t] - (q_hi[t] + d_g)))
            log_w[g] -= eta * (width + penalty) / (abs(y[t]) + 1.0)
            scores_hist[g].append(sc)
    return lo_out, hi_out, alphas_eff, errs


# ── V7-11: Több-horizontú EGYÜTTES lefedettség — CopulaCPTS ───────────────────
# A h∈{1,5,21} sávok külön kalibráltak: "a teljes pálya 85% eséllyel a sávban
# marad" nem tehető meg (a horizontok hibái összefüggnek). A CopulaCPTS
# (Sun–Yu 2024) a horizontok közti hibafüggőséget empirikus kopulával modellezi
# és véges mintás EGYÜTTES konfidencia-régiót ad. Olcsó referencia: Bonferroni.

def _pit_ranks(cal_scores: np.ndarray) -> np.ndarray:
    """Probability integral transform: minden kalibrációs pont rangja [0,1]-ben,
    horizontonként (a kopula-illesztés bemenete). cal_scores: (n_cal, H)."""
    n = len(cal_scores)
    ranks = np.argsort(np.argsort(cal_scores, axis=0), axis=0) + 1
    return ranks / (n + 1)


def copula_joint_level(cal_scores_by_h: np.ndarray, target_joint: float) -> float:
    """A közös (per-horizont azonos) marginális szint u*, amelynél az empirikus
    kopula szerint P(minden horizont a sávban) = target_joint.

    cal_scores_by_h: (n_cal, H) — a kalibrációs konformitás-pontszámok
    horizontonként (kisebb = jobban a sávban). Visszaad: u* ∈ (0,1) marginális
    kvantilis-szint, amit MINDEN horizonton alkalmazva az együttes lefedettség
    ≈ target_joint. Az empirikus kopulát a rang-transzformált pontokból becsüljük.
    """
    U = _pit_ranks(np.asarray(cal_scores_by_h, dtype=float))   # (n, H)
    # keressük a legkisebb u-t, amelynél a pontok target_joint hányada
    # MINDEN dimenzióban ≤ u (az együttes empirikus CDF a főátlón)
    grid = np.linspace(0.5, 0.999, 400)
    for u in grid:
        joint_cov = float(np.all(U <= u, axis=1).mean())
        if joint_cov >= target_joint:
            return float(u)
    return float(grid[-1])


def bonferroni_level(target_joint: float, n_horizons: int) -> float:
    """Bonferroni-referencia: minden horizontot 1−(1−target)/H szintre kalibrálva
    az együttes lefedettség ≥ target (konzervatív)."""
    return 1 - (1 - target_joint) / n_horizons


def joint_coverage(scores_test_by_h: np.ndarray, thresholds_by_h: np.ndarray) -> float:
    """Együttes empirikus lefedettség: a tesztpontok hányada, ahol MINDEN
    horizont pontszáma ≤ a saját küszöbe (azaz mind a sávban)."""
    s = np.asarray(scores_test_by_h, dtype=float)
    thr = np.asarray(thresholds_by_h, dtype=float)
    return float(np.all(s <= thr[None, :], axis=1).mean())


# ── Késleltetett visszacsatolású ACI (több napos horizont) ──────────────────

def aci_bands_delayed(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray,
                      positions: np.ndarray, n_cal: int, alpha_target: float,
                      horizon: int, gamma: float = 0.01):
    """ACI (Gibbs–Candès 2021) olyan előrejelzés-folyamra, ahol a t-ben kiadott
    sáv kimenetele csak `horizon` kereskedési nappal később ismert.

    Az aci_bands azonnali visszacsatolást feltételez (h=1-re helyes). h>1-nél a
    t origóban még NEM ismert a t−h+1 … t−1 origók kimenetele: ha ezeket is
    felhasználnánk, a sáv jövőbeli árakat látna. Itt egy i sorszámú pontszám
    és hiba akkor használható t-ben, ha positions[i] + horizon ≤ positions[t].

    y, q_lo, q_hi: a teljes folyam (kalibráció + értékelés), időrendben;
    positions: a sorok kereskedésinap-sorszáma (szigorúan növő egészek);
    az első n_cal sor csak kalibrál (sávot nem kap, α-t nem frissít).

    Visszaad: (lo, hi, alphas) — NaN az első n_cal soron.
    α ≤ 0 esetén a korrekció a legnagyobb ismert pontszám (nem végtelen sáv),
    α ≥ 1 esetén a legkisebb.
    """
    y = np.asarray(y, dtype=float)
    q_lo, q_hi = np.asarray(q_lo, dtype=float), np.asarray(q_hi, dtype=float)
    positions = np.asarray(positions)
    if len(positions) > 1 and (np.diff(positions) <= 0).any():
        raise ValueError("positions legyen szigorúan növő")
    if not 0 < n_cal < len(y):
        raise ValueError("n_cal a folyam belsejébe essen")
    scores = conformity_scores(y, q_lo, q_hi)
    lo = np.full(len(y), np.nan)
    hi = np.full(len(y), np.nan)
    alphas = np.full(len(y), np.nan)
    errs = np.zeros(len(y))
    alpha_t, known, pool = float(alpha_target), 0, []
    for t in range(n_cal, len(y)):
        # Minden, a t origóig megérett kimenetel beépítése — időrendben.
        while known < t and positions[known] + horizon <= positions[t]:
            pool.append(float(scores[known]))
            if known >= n_cal:
                alpha_t += gamma * (alpha_target - errs[known])
            known += 1
        if not pool:
            raise ValueError("Nincs megérett kalibrációs pontszám az első értékelt sorhoz")
        ordered = np.sort(pool)
        if alpha_t <= 0:
            delta = float(ordered[-1])
        elif alpha_t >= 1:
            delta = float(ordered[0])
        else:
            delta = cqr_delta(ordered, alpha_t)
        alphas[t] = alpha_t
        lo[t], hi[t] = q_lo[t] - delta, q_hi[t] + delta
        errs[t] = float(not (lo[t] <= y[t] <= hi[t]))
    return lo, hi, alphas
