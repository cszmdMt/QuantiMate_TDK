"""Rezsim-modell: HMM forward-SZŰRT valószínűségekkel.

A hmmlearn beépített poszterior-lekérdezése a forward–backward SIMÍTOTT
poszteriort adja (P(s_t | O_1..O_T)) — a t-napi rezsim-jellemző a jövőt is
látná. Itt a
kauzális forward-szűrést implementáljuk: P(s_t | O_1..O_t). Privát API-t
(_compute_log_likelihood) nem használunk — az verziófüggő (hibakatalógus 1.).

Címkézés (becsületesen): a 3 állapot statisztikai klaszter, nem szó szerinti
bika/medve. A legmagasabb volatilitású állapot "Turbulens", a maradék kettő
átlaghozam szerint "Csökkenő" / "Emelkedő tendencia".
"""
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def hmm_filtered_proba(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """P(s_t | O_1..O_t) — kauzális szűrt poszterior, alakja (T, K).

    Emissziós log-likelihood scipy-jal (covariance_type="full" → covars_
    alakja (K, D, D), pont ezt várja a multivariate_normal).
    """
    X = np.asarray(X, dtype=float)
    K = model.n_components
    log_emis = np.column_stack([
        multivariate_normal.logpdf(X, model.means_[k], model.covars_[k],
                                   allow_singular=True)
        for k in range(K)])                                  # (T, K)
    log_A = np.log(model.transmat_ + 1e-300)
    log_alpha = np.empty_like(log_emis)
    log_alpha[0] = np.log(model.startprob_ + 1e-300) + log_emis[0]
    for t in range(1, len(X)):
        log_alpha[t] = log_emis[t] + logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
    return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))


def fit_hmm(X_train: np.ndarray, n_components: int = 3, seed: int = 42,
            n_iter: int = 300) -> GaussianHMM:
    """HMM-illesztés KIZÁRÓLAG a tanító szakaszon."""
    hmm = GaussianHMM(n_components=n_components, covariance_type="full",
                      n_iter=n_iter, random_state=seed)
    hmm.fit(np.asarray(X_train, dtype=float))
    return hmm


def fit_hmm_scaled(X_train: np.ndarray, n_components: int = 3, seed: int = 42,
                   n_iter: int = 300) -> tuple[GaussianHMM, np.ndarray, np.ndarray]:
    """HMM-illesztés a tanító szakasz SAJÁT átlagával/szórásával standardizált
    megfigyeléseken. Vissza: (modell, center, scale) — a szűréshez a hívó
    ugyanezt a (center, scale) párt alkalmazza: hmm_filtered_proba(m, (X-c)/s).

    Miért: a hmmlearn induló kovarianciája cov(X) + min_covar·I (min_covar=1e-3).
    Napi hozam-skálán (szórás ~0,02 és ~0,007) ez a padló a valódi varianciák
    2–50-szerese: az EM rosszul kondicionált, a log-likelihood nem monoton, az
    állapotok összeolvadnak, és egyes ablakokon a kovariancia szingulárissá
    válik ("covars must be symmetric, positive-definite"). Egységnyi szóráson a
    padló ártalmatlan regularizáció. A center/scale kizárólag X_train-ből jön,
    így a szűrt poszterior prefix-invariáns marad.

    Tartalékág: ha a teljes kovarianciájú illesztés hibázik vagy nem véges,
    diagonális kovarianciával próbálkozunk; ha az is elbukik, hiba (nincs
    csendes helyettesítés). A `fit_hmm` változatlan — a V6 golden arra épül.
    """
    X = np.asarray(X_train, dtype=float)
    center, scale = X.mean(axis=0), X.std(axis=0)
    # Relatív küszöb: egy konstans oszlop szórása lebegőpontosan ~1e-18, nem pontosan 0.
    floor = 1e-12 * np.maximum(1.0, np.abs(center))
    if not (np.isfinite(center).all() and np.isfinite(scale).all() and (scale > floor).all()):
        raise ValueError("HMM: nem véges vagy konstans megfigyelés-oszlop")
    Z = (X - center) / scale
    errors = []
    for covariance_type in ("full", "diag"):
        hmm = GaussianHMM(n_components=n_components, covariance_type=covariance_type,
                          n_iter=n_iter, random_state=seed)
        try:
            hmm.fit(Z)
            healthy = all(np.isfinite(a).all() for a in
                          (hmm.means_, hmm.covars_, hmm.transmat_, hmm.startprob_))
        except ValueError as exc:
            errors.append(f"{covariance_type}: {exc}")
            continue
        if healthy:
            return hmm, center, scale
        errors.append(f"{covariance_type}: nem véges paraméterek")
    raise ValueError("HMM-illesztés sikertelen standardizált megfigyeléseken — "
                     + "; ".join(errors))


def label_regimes(model: GaussianHMM) -> tuple[np.ndarray, list[str]]:
    """Állapot-sorrend és becsületes címkék hozam ÉS vol alapján.

    Visszaad: (order, names) — az `order` az állapot-indexek átrendezése úgy,
    hogy names[i] a proba[:, order[i]] oszlophoz tartozik. A legmagasabb
    vol-ú állapot "Turbulens"; a maradék kettő hozam szerint rendezve.
    """
    means = model.means_[:, 0]
    vols = np.sqrt(np.array([model.covars_[k][0, 0]
                             for k in range(model.n_components)]))
    turbulent = int(np.argmax(vols))
    rest = [k for k in range(model.n_components) if k != turbulent]
    rest_sorted = sorted(rest, key=lambda k: means[k])
    order = np.array(rest_sorted + [turbulent])
    names = (["Csökkenő tendencia", "Emelkedő tendencia"][:len(rest_sorted)]
             + ["Turbulens"])
    return order, names
