"""Triple-barrier címkézés + meta-labeling (V7-12, López de Prado AFML 3. fej.).

A belépési döntés a fő láncban egyetlen kalibrált küszöb. A **meta-labeling**
külön ML-réteget tesz az elsődleges jel fölé: az elsődleges modell adja az
IRÁNYT, a másodlagos modell azt tanulja, *mikor higgyünk neki* (és ezzel szűr +
méretez). A címkézés a **triple-barrier** módszerrel történik: profit-take /
stop-loss / időkorlát hármas, σ_t-hez skálázott gátakkal — út-függő,
kereskedésileg értelmes kimenet.

**Szivárgás:** a triple-barrier a gát-érintés JÖVŐBELI idejéig néz előre
→ a purge a maximális tartási időhöz igazítandó: purge = max(H_max, max_holding).
A másodlagos modell kizárólag train+val-on tanul/kalibrál.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(close: pd.Series, sigma: pd.Series, *,
                          pt: float = 1.5, sl: float = 1.5,
                          max_holding: int = 5) -> pd.DataFrame:
    """Minden t-re: melyik gátat éri el ELŐSZÖR a [t+1, t+max_holding] ablakban.

    Gátak: felső = +pt·σ_t·√(nap), alsó = −sl·σ_t·√(nap) a t-től számított
    kumulatív log-hozamon. Kimenet: label ∈ {+1 (profit), −1 (stop), 0 (idő)},
    ret (a realizált kum. log-hozam az érintésig/időkorlátig), holding (napok).

    Kauzális címke-KÉPZÉS (a jövőt használja, mint minden target) — ezért purge
    kell a splitnél; a jellemzőket ez NEM érinti.
    """
    c = close.to_numpy(dtype=float)
    sig = sigma.clip(lower=1e-8).to_numpy()
    n = len(c)
    labels = np.zeros(n, dtype=int)
    rets = np.full(n, np.nan)
    holds = np.zeros(n, dtype=int)
    log_c = np.log(c)

    for t in range(n - 1):
        upper = pt * sig[t]
        lower = -sl * sig[t]
        end = min(t + max_holding, n - 1)
        hit = 0
        cum = 0.0
        h = 0
        for i in range(t + 1, end + 1):
            cum = log_c[i] - log_c[t]
            h = i - t
            # a gátat √h-val skálázzuk (a σ egy napra vonatkozik)
            if cum >= upper * np.sqrt(h):
                hit = 1
                break
            if cum <= lower * np.sqrt(h):
                hit = -1
                break
        labels[t] = hit
        rets[t] = cum
        holds[t] = h
    out = pd.DataFrame({"tb_label": labels, "tb_ret": rets, "tb_holding": holds},
                       index=close.index)
    out.iloc[n - 1] = [0, np.nan, 0]   # az utolsó napnak nincs jövője
    return out


def make_meta_labels(primary_dir: np.ndarray, tb_ret: np.ndarray) -> np.ndarray:
    """Meta-címke: 1, ha az elsődleges IRÁNY egyezik a realizált triple-barrier
    hozam előjelével (azaz a fogadás nyert volna), különben 0.

    primary_dir ∈ {+1, −1} (a kvantilis-medián előjele); tb_ret a realizált
    kum. log-hozam."""
    primary_dir = np.asarray(primary_dir)
    tb_ret = np.asarray(tb_ret, dtype=float)
    correct = np.sign(tb_ret) == np.sign(primary_dir)
    return correct.astype(int)


def required_purge(h_max: int, max_holding: int) -> int:
    """Purge = max(H_max, max_holding) — a triple-barrier időkorlátja miatt."""
    return int(max(h_max, max_holding))


def train_meta_model(X_meta: np.ndarray, meta_y: np.ndarray,
                     train_pos: np.ndarray, val_pos: np.ndarray, *,
                     seed: int = 42):
    """Másodlagos LightGBM-osztályozó a 'fogadjunk-e' kérdésre; train-en tanul,
    val-on early-stop. Determinisztikus."""
    import lightgbm as lgb
    from quantimate.models.models_gbdt import DETERMINISM_PARAMS

    X = np.asarray(X_meta, dtype=float)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           min_child_samples=20, seed=seed, **DETERMINISM_PARAMS)
    m.fit(X[train_pos], meta_y[train_pos],
          eval_set=[(X[val_pos], meta_y[val_pos])], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def meta_position_size(p_meta: np.ndarray, primary_dir: np.ndarray,
                       base_fraction: np.ndarray) -> np.ndarray:
    """Pozíció = előjel(elsődleges) × P_meta × alapméret (fél-Merton).
    A meta-valószínűség szűr (P_meta alacsony → kis/nincs pozíció) ÉS méretez."""
    return np.sign(primary_dir) * np.asarray(p_meta) * np.asarray(base_fraction)
