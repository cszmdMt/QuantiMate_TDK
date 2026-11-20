"""Felosztás-fegyelem (A10a) + szekvenciaépítés + train-only skálázás.

Purge + embargo: a train→val és val→teszt határokon purge (=H_max=21) +
embargo (=5) napot EGYIK halmaz sem kap meg — így a train utolsó címkéi nem
realizálódnak a val/teszt időszakban.

A jellemző-mátrix (X_full) a TELJES idősoron él: a minta-pozíciók a
címkével rendelkező részhalmazból jönnek, az élő ablak X_full legvégéből.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class Splits:
    """Minta-pozíciók X_full sor-koordinátáiban."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    i_train: int   # címke-térbeli határ (info)
    i_val: int


def make_purged_splits(n_labeled: int, window: int, train_frac: float = 0.70,
                       val_frac: float = 0.15, purge: int = 21,
                       embargo: int = 5) -> Splits:
    """Időrendi 70/15/15 felosztás purge+embargóval.

    Feltevés: a címkézett sorok X_full első n_labeled sora (a targetek a sor
    végén hiányoznak, az elején nem — a feature-dropna után ez teljesül).
    A train/val a határ ELŐTTI purge+embargo napot adja le; a val/teszt
    határon ugyanígy. Egy minta ablakának a VÉGE dönt a hovatartozásról.
    """
    gap = purge + embargo
    i_train = int(n_labeled * train_frac)
    i_val = int(n_labeled * (train_frac + val_frac))

    all_pos = np.arange(n_labeled)
    train = all_pos[(all_pos >= window - 1) & (all_pos < i_train - gap)]
    val = all_pos[(all_pos >= i_train) & (all_pos < i_val - gap)]
    test = all_pos[all_pos >= i_val]
    return Splits(train=train, val=val, test=test, i_train=i_train, i_val=i_val)


def fit_train_scaler(X_full: np.ndarray, train_pos: np.ndarray,
                     clip: float = 8.0) -> tuple[RobustScaler, np.ndarray]:
    """RobustScaler KIZÁRÓLAG a train-sorokon illesztve; transform mindenhol.

    Nulla-IQR őr (hibakatalógus 2.): a közel-konstans oszlopok (pl. dummy-k)
    scale_-ját 1.0-ra cseréljük, hogy ne osszunk nullával.
    """
    scaler = RobustScaler().fit(X_full[train_pos])
    scaler.scale_ = np.where(np.abs(scaler.scale_) < 1e-12, 1.0, scaler.scale_)
    assert (np.abs(scaler.scale_) > 0).all()
    X_scaled = np.clip(scaler.transform(X_full), -clip, clip)
    return scaler, X_scaled


def make_sequences(X_scaled: np.ndarray, positions: np.ndarray,
                   window: int) -> np.ndarray:
    """(len(positions), window, n_feat) tenzor; a p pozíciójú minta ablaka
    X_scaled[p-window+1 .. p] — kizárólag p-ig tartó múlt."""
    assert (positions >= window - 1).all(), "Ablak a sor eleje elé nyúlna"
    return np.stack([X_scaled[p - window + 1: p + 1] for p in positions]
                    ).astype(np.float32)
