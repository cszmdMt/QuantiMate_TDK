"""Anti-leakage tesztek (shift-grep, index-diszjunktság,
scaler-függetlenség)."""
from pathlib import Path

import numpy as np

from quantimate.validation.splits import fit_train_scaler, make_purged_splits, make_sequences

REPO = Path(__file__).resolve().parent.parent


def test_no_negative_shift_in_features():
    """Szabály: jövőre néző shift(-...) KIZÁRÓLAG a targets.py-ban élhet —
    a csomag MINDEN más moduljára statikus grep-teszt (V7-ben kiterjesztve)."""
    import re
    offenders = {}
    for py in sorted((REPO / "quantimate").rglob("*.py")):
        if py.name == "targets.py":
            continue
        matches = re.findall(r"shift\(\s*-", py.read_text(encoding="utf-8"))
        if matches:
            offenders[py.relative_to(REPO / "quantimate").as_posix()] = matches
    assert offenders == {}, f"Jövőre néző shift a csomagban: {offenders}"


def test_split_disjoint_with_purge_embargo():
    """A határok körüli 26 (21+5) nap egyik halmazban sincs; a train
    utolsó címkéje (purge=21 nap alatt realizálódik) nem ér át a val-ba."""
    n, window, purge, embargo = 2443, 90, 21, 5
    s = make_purged_splits(n, window=window, purge=purge, embargo=embargo)

    # Diszjunktság
    assert len(set(s.train) & set(s.val)) == 0
    assert len(set(s.val) & set(s.test)) == 0
    assert len(set(s.train) & set(s.test)) == 0

    # A train→val határ előtti 26 pozíció sehol sem szerepel
    gap_positions = set(range(s.i_train - purge - embargo, s.i_train))
    for name, block in (("train", s.train), ("val", s.val), ("test", s.test)):
        assert len(gap_positions & set(block)) == 0, f"purge-sáv a(z) {name}-ben"
    gap2 = set(range(s.i_val - purge - embargo, s.i_val))
    for name, block in (("train", s.train), ("val", s.val), ("test", s.test)):
        assert len(gap2 & set(block)) == 0, f"purge-sáv2 a(z) {name}-ben"

    # A train utolsó mintájának 21 napos címke-realizációja a val előtt zárul
    assert s.train.max() + purge < s.val.min()
    assert s.val.max() + purge < s.test.min()

    # Ablak-jogosultság
    assert s.train.min() >= window - 1


def test_scaler_independent_of_val_test():
    """A scaler paraméterei függetlenek a val/teszt adattól — val sor
    felülírása után a transzformált train BITRE változatlan."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (500, 7))
    s = make_purged_splits(500, window=30, purge=10, embargo=2)

    _, X_scaled_a = fit_train_scaler(X, s.train)

    X_mutated = X.copy()
    X_mutated[s.val] = 1e6          # val-sorok elrontása
    X_mutated[s.test] = -1e6        # teszt-sorok elrontása
    _, X_scaled_b = fit_train_scaler(X_mutated, s.train)

    np.testing.assert_array_equal(X_scaled_a[s.train], X_scaled_b[s.train])


def test_scaler_zero_iqr_guard():
    """Nulla-IQR (konstans dummy) oszlop nem okoz 0-val osztást/robbanást."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (200, 3))
    X[:, 1] = 1.0                   # konstans dummy oszlop
    s = make_purged_splits(200, window=10, purge=5, embargo=1)
    scaler, X_scaled = fit_train_scaler(X, s.train)
    assert np.isfinite(X_scaled).all()
    assert (np.abs(scaler.scale_) > 0).all()


def test_sequences_use_only_past():
    """A p pozíciójú minta ablakának utolsó sora X[p], első sora X[p-W+1]."""
    X = np.arange(100, dtype=float).reshape(-1, 1)
    seq = make_sequences(X, np.array([9, 50]), window=10)
    assert seq.shape == (2, 10, 1)
    assert seq[0, -1, 0] == 9 and seq[0, 0, 0] == 0
    assert seq[1, -1, 0] == 50 and seq[1, 0, 0] == 41
