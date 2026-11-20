# -*- coding: utf-8 -*-
"""Golden-referencia generálása (egyszeri futtatás).

1. A data/*.parquet cache fagyasztott másolata → tests/golden/data/
2. A V6-ekvivalens pipeline artefaktum-hash-ei → tests/golden/reference_hashes.json
3. A fagyasztott V6-győztes (LSTM_stacked_s42_r0) val-pinballja referencia-értékként.

FIGYELEM: újragenerálás előtt — ha a hash eltér,
először azt kell tisztázni, forrás-adat revízió vagy kód-regresszió okozza-e.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from quantimate.config import DEFAULT_CONFIG, DATA_DIR  # noqa: E402
from quantimate.data import _safe_name, load_universe_from_dir  # noqa: E402
from quantimate.pipeline import build_pipeline  # noqa: E402

GOLDEN_DIR = REPO / "tests" / "golden"
GOLDEN_DATA = GOLDEN_DIR / "data"
GOLDEN_END = "2026-07-02"   # a V6 leadott futásának utolsó adatnapja


def sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main():
    GOLDEN_DATA.mkdir(parents=True, exist_ok=True)
    tickers = list(DEFAULT_CONFIG.ohlcv_tickers) + list(DEFAULT_CONFIG.light_tickers)
    total = 0
    for t in tickers:
        src = DATA_DIR / f"{_safe_name(t)}.parquet"
        if not src.exists():
            sys.exit(f"HIBA: hiányzó cache: {src} — előbb futtasd a pipeline-t.")
        dst = GOLDEN_DATA / src.name
        shutil.copy2(src, dst)
        total += dst.stat().st_size
    print(f"Fagyasztott parquet-ek: {len(tickers)} db, {total/1024:.0f} KB összesen")
    assert total < 5 * 1024 * 1024, "A golden-adat >5 MB — nem commitolható!"

    df = load_universe_from_dir(GOLDEN_DATA, DEFAULT_CONFIG.ohlcv_tickers,
                                DEFAULT_CONFIG.light_tickers, end_date=GOLDEN_END)
    art = build_pipeline(df, DEFAULT_CONFIG, verbose=True)

    # A HMM-EM folyamatok közti BLAS-zaja ~1e-12 a rezsim-valószínűségekben
    # → a rezsim-oszlopok TOLERANCIÁS (npy-referencia), minden más
    # oszlop EXAKT hash-sel védett.
    n_reg = len(art.regime_cols)
    feats_mat = art.feats.to_numpy(dtype=float)
    np.save(GOLDEN_DIR / "filtered_o.npy", art.filtered_o)

    hashes = {
        "golden_end": GOLDEN_END,
        "shape_feats": list(art.feats.shape),
        "n_labeled": art.n_labeled,
        "splits": {k: [int(getattr(art.S, k).min()), int(getattr(art.S, k).max()),
                       len(getattr(art.S, k))] for k in ("train", "val", "test")},
        "n_regime_cols": n_reg,
        "sha_feats_nonregime": sha(feats_mat[:, :-n_reg]),
        "sha_X_scaled_nonregime": sha(art.X_scaled[:, :-n_reg]),
        "sha_Y_all": sha(np.nan_to_num(art.Y_all, nan=-9e9)),
        "sha_splits": sha(np.concatenate([art.S.train, art.S.val, art.S.test])),
        "regime_names": art.regime_names,
    }

    # A fagyasztott V6-győztes val-pinballja (bit-repro a jelen gépen)
    winner_w = DATA_DIR / "models_v6" / "LSTM_stacked_s42_r0.weights.h5"
    if winner_w.exists():
        import tensorflow as tf
        from quantimate.models import (build_lstm, pinball_loss_np,
                                       predict_quantiles)
        from quantimate.splits import make_sequences
        tf.keras.backend.clear_session()
        H, Q = DEFAULT_CONFIG.horizons, DEFAULT_CONFIG.quantiles
        model = build_lstm((DEFAULT_CONFIG.window, art.X_scaled.shape[1]),
                           len(H) * len(Q), use_revin=False)
        model.load_weights(winner_w)
        X_va = make_sequences(art.X_scaled, art.S.val, DEFAULT_CONFIG.window)
        q_va = predict_quantiles(model, X_va, n_horizons=len(H), n_quantiles=len(Q))
        hashes["winner_val_pinball"] = float(
            pinball_loss_np(art.Y_all[art.S.val], q_va, quantiles=Q))
        print(f"V6-győztes val-pinball (referencia): {hashes['winner_val_pinball']:.10f}")
    else:
        hashes["winner_val_pinball"] = None
        print("FIGYELEM: nincs V6-győztes súlyfájl — a (b) golden-ág referencia nélkül marad.")

    out = GOLDEN_DIR / "reference_hashes.json"
    out.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {out}")


if __name__ == "__main__":
    main()
