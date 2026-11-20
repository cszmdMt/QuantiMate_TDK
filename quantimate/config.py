"""Központi konfiguráció — minden tréning- és pipeline-paraméter egy helyen (A10e)."""
from dataclasses import dataclass, field
from pathlib import Path

# A repo gyökere a fájl helyéből származtatva — soha nem hardcode-olt abszolút út.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class PipelineConfig:
    """A teljes V6 pipeline paraméterei."""
    # Adat
    target: str = "NVDA"
    ohlcv_tickers: tuple = ("NVDA", "SPY", "QQQ", "AMD", "SMH")
    light_tickers: tuple = ("^VIX", "^VXN", "^TNX")   # nincs Volume mezőjük
    years: int = 10

    # Célváltozó
    horizons: tuple = (1, 5, 21)
    quantiles: tuple = (0.05, 0.25, 0.5, 0.75, 0.95)
    ewma_lambda: float = 0.94

    # Felosztás (A10)
    train_frac: float = 0.70
    val_frac: float = 0.15
    purge: int = 21           # = H_max
    embargo: int = 5

    # Szekvenciák
    window: int = 90
    clip_sigma: float = 8.0

    # Tréning (A10c)
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    clipnorm: float = 1.0
    patience: int = 6
    seeds: tuple = (7, 42, 123)

    # Backtest
    cost: float = 0.0005
    merton_cap: float = 0.5

    # Konformális (B2)
    coverage_wide: float = 0.85   # (q05, q95) sáv cél-lefedettsége
    coverage_mid: float = 0.50    # (q25, q75) sáv cél-lefedettsége
    aci_gamma: float = 0.01

    # Bayes-réteg
    lr_cap: float = 1.5
    signal_clip: float = 3.0

    # ── V7 flagek — alapállapot = V6-EKVIVALENS (minden blokk kapcsolható) ──
    sigma_mode: str = "ewma"          # "ewma" | "yz" | "yz_har"  (F1-1)
    yz_window: int = 21
    use_har_feature: bool = False     # F1-1b: HAR-σ előrejelzés jellemzőként
    use_gbdt: bool = False            # F1-5
    use_ensemble: bool = False        # F1-7
    ensemble_top_k: int = 3
    mondrian: bool = False            # F1-9 (feltételes kalibráció; a diagnosztika mindig fut)
    mondrian_min_n: int = 50
    conformal_method: str = "aci"     # "aci" | "nexcp" | "agaci"  (F2-10)
    nexcp_rho: float = 0.995
    use_fracdiff: bool = False        # F2-3
    use_macro: bool = False           # F2-2
    xai: bool = False                 # F2-17
    cpcv_groups: int = 8              # F1-15 (SMOKE: 6)
    cpcv_k_test: int = 2
    cpcv_refit: bool = False
    pbo_blocks: int = 16
    spa_n_boot: int = 1000

    # ── F3 flagek ──
    use_jump_model: bool = False      # F3-8 (rezsim-jellemző A/B)
    jump_lambda: float = 50.0
    use_metalabel: bool = False       # F3-12
    tb_pt: float = 1.5
    tb_sl: float = 1.5
    tb_max_holding: int = 5
    use_vol_target: bool = False      # F3-13
    use_cvar_cap: bool = False
    use_short: bool = False
    use_cs_cost: bool = False         # F3-14
    use_news: bool = False            # F3-4
    run_modern_arch: bool = False     # F3-6 (PatchTST/iTransformer a versenyben)
    run_zeroshot: bool = False        # F3-6 (Chronos)

    # ── V8 flagek — alapállapot = V6/V7-EKVIVALENS (minden alapból OFF) ──
    earnings_source: str = "generated"   # "generated" | "verified"  (V8-4)
    use_intraday: bool = False           # V8-1 (extras-út, GBDT-A/B)
    use_fundamentals: bool = False       # V8-2 (extras-út; 13F: dok. skip)
    fund_insider: bool = True            # V8-2 al-flagek (use_fundamentals alatt)
    fund_si: bool = True
    fund_shortvol: bool = True
    use_geo: bool = False                # V8-3 (macro-minta: a feats-be épül)
    geo_gpr: bool = True                 # V8-3 al-flagek (use_geo alatt)
    geo_tpu: bool = True                 # (geo_events/geo_gdelt: EXT, skip)
    sizing_mode: str = "merton"          # "merton" | "conformal"  (V8-11)
    sizing_smooth_lambda: float = 0.94   # V8-11: sávszélesség EWMA-simítása

    # ── V8.5 flagek — alapállapot OFF (a kapu dönt a presetről) ──
    use_pooled: bool = False             # V8-6: keresztmetszeti GBDT-tanítás


DEFAULT_CONFIG = PipelineConfig()

# ── V8_LIVE preset (kapu, mainV8 teljes futás, 2026-07-12) ────────────────
# Kizárólag az elfogadási kritériumot teljesítő beállítás térhet el a defaulttól.
# A jellemzőblokk-A/B-k egyike sem érte el a kaput (Δpinball ≤ −0.002 ÉS
# DM p<0.10) → minden V8 adatréteg-flag OFF marad; egyedül a V8-11 sizing ment
# át (val net Sharpe +0.1306 vs. fél-Merton +0.1155; teszt-riport: DSR 0.7605
# vs. 0.7246). Élesítés (élő réteg) csak külön döntéssel.
V8_LIVE_CONFIG = PipelineConfig(sizing_mode="conformal")
