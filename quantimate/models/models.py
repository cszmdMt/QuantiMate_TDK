"""Modellek: kvantilis-regressziós fejek pinball loss-szal,
RevIN, LayerNorm-os architektúrák, AdamW+cosine+clipnorm, determinizmus.

Kimeneti konvenció: minden modell (batch, H·Q) alakú lineáris kimenetet ad,
amit (batch, H, Q)-ra formálunk; H = horizontok, Q = kvantilisek száma.
A kvantilis-keresztezést predikció UTÁN rendezéssel javítjuk (np.sort a
kvantilis-tengelyen) — kis adaton a keresztezés gyakori (hibakatalógus 3.).
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (LSTM, GRU, Add, AveragePooling1D,
                                     Bidirectional, Concatenate, Conv1D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling1D,
                                     Input, Layer, LayerNormalization,
                                     MaxPooling1D, MultiHeadAttention,
                                     SimpleRNN)

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
HORIZONS = (1, 5, 21)


# ── Determinizmus (2.5) ──────────────────────────────────────────────────────

def set_determinism(seed: int) -> None:
    """Seed + op-determinizmus minden tanítás előtt. CPU-n 10–30% lassulás —
    belefér (hibakatalógus 9.); ha egy op nem támogatja, dokumentált kivétel."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"enable_op_determinism nem elérhető: {exc} — a seed-elés marad.")


# ── Loss és kvantilis-műveletek ──────────────────────────────────────────────

def make_pinball_loss(quantiles=QUANTILES, n_horizons=len(HORIZONS)):
    """Átlagolt pinball (kvantilis-) loss: L_q(e)=mean(max(q·e,(q−1)·e)),
    e = y − ŷ_q; összloss = kvantilisek és horizontok átlaga."""
    q = tf.constant(quantiles, dtype=tf.float32)[None, None, :]        # (1,1,Q)

    def pinball(y_true, y_pred):
        y_pred = tf.reshape(y_pred, (-1, n_horizons, len(quantiles)))  # (B,H,Q)
        e = tf.expand_dims(y_true, -1) - y_pred                        # (B,H,Q)
        return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))

    pinball.__name__ = "pinball"
    return pinball


def pinball_loss_np(y_true: np.ndarray, y_pred_q: np.ndarray,
                    quantiles=QUANTILES) -> float:
    """Numpy pinball loss kiértékeléshez. y_true (B,H) vagy (B,);
    y_pred_q (B,H,Q) vagy (B,Q)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_q = np.asarray(y_pred_q, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred_q = y_pred_q[:, None, :]
    q = np.asarray(quantiles)[None, None, :]
    e = y_true[:, :, None] - y_pred_q
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def predict_quantiles(model, X, n_horizons=len(HORIZONS),
                      n_quantiles=len(QUANTILES), batch_size=256) -> np.ndarray:
    """(B, H, Q) kvantilis-predikció, a Q-tengely mentén RENDEZVE
    (kvantilis-keresztezés elleni monotonizálás — dokumentált korrekció)."""
    raw = model.predict(X, verbose=0, batch_size=batch_size)
    q = raw.reshape(-1, n_horizons, n_quantiles)
    return np.sort(q, axis=-1)


def prob_up_from_quantiles(q_values: np.ndarray, quantiles=QUANTILES) -> np.ndarray:
    """P(y>0) a kvantilisekből lineáris interpolációval.

    A (τ, ŷ_τ) pontpárokra: P(Y≤0) = interp(0, ŷ, τ), szélein az utolsó
    kvantilis-pár meredekségével extrapolálva, majd [0.001, 0.999] közé vágva.
    q_values: (B, Q) — EGY horizont rendezett kvantilisei.
    """
    taus = np.asarray(quantiles, dtype=float)
    q_values = np.asarray(q_values, dtype=float)
    out = np.empty(len(q_values))
    for i, qv in enumerate(q_values):
        if 0.0 <= qv[0]:       # minden kvantilis ≥ 0 → bal szélső extrapoláció
            slope = (taus[1] - taus[0]) / max(qv[1] - qv[0], 1e-12)
            p_le = taus[0] + slope * (0.0 - qv[0])
        elif 0.0 >= qv[-1]:    # minden kvantilis ≤ 0 → jobb szélső extrapoláció
            slope = (taus[-1] - taus[-2]) / max(qv[-1] - qv[-2], 1e-12)
            p_le = taus[-1] + slope * (0.0 - qv[-1])
        else:
            p_le = np.interp(0.0, qv, taus)
        out[i] = 1.0 - np.clip(p_le, 0.001, 0.999)
    return out


# ── RevIN (B1) ───────────────────────────────────────────────────────────────

class RevIN(Layer):
    """Reversible Instance Normalization (Kim et al., ICLR 2022) — bemeneti ág.

    Ablakonkénti (instance) μ/σ-normalizálás tanulható affin paraméterrel.
    A kimeneti oldalon nincs visszaállítás: a target eleve σ_t-normált
    hozam-térben él, ott nincs mit "visszaskálázni" (B1 specifikáció).
    """

    def __init__(self, eps: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(d,), initializer="ones")
        self.beta = self.add_weight(name="beta", shape=(d,), initializer="zeros")

    def call(self, x):
        mu = tf.reduce_mean(x, axis=1, keepdims=True)
        var = tf.math.reduce_variance(x, axis=1, keepdims=True)
        return (x - mu) / tf.sqrt(var + self.eps) * self.gamma + self.beta

    def get_config(self):
        cfg = super().get_config()
        cfg["eps"] = self.eps
        return cfg


# ── Architektúrák (A9/A10b: BatchNorm→LayerNorm mindenhol) ──────────────────

def _head(n_out: int):
    return Dense(n_out, activation="linear")


def _maybe_revin(inp, use_revin: bool):
    return RevIN(name="revin")(inp) if use_revin else inp


def build_simple_rnn(input_shape, n_out, use_revin=False):
    inp = Input(shape=input_shape)
    x = _maybe_revin(inp, use_revin)
    x = SimpleRNN(64, activation="tanh")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    return Model(inp, _head(n_out)(x), name="SimpleRNN")


def build_lstm(input_shape, n_out, use_revin=False):
    inp = Input(shape=input_shape)
    x = _maybe_revin(inp, use_revin)
    x = LSTM(96, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(48)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    return Model(inp, _head(n_out)(x), name="LSTM_stacked")


def build_gru(input_shape, n_out, use_revin=False):
    inp = Input(shape=input_shape)
    x = _maybe_revin(inp, use_revin)
    x = GRU(96)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    return Model(inp, _head(n_out)(x), name="GRU")


def build_cnn_bilstm(input_shape, n_out, use_revin=False, pooling="last"):
    """CNN-BiLSTM. pooling='last' → a BiLSTM utolsó állapota;
    pooling='gap' → return_sequences + GlobalAveragePooling1D.
    A kettő közti döntés val-kísérlettel történik (A10b)."""
    inp = Input(shape=input_shape)
    x = _maybe_revin(inp, use_revin)
    x = Conv1D(64, kernel_size=3, activation="relu", padding="causal")(x)
    x = LayerNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(32, kernel_size=3, activation="relu", padding="causal")(x)
    x = MaxPooling1D(2)(x)
    if pooling == "gap":
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = GlobalAveragePooling1D()(x)
    else:
        x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    name = "CNN_BiLSTM_gap" if pooling == "gap" else "CNN_BiLSTM"
    return Model(inp, _head(n_out)(x), name=name)


def build_attention_lstm(input_shape, n_out, use_revin=False):
    inp = Input(shape=input_shape)
    x = _maybe_revin(inp, use_revin)
    seq = LSTM(64, return_sequences=True)(x)
    att = MultiHeadAttention(num_heads=4, key_dim=16)(seq, seq)
    seq = LayerNormalization()(Add()([seq, att]))
    vec = GlobalAveragePooling1D()(seq)
    vec = Dense(32, activation="relu")(vec)
    vec = Dropout(0.3)(vec)
    return Model(inp, _head(n_out)(vec), name="Attention_LSTM")


# ── Lineáris baseline-ok (A9: Zeng et al. 2023) ─────────────────────────────

def build_nlinear(input_shape, n_out, use_revin=False):
    """NLinear: az ablak utolsó értékének levonása + 1 lineáris réteg.
    (A kimenet normált hozam-térben van, visszaadás nem értelmezett.)"""
    inp = Input(shape=input_shape)
    last = inp[:, -1:, :]
    x = Flatten()(inp - last)
    return Model(inp, _head(n_out)(x), name="NLinear")


def build_dlinear(input_shape, n_out, use_revin=False, kernel=25):
    """DLinear: mozgóátlag-trend + maradék dekompozíció, áganként 1 lineáris."""
    inp = Input(shape=input_shape)
    trend = AveragePooling1D(pool_size=kernel, strides=1, padding="same")(inp)
    seasonal = inp - trend
    out = Add()([_head(n_out)(Flatten()(trend)),
                 _head(n_out)(Flatten()(seasonal))])
    return Model(inp, out, name="DLinear")


# ── Modern architektúrák (V7-6, TF-implementáció — keretkeverés tilos) ───────

def _transformer_block(x, num_heads, key_dim, ff_dim, dropout=0.1):
    att = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = LayerNormalization()(Add()([x, att]))
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1])(ff)
    return LayerNormalization()(Add()([x, ff]))


def build_patchtst(input_shape, n_out, use_revin=False, patch_len=10, stride=10,
                   d_model=64, num_heads=4, n_blocks=2):
    """PatchTST (Nie et al. 2023): az időtengelyt patch-ekre bontja, lineárisan
    beágyazza, transformer-kódolóval dolgozza fel. A patch-elés csökkenti a
    szekvencia-hosszt és lokális mintázatokat emel ki. (Egyszerűsített,
    channel-mixing változat a közös kvantilis-fejhez.)"""
    inp = Input(shape=input_shape)
    x = _maybe_revin(inp, use_revin)
    # Patch-beágyazás Conv1D-vel (kernel=patch_len, stride=stride) → (B, n_patches, d_model)
    tokens = Conv1D(d_model, kernel_size=patch_len, strides=stride,
                    padding="valid")(x)
    for _ in range(n_blocks):
        tokens = _transformer_block(tokens, num_heads, d_model // num_heads, d_model * 2)
    vec = GlobalAveragePooling1D()(tokens)
    vec = Dropout(0.2)(vec)
    return Model(inp, _head(n_out)(vec), name="PatchTST")


def build_itransformer(input_shape, n_out, use_revin=False, d_model=64,
                       num_heads=4, n_blocks=2):
    """iTransformer (Liu et al. 2024): a VÁLTOZÓK (nem az idő) a tokenek —
    az attention a csatornák közti interakciókat tanulja. A teljes idősort
    változónként egy d_model vektorba ágyazzuk, majd változó-attention."""
    from tensorflow.keras.layers import Permute
    inp = Input(shape=input_shape)
    x = _maybe_revin(inp, use_revin)
    xt = Permute((2, 1))(x)                               # (B, f, w) — változók a tengelyen
    tokens = Dense(d_model)(xt)                           # változónkénti embedding
    for _ in range(n_blocks):
        tokens = _transformer_block(tokens, num_heads, d_model // num_heads, d_model * 2)
    vec = GlobalAveragePooling1D()(tokens)               # változók fölött
    vec = Dropout(0.2)(vec)
    return Model(inp, _head(n_out)(vec), name="iTransformer")


BUILDERS = {
    "SimpleRNN": build_simple_rnn,
    "LSTM_stacked": build_lstm,
    "GRU": build_gru,
    "CNN_BiLSTM": build_cnn_bilstm,
    "Attention_LSTM": build_attention_lstm,
    "NLinear": build_nlinear,
    "DLinear": build_dlinear,
    "PatchTST": build_patchtst,
    "iTransformer": build_itransformer,
}


# ── Tréning (A10c) ───────────────────────────────────────────────────────────

def train_quantile_model(builder, X_tr, Y_tr, X_va, Y_va, *, seed: int,
                         epochs: int = 40, batch_size: int = 64,
                         learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                         clipnorm: float = 1.0, patience: int = 6,
                         quantiles=QUANTILES, horizons=HORIZONS,
                         builder_kwargs: dict | None = None, verbose: int = 0):
    """Egy kvantilis-modell determinisztikus tanítása AdamW + cosine-decay
    + gradient clipping beállítással (A10c). A cosine decay_steps a TELJES
    epoch-plafonra méretezett (hibakatalógus 10.)."""
    set_determinism(seed)
    n_out = len(horizons) * len(quantiles)
    model = builder((X_tr.shape[1], X_tr.shape[2]), n_out,
                    **(builder_kwargs or {}))

    steps_per_epoch = math.ceil(len(X_tr) / batch_size)
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        learning_rate, decay_steps=epochs * steps_per_epoch)
    opt = tf.keras.optimizers.AdamW(learning_rate=schedule,
                                    weight_decay=weight_decay,
                                    clipnorm=clipnorm)
    model.compile(optimizer=opt, loss=make_pinball_loss(quantiles, len(horizons)))
    es = EarlyStopping(monitor="val_loss", patience=patience,
                       restore_best_weights=True)
    hist = model.fit(X_tr, Y_tr, validation_data=(X_va, Y_va), epochs=epochs,
                     batch_size=batch_size, callbacks=[es], verbose=verbose)
    return model, hist


# ── Fúziós háló (architektúra-demonstráció) ────────────────────────────

def build_fusion(input_shape, n_senti: int = 4):
    """Multi-input fúziós háló (technikai + fundamentális ág), BCE-iránnyal.

    A szentiment-ág tanításkor ÉS élő predikciókor is nulla-vektort kap
    (konzisztens); a háló szerepe architektúra-demonstráció — a backtest-lánc
    nem használja."""
    tech_in = Input(shape=input_shape, name="technikai_ag")
    x = Conv1D(64, 3, activation="relu", padding="causal")(tech_in)
    x = MaxPooling1D(2)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)

    senti_in = Input(shape=(n_senti,), name="fundamentalis_ag")
    s = Dense(16, activation="relu")(senti_in)
    s = Dense(8, activation="relu")(s)

    z = Concatenate()([x, s])
    z = Dense(32, activation="relu")(z)
    z = Dropout(0.2)(z)
    out = Dense(1, activation="sigmoid")(z)
    return Model([tech_in, senti_in], out, name="Fuzios_halo")
