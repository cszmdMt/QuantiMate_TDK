"""Idősor-alapmodell zero-shot baseline (V7-6): Chronos.

2024–2026 paradigmaváltás: százmilliárd pontnyi idősoron előtanított modellek
tanítás nélkül (zero-shot) adnak kvantilis-előrejelzést. Itt a kérdés: "mennyit
ér a domain-tudásunk a generikus előtanulás ellen?" — a Chronos BASELINE-ként
lép a versenybe (nem megváltóként; a pénzügyi napi hozamokon az alapmodellek
gyakran NEM verik az egyszerű baseline-okat — Re(Visiting) TSFMs 2025).

**Import-guard:** a torch/chronos opcionális; ha hiányzik vagy a súly nem
tölthető, a modul őszinte "nem futott" jelzést ad, és a fő lánc független marad.

**Szivárgás-fenntartás:** az alapmodell előtanító korpusza nem auditálható →
elvi kontamináció-gyanú; a teszt-időszak 2024 utáni, a modell-cutoff korábbi,
így a gyanú alacsony, de kimondandó.
"""
from __future__ import annotations

import numpy as np


def chronos_available() -> bool:
    import importlib.util as u
    return u.find_spec("torch") is not None and u.find_spec("chronos") is not None


def zeroshot_quantiles(context_series_list, horizons, quantiles,
                       model_name: str = "amazon/chronos-bolt-small") -> np.ndarray:
    """(B, H, Q) kvantilis-predikció a Chronos-szal, minden minta múltbeli
    kontextusából. context_series_list: B db 1D log-hozam kontextus (múlt).

    Ha a chronos/torch hiányzik vagy a betöltés hibázik: RuntimeError, amit a
    hívó elkap és a szakaszt őszintén kihagyja."""
    if not chronos_available():
        raise RuntimeError("chronos/torch nem elérhető — a zero-shot szakasz kihagyva")
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(model_name, device_map="cpu",
                                               torch_dtype=torch.float32)
    max_h = max(horizons)
    out = np.empty((len(context_series_list), len(horizons), len(quantiles)))
    q_levels = list(quantiles)
    for i, ctx in enumerate(context_series_list):
        ctx_t = torch.tensor(np.asarray(ctx, dtype=np.float32))
        # kumulatív hozam-kvantilisek: a Chronos napi hozamot jósol; a h napos
        # kumulatívhoz a napi predikciókat összegezzük (közelítés).
        q_pred, _ = pipe.predict_quantiles(ctx_t, prediction_length=max_h,
                                           quantile_levels=q_levels)
        qp = q_pred[0].numpy()               # (max_h, Q)
        for hi, h in enumerate(horizons):
            out[i, hi, :] = np.sort(qp[:h].sum(axis=0))   # kum. + monotonizálás
    return out
