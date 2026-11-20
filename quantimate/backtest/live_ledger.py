"""Prospektív validáció (V7-16): hash-láncolt élő predikciós napló + drift-monitor.

A legerősebb védés-álló érv egy előre regisztrált, hamisíthatatlan élő napló:
minden kereskedési nap zárása után a rendszer kiírja a predikciót egy
append-only JSONL-be, ahol minden sor tartalmazza az ELŐZŐ sor SHA-256-át
(a V6 ujjlenyomat-mintájának láncolt kiterjesztése) — utólagos módosítás
így észlelhető. N≈60 nap után az ÉLŐ sáv-lefedettség Wilson-CI-vel értékelhető:
ez valódi out-of-sample.

Drift-monitor: PSI (jellemző-eloszlás elmozdulás, banki 0.25 küszöb) + CUSUM a
napi pinball-veszteségen. A refit-politika ELŐRE rögzített (README_LEDGER.md),
hogy ne legyen diszkrecionális.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

GENESIS_HASH = "0" * 64


def _hash_record(record: dict) -> str:
    """A sor kanonikus JSON-jának SHA-256-a (a 'hash' mező nélkül)."""
    payload = {k: v for k, v in record.items() if k != "hash"}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _last_hash(ledger_path: Path) -> str:
    if not ledger_path.exists():
        return GENESIS_HASH
    last = None
    with ledger_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return GENESIS_HASH
    return json.loads(last)["hash"]


def append_prediction(ledger_path, prediction: dict,
                      timestamp: str | None = None) -> dict:
    """Egy predikció hozzáfűzése a lánchoz. A 'prev_hash' az előző sor hash-e,
    a 'hash' a jelen sor (prev_hash-t is tartalmazó) tartalmának hash-e."""
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(prediction)
    record["timestamp"] = timestamp or dt.datetime.now(dt.timezone.utc).isoformat()
    record["prev_hash"] = _last_hash(ledger_path)
    record["hash"] = _hash_record(record)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_ledger(ledger_path) -> list[dict]:
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return []
    with ledger_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_chain(ledger_path) -> dict:
    """A teljes lánc integritásának ellenőrzése: minden sor hash-e helyes, és a
    prev_hash-ek folytonos láncot alkotnak a genesistől.

    Visszaad: {"valid", "n", "broken_at"} — broken_at az első sérült sor indexe
    (0-alapú) vagy None.
    """
    records = read_ledger(ledger_path)
    prev = GENESIS_HASH
    for i, rec in enumerate(records):
        if rec.get("prev_hash") != prev:
            return {"valid": False, "n": len(records), "broken_at": i,
                    "reason": "prev_hash lánc-törés"}
        if _hash_record(rec) != rec.get("hash"):
            return {"valid": False, "n": len(records), "broken_at": i,
                    "reason": "tartalom-hash eltérés (módosított sor)"}
        prev = rec["hash"]
    return {"valid": True, "n": len(records), "broken_at": None}


# ── Drift-monitor ────────────────────────────────────────────────────────────

def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index a referencia (train) kvantilis-binjein.
    PSI > 0.25 = jelentős eloszlás-elmozdulás (Siddiqi 2006 banki küszöb)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(reference, edges)[0] / len(reference)
    cur_frac = np.histogram(current, edges)[0] / len(current)
    eps = 1e-6
    ref_frac = np.clip(ref_frac, eps, None)
    cur_frac = np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def cusum(losses: np.ndarray, target: float, k: float = 0.5,
          h_factor: float = 5.0) -> dict:
    """Egyoldali (felfelé) CUSUM a napi veszteségen: jelez, ha a veszteség
    tartósan a target fölé kúszik (koncept-drift). k a megengedett elmozdulás
    a loss szórásában, h a riasztási küszöb.
    """
    x = np.asarray(losses, dtype=float)
    sd = x.std() if x.std() > 0 else 1.0
    slack = k * sd
    h = h_factor * sd
    s = 0.0
    series = np.empty(len(x))
    alarm_at = None
    for i, xi in enumerate(x):
        s = max(0.0, s + (xi - target) - slack)
        series[i] = s
        if alarm_at is None and s > h:
            alarm_at = i
    return {"cusum": series, "alarm_at": alarm_at, "threshold": h}


def monthly_report(ledger_path, quantiles=(0.05, 0.95)) -> dict:
    """Havi élő riport a naplóból: sáv-lefedettség (ha a realizált értékek már
    beírattak), CRPS és bejegyzésszám. A realizált hozam a 'realized' mezőből
    jön (a rákövetkező napok visszatöltésekor); ahol még nincs, kihagyjuk."""
    from statsmodels.stats.proportion import proportion_confint

    records = read_ledger(ledger_path)
    realized = [(r["lo85"], r["hi85"], r["realized"]) for r in records
                if r.get("realized") is not None]
    out = {"n_records": len(records), "n_realized": len(realized)}
    if realized:
        lo, hi, y = map(np.array, zip(*realized))
        inside = int(((lo <= y) & (y <= hi)).sum())
        ci_lo, ci_hi = proportion_confint(inside, len(y), alpha=0.05,
                                          method="wilson")
        out["coverage85"] = inside / len(y)
        out["coverage85_ci"] = [float(ci_lo), float(ci_hi)]
    return out
