"""V7-16 tesztek: hash-lánc integritás (manipuláció → detektálás), PSI kézi
példán, CUSUM drift-jelzés, havi riport."""
import json

import numpy as np
import pytest

from quantimate.backtest.live_ledger import (append_prediction, cusum, monthly_report,
                                    psi, read_ledger, verify_chain)


def _sample_pred(i):
    return {"date": f"2026-07-{i:02d}", "median": 0.001 * i, "p_up": 0.55,
            "lo85": -0.05, "hi85": 0.05, "position": 0.3}


def test_chain_valid_after_appends(tmp_path):
    led = tmp_path / "ledger.jsonl"
    for i in range(1, 6):
        append_prediction(led, _sample_pred(i))
    res = verify_chain(led)
    assert res["valid"] is True and res["n"] == 5 and res["broken_at"] is None


def test_chain_detects_tampering(tmp_path):
    """Egy sor tartalmának utólagos módosítása → verify_chain elbukik."""
    led = tmp_path / "ledger.jsonl"
    for i in range(1, 6):
        append_prediction(led, _sample_pred(i))

    lines = led.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[2])
    rec["position"] = 0.99                    # csendes módosítás
    lines[2] = json.dumps(rec, ensure_ascii=False)
    led.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = verify_chain(led)
    assert res["valid"] is False
    assert res["broken_at"] == 2


def test_chain_detects_deleted_row(tmp_path):
    led = tmp_path / "ledger.jsonl"
    for i in range(1, 6):
        append_prediction(led, _sample_pred(i))
    lines = led.read_text(encoding="utf-8").splitlines()
    del lines[2]                              # sor törlése → prev_hash lánc törik
    led.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify_chain(led)["valid"] is False


def test_prev_hash_links_are_chained(tmp_path):
    led = tmp_path / "ledger.jsonl"
    append_prediction(led, _sample_pred(1))
    r2 = append_prediction(led, _sample_pred(2))
    recs = read_ledger(led)
    assert r2["prev_hash"] == recs[0]["hash"]
    assert recs[0]["prev_hash"] == "0" * 64


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(0, 1, 5000)
    assert psi(ref, cur) < 0.05


def test_psi_flags_shift():
    """Eltolt eloszlás → PSI a jelentős küszöb (0.25) fölé."""
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(1.5, 1, 5000)
    assert psi(ref, cur) > 0.25


def test_cusum_alarms_on_rising_loss():
    """Ugró átlagú veszteség-sor → CUSUM riaszt; stabil sor → nem."""
    rng = np.random.default_rng(2)
    stable = rng.normal(0.25, 0.02, 200)
    assert cusum(stable, target=0.25)["alarm_at"] is None
    rising = np.concatenate([rng.normal(0.25, 0.02, 100),
                             rng.normal(0.32, 0.02, 100)])
    res = cusum(rising, target=0.25)
    assert res["alarm_at"] is not None and res["alarm_at"] >= 100


def test_monthly_report_coverage(tmp_path):
    led = tmp_path / "ledger.jsonl"
    rng = np.random.default_rng(3)
    for i in range(1, 101):
        pred = _sample_pred(i % 28 + 1)
        y = rng.normal(0, 0.02)
        pred["realized"] = float(y)
        pred["lo85"], pred["hi85"] = -0.033, 0.033   # ~85% egy N(0,0.02)-nál
        append_prediction(led, pred)
    rep = monthly_report(led)
    assert rep["n_records"] == 100 and rep["n_realized"] == 100
    assert 0.75 < rep["coverage85"] < 0.95
    assert rep["coverage85_ci"][0] <= rep["coverage85"] <= rep["coverage85_ci"][1]
