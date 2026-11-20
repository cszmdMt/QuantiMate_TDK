# -*- coding: utf-8 -*-
"""V8-4: NVDA earnings-naptár hitelesítés futtatása.

Két forrás (EDGAR 8-K Item 2.02 + Yahoo visualization) keresztellenőrzése →
data/nvda_earnings_verified.csv. Elfogadási küszöb: ≥95% egyezés; az eltérések listázva.
A CSV élesítése (config.earnings_source="verified") csak a CSV kézi
átnézése után történjen.

Futtatás:  ./.venv/Scripts/python.exe experiments/runners/verify_earnings.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from quantimate.earnings import VERIFIED_CSV, build_verified_csv  # noqa: E402


def main() -> int:
    tbl, match_rate = build_verified_csv()
    n = len(tbl)
    sched = tbl[tbl["in_overlap"] & ~tbl["is_preliminary"]]
    print(f"EDGAR 8-K Item 2.02 filingek a vizsgalati ablakban: {n} db "
          f"({tbl['announce_date'].min().date()} -> "
          f"{tbl['announce_date'].max().date()})")
    print(f"Forras-cimkek: {tbl['verified_by'].value_counts().to_dict()}")
    print(f"Yahoo-egyezes az atfedesi ablak UTEMEZETT jelentesein: "
          f"{int(sched['match'].sum())}/{len(sched)} = {match_rate:.1%} "
          f"(kuszob: 95%)")
    print(f"Napszak-eloszlas: {tbl['timing'].value_counts().to_dict()}")

    prelim = tbl[tbl["is_preliminary"]]
    if len(prelim):
        print("\nELOZETES/FIGYELMEZTETO 8-K-k (nem utemezett jelentes — a "
              "naptarbol alapbol kizarva):")
        print(prelim[["announce_date", "timing", "accession"]]
              .to_string(index=False))
    edonly = tbl[tbl["verified_by"] == "edgar_only"]
    if len(edonly):
        print("\nCSAK-EDGAR sorok (a Yahoo-lefedettseg utan — hatosagi forras "
              "onmagaban):")
        print(edonly[["announce_date", "timing", "accession"]]
              .to_string(index=False))
    mism = tbl[tbl["verified_by"] == "unmatched"]
    if len(mism):
        print("\nVALODI ELTERESEK (atfedesen belul, parositas nelkul):")
        print(mism[["announce_date", "timing", "accession"]].to_string(index=False))
    off = tbl[tbl["match"] & (tbl["yahoo_delta_days"] != 0)]
    if len(off):
        print("\nNEM NULLA NAPELTERESSEL parositott sorok:")
        print(off[["announce_date", "yahoo_date", "yahoo_delta_days",
                   "timing"]].to_string(index=False))

    print(f"\nCSV kiirva: {VERIFIED_CSV}")
    verdict = "TELJESUL" if match_rate >= 0.95 else "NEM TELJESUL"
    print(f"Verdikt (>=95% egyezes): {verdict}")
    print("KOVETKEZO LEPES: a felhasznalo attekinti es jovahagyja a CSV-t, "
          "elesites csak utana (earnings_source='verified').")
    return 0 if match_rate >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
