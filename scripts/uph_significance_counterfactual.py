"""
Step 3 (Significance) -> Step 4 (Counterfactual), chained.

Uses the same Availability_%/Performance_% columns as
uph_decomposition.py (Quality fixed at 100%, not read/derived).

STEP 3 - SIGNIFICANCE (z-score)
--------------------------------
z = (observed_change) / historical_std_of_that_metric

Needs a historical distribution (mean/std across several past
periods, not just the two months being compared) to be meaningful.
If you don't have monthly history, pass std_a / std_p yourself
(e.g. from a longer log) via --std_a / --std_p. Without them the
script still runs the counterfactual step but flags Step 3 as
"not computed" rather than guessing a std from two points.

Rule: |z| > 2 => the change is unlikely to be random noise.

STEP 4 - COUNTERFACTUAL
--------------------------------
Holding Quality = 100% throughout, OEE = A x P, so:
    OEE_jul = A0 * P0
    OEE_aug = A1 * P1
    gap     = OEE_jul - OEE_aug   (the drop to explain)

For each factor, ask: "what if THAT factor alone had stayed at
its July value, with the other factor left at its actual August
value?"
    OEE_cf_restore_A = A0 * P1   (Availability restored, P as-is)
    OEE_cf_restore_P = A1 * P0   (Performance restored, A as-is)

gap_closed_% = (OEE_cf - OEE_aug) / gap * 100

A factor is the CONFIRMED DOMINANT CAUSE when:
    - its z-score is significant (|z| > 2), AND
    - restoring it alone closes > 50% of the gap, AND
    - that gap-closed is >= 1.5x the other factor's gap-closed
      (using absolute values, since a factor that improved will
      show a negative or >100% gap-closed rather than a clean
      partial share - see printed note)
"""

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def zscore(delta, std):
    if std is None or std == 0:
        return None
    return delta / std


def run(csv_path, period0="Jul", period1="Aug", std_a=None, std_p=None):
    df = pd.read_csv(csv_path)
    row = df.iloc[0]

    A0 = row[f"Availability_{period0}_%"] / 100
    A1 = row[f"Availability_{period1}_%"] / 100
    P0 = row[f"Performance_{period0}_%"] / 100
    P1 = row[f"Performance_{period1}_%"] / 100

    delta_A = A1 - A0
    delta_P = P1 - P0

    z_A = zscore(delta_A, std_a)
    z_P = zscore(delta_P, std_p)

    lines = [
        "UPH SIGNIFICANCE -> COUNTERFACTUAL REPORT",
        "=" * 70,
        f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Source    : {os.path.abspath(csv_path)}",
        f"Periods   : {period0} -> {period1}",
        "",
        "=== STEP 3 - SIGNIFICANCE ===",
    ]
    if z_A is None:
        lines.append("Availability: z-score not computed (no historical std supplied via --std_a)")
    else:
        lines.append(f"Availability: delta={delta_A*100:+.2f} pts, z={z_A:+.2f} "
                      f"-> {'SIGNIFICANT' if abs(z_A) > 2 else 'not significant'} (|z|>2 rule)")
    if z_P is None:
        lines.append("Performance: z-score not computed (no historical std supplied via --std_p)")
    else:
        lines.append(f"Performance: delta={delta_P*100:+.2f} pts, z={z_P:+.2f} "
                      f"-> {'SIGNIFICANT' if abs(z_P) > 2 else 'not significant'} (|z|>2 rule)")

    OEE_jul = A0 * P0
    OEE_aug = A1 * P1
    gap = OEE_jul - OEE_aug

    OEE_cf_A = A0 * P1  # restore Availability only
    OEE_cf_P = A1 * P0  # restore Performance only

    closed_A = (OEE_cf_A - OEE_aug) / gap * 100 if gap != 0 else None
    closed_P = (OEE_cf_P - OEE_aug) / gap * 100 if gap != 0 else None

    lines.append("")
    lines.append("=== STEP 4 - COUNTERFACTUAL ===")
    lines.append(f"OEE_{period0}={OEE_jul*100:.2f}%  OEE_{period1}={OEE_aug*100:.2f}%  gap={gap*100:.2f} pts")
    lines.append(f"Restore Availability only -> OEE={OEE_cf_A*100:.2f}%  gap closed={closed_A:.1f}%")
    lines.append(f"Restore Performance only  -> OEE={OEE_cf_P*100:.2f}%  gap closed={closed_P:.1f}%")

    candidates = {"Availability": (z_A, closed_A), "Performance": (z_P, closed_P)}
    valid = {k: v for k, v in candidates.items() if v[1] is not None}
    ranked = sorted(valid.items(), key=lambda kv: abs(kv[1][1]), reverse=True)

    lines.append("")
    lines.append("=== VERDICT ===")
    if len(ranked) < 2:
        lines.append("Not enough data to compare factors.")
    else:
        (top_name, (top_z, top_closed)), (second_name, (second_z, second_closed)) = ranked[0], ranked[1]
        dominance_ok = abs(top_closed) > 50 and (second_closed == 0 or abs(top_closed) >= 1.5 * abs(second_closed))
        sig_ok = top_z is not None and abs(top_z) > 2

        if dominance_ok and sig_ok:
            lines.append(f"{top_name} is the CONFIRMED DOMINANT CAUSE: "
                          f"z={top_z:+.2f} (significant), gap closed={top_closed:.1f}% "
                          f"(vs {second_name}'s {second_closed:.1f}%).")
        elif dominance_ok and not sig_ok:
            lines.append(f"{top_name} dominates the counterfactual (gap closed={top_closed:.1f}%), "
                          f"but significance could not be confirmed (need --std_{top_name[0].lower()}).")
        else:
            lines.append(f"No single factor clears both the significance and dominance bar. "
                          f"Leading candidate: {top_name} (gap closed={top_closed:.1f}%, z={top_z}).")

        if abs(top_closed) > 100 or top_closed < 0:
            lines.append(f"\nNote: {top_name}'s gap-closed value is outside the normal 0-100% range. "
                          f"This happens when the OTHER factor moved strongly enough in one direction "
                          f"to distort the counterfactual overshoot/undershoot - read it as a directional "
                          f"signal (very large vs very small/negative), not a literal percentage of the gap.")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="global_park_performance.csv")
    parser.add_argument("report_path", nargs="?", default="uph_significance_counterfactual_report.txt")
    parser.add_argument("--period0", default="Jul")
    parser.add_argument("--period1", default="Aug")
    parser.add_argument("--std_a", type=float, default=None,
                         help="historical std dev of Availability_% (as a fraction, e.g. 0.03 for 3 pts)")
    parser.add_argument("--std_p", type=float, default=None,
                         help="historical std dev of Performance_% (as a fraction, e.g. 0.05 for 5 pts)")
    args = parser.parse_args()

    report_text = run(args.csv_path, args.period0, args.period1, args.std_a, args.std_p)
    print(report_text)

    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Report written to: {os.path.abspath(args.report_path)}")
