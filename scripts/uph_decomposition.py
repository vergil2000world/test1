"""
UPH log-decomposition (Availability / Performance / Quality)
--------------------------------------------------------------
Reads a CSV with columns following this exact naming pattern
(as found in global_park_performance.csv):

    UPH_Theoretical
    Avg_UPH_<P0>, Median_UPH_<P0>, Avg_OEE_<P0>, Median_OEE_<P0>,
    Performance_<P0>_%, Availability_<P0>_%
    Avg_UPH_<P1>, Median_UPH_<P1>, Avg_OEE_<P1>, Median_OEE_<P1>,
    Performance_<P1>_%, Availability_<P1>_%

where <P0> and <P1> are the two period labels being compared
(e.g. "Jul" and "Aug").

Quality is not given directly, so it is derived from the identity
    OEE = A x P x Q  =>  Q = OEE / (A x P)

For each row it computes the exact log decomposition:
    ln(OEE1/OEE0) = ln(A1/A0) + ln(P1/P0) + ln(Q1/Q0)

then splits the three terms into a LOSS group (negative terms,
factors that dragged performance down) and an OFFSET group
(positive terms, factors that partly compensated), following the
mixed-sign decomposition rule: gross % share across all terms is
misleading whenever the signs differ, so shares are computed
within each group separately. The dominant factor is the one with
the largest-magnitude loss term.
"""

import numpy as np
import pandas as pd


def decompose(csv_path, period0="Jul", period1="Aug", metric="Avg"):
    df = pd.read_csv(csv_path)
    rows = []

    for idx, row in df.iterrows():
        A0 = row[f"Availability_{period0}_%"] / 100
        A1 = row[f"Availability_{period1}_%"] / 100
        P0 = row[f"Performance_{period0}_%"] / 100
        P1 = row[f"Performance_{period1}_%"] / 100
        OEE0 = row[f"{metric}_OEE_{period0}"] / 100
        OEE1 = row[f"{metric}_OEE_{period1}"] / 100

        # Derive Quality from OEE = A x P x Q
        Q0 = OEE0 / (A0 * P0)
        Q1 = OEE1 / (A1 * P1)

        terms = {
            "Availability": np.log(A1 / A0),
            "Performance": np.log(P1 / P0),
            "Quality": np.log(Q1 / Q0),
        }
        ln_total = sum(terms.values())

        loss = {k: v for k, v in terms.items() if v < 0}
        offset = {k: v for k, v in terms.items() if v >= 0}
        loss_sum = sum(loss.values())
        offset_sum = sum(offset.values())

        shares = {}
        for k, v in terms.items():
            group_sum = loss_sum if v < 0 else offset_sum
            shares[f"{k}_share_%"] = round(v / group_sum * 100, 1) if group_sum else 0.0

        dominant = min(loss, key=loss.get) if loss else None

        rows.append({
            "row": idx,
            f"A_{period0}": round(A0 * 100, 2), f"A_{period1}": round(A1 * 100, 2),
            f"P_{period0}": round(P0 * 100, 2), f"P_{period1}": round(P1 * 100, 2),
            f"Q_{period0}": round(Q0 * 100, 2), f"Q_{period1}": round(Q1 * 100, 2),
            "ln_A": round(terms["Availability"], 4),
            "ln_P": round(terms["Performance"], 4),
            "ln_Q": round(terms["Quality"], 4),
            "ln_total": round(ln_total, 4),
            "OEE_change_%": round((np.exp(ln_total) - 1) * 100, 2),
            **shares,
            "loss_sum": round(loss_sum, 4),
            "offset_sum": round(offset_sum, 4),
            "dominant_loss_factor": dominant,
        })

    return pd.DataFrame(rows)


def build_report_text(result, csv_path, period0, period1, metric):
    import os
    from datetime import datetime, timezone

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    lines = [
        "UPH LOG-DECOMPOSITION REPORT (Availability / Performance / Quality)",
        "=" * 70,
        f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Source    : {os.path.abspath(csv_path)}",
        f"Periods   : {period0} -> {period1}   Metric: {metric}",
        f"Rows      : {len(result)}",
        "",
        "Identity: ln(OEE1/OEE0) = ln(A1/A0) + ln(P1/P0) + ln(Q1/Q0)",
        "Shares are computed within the LOSS group and OFFSET group separately",
        "(mixing signs into one gross % would be misleading).",
        "",
        result.to_string(index=False),
        "",
        "Dominant loss factor per row",
        "-" * 40,
    ]

    for _, r in result.iterrows():
        factor = r["dominant_loss_factor"]
        if factor:
            ln_key = "ln_" + factor[0]
            lines.append(f"Row {r['row']}: dominant loss factor = {factor} (ln term = {r[ln_key]:.4f})")
        else:
            lines.append(f"Row {r['row']}: no loss factor (OEE improved on all fronts)")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import os
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "global_park_performance.csv"
    report_path = sys.argv[2] if len(sys.argv) > 2 else "uph_decomposition_report.txt"
    period0, period1, metric = "Jul", "Aug", "Avg"

    result = decompose(csv_path, period0=period0, period1=period1, metric=metric)
    report_text = build_report_text(result, csv_path, period0, period1, metric)

    print(report_text)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Report written to: {os.path.abspath(report_path)}")
