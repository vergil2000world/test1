"""
Unified OEE root-cause analysis (Availability / Performance)
--------------------------------------------------------------
Reads a per-machine month-comparison CSV and, for each machine, chains
four steps into one pass:

  1. KPI evolution     - Performance/Availability/OEE gaps between two
                          periods (OEE = Availability x Performance).
  2. Significance       - z-score of each machine's gap against the
                          cross-machine distribution of that gap (a
                          practical stand-in for a real historical std
                          dev when only two periods are available -
                          see NOTE below).
  3. Log decomposition   - ln(A1/A0) and ln(P1/P0); whichever term is
                          negative is a loss factor, and its share of
                          the combined negative terms tells you which
                          one dragged OEE down more.
  4. Counterfactual      - "what if this ONE factor alone had stayed at
                          its baseline value, the other left as-is?"
                              gap_closed_% = (OEE_cf - OEE_actual) / gap * 100
                          A NEGATIVE gap_closed_% means restoring that
                          factor alone would have made things WORSE -
                          i.e. that factor already improved and is not
                          a loss driver, whatever its magnitude. This
                          script does NOT take abs() of gap_closed when
                          picking a root cause - an earlier version of
                          this script did, which let a factor that had
                          actually gotten better "win" on magnitude and
                          get blamed for the drop anyway.

NOTE on z-scores: with only two periods (Jul, Aug) there is no real
time history to compute a std dev from, so this script uses the
spread of each machine's gap ACROSS THE PARK as a proxy - "is this
machine's gap unusual compared to its peers this month?", not "is it
unusual compared to its own history?". Wire in a real historical std
dev (from a longer monthly log) if you have one; until then, read
Significance as "outlier vs. the park", not as statistical certainty.

Zero-baseline machines (Availability or Performance = 0% in either
period) are skipped from the decomposition/counterfactual/root-cause
columns and listed separately in the report - log(x/0) is undefined
and any "root cause" derived from it would be meaningless. Their raw
KPI rows are still written to the CSV and report.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "Output/Machine_UPH_Month_Comparison/machine_uph_month_comparison.csv"
OUTPUT_DIR = "Output/Unified_OEE_Root_Cause"
PERIOD0, PERIOD1 = "Jul", "Aug"

REQUIRED_COLS = [
    "MACHINE_ID",
    f"Avg_UPH_{PERIOD0}", f"Avg_UPH_{PERIOD1}",
    f"Performance_%_{PERIOD0}", f"Performance_%_{PERIOD1}",
    f"Availability_%_{PERIOD0}", f"Availability_%_{PERIOD1}",
]


# ============================================================
# LOAD / VALIDATE
# ============================================================

def load_and_validate(csv_path):
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    dup_ids = df.loc[df["MACHINE_ID"].duplicated(), "MACHINE_ID"].unique().tolist()
    if dup_ids:
        raise ValueError(f"MACHINE_ID must be unique - duplicates found: {dup_ids}")

    return df


# ============================================================
# KPI EVOLUTION
# ============================================================

def add_kpi_evolution(df):
    df["Performance_Gap_%"] = df[f"Performance_%_{PERIOD1}"] - df[f"Performance_%_{PERIOD0}"]
    df["Availability_Gap_%"] = df[f"Availability_%_{PERIOD1}"] - df[f"Availability_%_{PERIOD0}"]
    df[f"OEE_%_{PERIOD0}"] = df[f"Performance_%_{PERIOD0}"] * df[f"Availability_%_{PERIOD0}"] / 100
    df[f"OEE_%_{PERIOD1}"] = df[f"Performance_%_{PERIOD1}"] * df[f"Availability_%_{PERIOD1}"] / 100
    df["OEE_Gap_%"] = df[f"OEE_%_{PERIOD1}"] - df[f"OEE_%_{PERIOD0}"]
    return df


# ============================================================
# Z-SCORE / SIGNIFICANCE / RANDOMNESS
# ============================================================

def add_zscores(df):
    zscore_stats = {}
    for label, col in [("Performance", "Performance_Gap_%"), ("Availability", "Availability_Gap_%")]:
        mean = df[col].mean()
        std = df[col].std(ddof=1)
        if not std or np.isnan(std):
            std = 1.0  # one machine / zero spread across the park - avoid div-by-zero
        df[f"{label}_ZScore"] = (df[col] - mean) / std
        zscore_stats[label] = {"mean": mean, "std": std}
    return df, zscore_stats


def significance_label(z):
    z = abs(z)
    if z >= 3:
        return "VERY_SIGNIFICANT"
    if z >= 2:
        return "SIGNIFICANT"
    if z >= 1:
        return "MODERATE"
    return "LIKELY_RANDOM"


def randomness_label(perf_z, avail_z):
    perf_unlikely = abs(perf_z) >= 2
    avail_unlikely = abs(avail_z) >= 2
    if perf_unlikely and avail_unlikely:
        return "BOTH_UNLIKELY_RANDOM"
    if perf_unlikely:
        return "PERFORMANCE_UNLIKELY_RANDOM"
    if avail_unlikely:
        return "AVAILABILITY_UNLIKELY_RANDOM"
    return "COMPATIBLE_WITH_RANDOM_VARIATION"


# ============================================================
# DECOMPOSITION + COUNTERFACTUAL (per machine)
# ============================================================

EMPTY_RESULT = {
    "ln_Availability": None, "ln_Performance": None,
    "Availability_Share_%": None, "Performance_Share_%": None,
    "Dominant_Loss_Factor": None,
    "Gap_Closed_Availability_%": None, "Gap_Closed_Performance_%": None,
    "Recovered_UPH_Availability": None, "Recovered_UPH_Performance": None,
    "UPH_Gain_Availability": None, "UPH_Gain_Performance": None,
    "Root_Cause": None, "Root_Cause_Contribution_%": None,
}


def decompose_row(row):
    """Log decomposition + counterfactual + UPH recovery for one machine.

    Returns a dict of results, or a copy of EMPTY_RESULT (Root_Cause=None)
    if Availability/Performance is 0% in either period - a log-ratio
    against zero is undefined and any conclusion drawn from it would be
    fabricated, not derived.
    """
    A0 = row[f"Availability_%_{PERIOD0}"] / 100
    A1 = row[f"Availability_%_{PERIOD1}"] / 100
    P0 = row[f"Performance_%_{PERIOD0}"] / 100
    P1 = row[f"Performance_%_{PERIOD1}"] / 100

    if 0 in (A0, A1, P0, P1):
        return dict(EMPTY_RESULT)

    UPH0 = row[f"Avg_UPH_{PERIOD0}"]
    UPH1 = row[f"Avg_UPH_{PERIOD1}"]

    # --- log decomposition ---
    ln_A = np.log(A1 / A0)
    ln_P = np.log(P1 / P0)
    losses = {k: v for k, v in {"Availability": ln_A, "Performance": ln_P}.items() if v < 0}
    total_loss = sum(losses.values())

    share_A = (ln_A / total_loss * 100) if (ln_A < 0 and total_loss) else 0.0
    share_P = (ln_P / total_loss * 100) if (ln_P < 0 and total_loss) else 0.0
    dominant_loss = min(losses, key=losses.get) if losses else None

    # --- counterfactual ---
    OEE0, OEE1 = A0 * P0, A1 * P1
    gap = OEE0 - OEE1
    OEE_cf_A = A0 * P1  # restore Availability only, Performance left at its actual value
    OEE_cf_P = A1 * P0  # restore Performance only, Availability left at its actual value

    if gap != 0:
        gap_closed_A = (OEE_cf_A - OEE1) / gap * 100
        gap_closed_P = (OEE_cf_P - OEE1) / gap * 100
    else:
        gap_closed_A = gap_closed_P = 0.0

    # --- UPH recovery: signed, proportional to the (signed) gap closed.
    # A negative gap_closed correctly SUBTRACTS here - restoring a factor
    # that already improved would only make actual UPH worse, not better.
    recovered_uph_A = UPH1 + (UPH0 - UPH1) * (gap_closed_A / 100)
    recovered_uph_P = UPH1 + (UPH0 - UPH1) * (gap_closed_P / 100)

    # --- root cause: whichever factor closes MORE of the gap, signed.
    # (Not abs() - a factor that improved must never outrank a factor
    # that actually caused loss just because its magnitude is bigger.)
    if gap_closed_A >= gap_closed_P:
        root, contribution = "Availability", gap_closed_A
    else:
        root, contribution = "Performance", gap_closed_P

    return {
        "ln_Availability": round(ln_A, 5),
        "ln_Performance": round(ln_P, 5),
        "Availability_Share_%": round(share_A, 2),
        "Performance_Share_%": round(share_P, 2),
        "Dominant_Loss_Factor": dominant_loss,
        "Gap_Closed_Availability_%": round(gap_closed_A, 2),
        "Gap_Closed_Performance_%": round(gap_closed_P, 2),
        "Recovered_UPH_Availability": round(recovered_uph_A, 2),
        "Recovered_UPH_Performance": round(recovered_uph_P, 2),
        "UPH_Gain_Availability": round(recovered_uph_A - UPH1, 2),
        "UPH_Gain_Performance": round(recovered_uph_P - UPH1, 2),
        "Root_Cause": root,
        "Root_Cause_Contribution_%": round(contribution, 2),
    }


def verdict_for(row):
    root = row["Root_Cause"]
    if root is None:
        return "SKIPPED_ZERO_BASELINE"

    z = abs(row[f"{root}_ZScore"])
    contribution = row["Root_Cause_Contribution_%"]

    if z >= 2 and contribution >= 50:
        return f"CONFIRMED_ROOT_CAUSE_{root.upper()}"
    if contribution >= 50:
        return f"LIKELY_ROOT_CAUSE_{root.upper()}"
    return "NO_CONFIRMED_ROOT_CAUSE"


def build_final_df(df):
    extra_rows = [decompose_row(row) for _, row in df.iterrows()]
    extra_df = pd.DataFrame(extra_rows)
    # Positional concat, not a MACHINE_ID merge: extra_rows is built in the
    # same row order as df.iterrows(), and a merge on a key that turned out
    # not to be unique would silently fan out rows - already guarded above,
    # but concat sidesteps the risk entirely rather than relying on the guard.
    final_df = pd.concat([df.reset_index(drop=True), extra_df], axis=1)
    final_df["Verdict"] = final_df.apply(verdict_for, axis=1)
    final_df["Performance_Significance"] = final_df["Performance_ZScore"].apply(significance_label)
    final_df["Availability_Significance"] = final_df["Availability_ZScore"].apply(significance_label)
    final_df["Randomness_Assessment"] = final_df.apply(
        lambda r: randomness_label(r["Performance_ZScore"], r["Availability_ZScore"]), axis=1)
    return final_df


# ============================================================
# REPORT
# ============================================================

def build_report_text(final_df, csv_path, zscore_stats):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    skipped = final_df.loc[final_df["Root_Cause"].isna(), "MACHINE_ID"].tolist()
    scored = final_df.loc[final_df["Root_Cause"].notna()]

    lines = [
        "UNIFIED OEE ROOT CAUSE REPORT",
        "=" * 140,
        f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Source    : {os.path.abspath(csv_path)}",
        f"Periods   : {PERIOD0} -> {PERIOD1}",
        f"Machines  : {len(final_df)} total, {len(scored)} scored, {len(skipped)} skipped (zero baseline)",
        "",
    ]

    if skipped:
        lines.append("SKIPPED MACHINES (Availability or Performance = 0% in Jul or Aug - log-ratio undefined)")
        lines.append("-" * 140)
        lines.append(", ".join(skipped))
        lines.append("")

    lines.append("STEP 1 - KPI SUMMARY")
    lines.append("-" * 140)
    lines.append(f"Average Performance Gap  : {final_df['Performance_Gap_%'].mean():.2f}%")
    lines.append(f"Median Performance Gap   : {final_df['Performance_Gap_%'].median():.2f}%")
    lines.append(f"Average Availability Gap : {final_df['Availability_Gap_%'].mean():.2f}%")
    lines.append(f"Median Availability Gap  : {final_df['Availability_Gap_%'].median():.2f}%")
    lines.append(f"Average OEE Gap          : {final_df['OEE_Gap_%'].mean():.2f} pts")
    lines.append(f"Median OEE Gap           : {final_df['OEE_Gap_%'].median():.2f} pts")
    lines.append("")

    lines.append("STEP 2 - Z-SCORE SUMMARY (spread is across machines this month, not each machine's own history)")
    lines.append("-" * 140)
    z_summary = pd.DataFrame({
        "Metric": ["Performance", "Availability"],
        "Mean Gap": [round(zscore_stats["Performance"]["mean"], 2), round(zscore_stats["Availability"]["mean"], 2)],
        "Std Dev": [round(zscore_stats["Performance"]["std"], 2), round(zscore_stats["Availability"]["std"], 2)],
        "Abs(Z)>=2": [(final_df["Performance_ZScore"].abs() >= 2).sum(),
                      (final_df["Availability_ZScore"].abs() >= 2).sum()],
        "Abs(Z)>=3": [(final_df["Performance_ZScore"].abs() >= 3).sum(),
                      (final_df["Availability_ZScore"].abs() >= 3).sum()],
    })
    lines.append(z_summary.to_string(index=False))
    lines.append("")

    lines.append("STEP 3 - ROOT CAUSE SUMMARY (scored machines only)")
    lines.append("-" * 140)
    lines.append(scored["Root_Cause"].value_counts().to_string() if len(scored) else "(no scored machines)")
    lines.append("")

    lines.append("TOP AVAILABILITY CONTRIBUTORS")
    lines.append("-" * 140)
    lines.append(scored[["MACHINE_ID", "Availability_Share_%", "Availability_ZScore", "Verdict"]]
                  .sort_values("Availability_Share_%", ascending=False).head(10).to_string(index=False))
    lines.append("")

    lines.append("TOP PERFORMANCE CONTRIBUTORS")
    lines.append("-" * 140)
    lines.append(scored[["MACHINE_ID", "Performance_Share_%", "Performance_ZScore", "Verdict"]]
                  .sort_values("Performance_Share_%", ascending=False).head(10).to_string(index=False))
    lines.append("")

    lines.append("STEP 4 - COUNTERFACTUAL UPH RECOVERY")
    lines.append("-" * 140)
    lines.append("A NEGATIVE UPH_Gain means restoring that factor alone would have made UPH worse - i.e.")
    lines.append("that factor already improved and is not a loss driver. Gap_Closed_% can also land outside")
    lines.append("0-100%: that happens when the OTHER factor moved strongly enough to distort the single-factor")
    lines.append("overshoot/undershoot - read it as a directional signal, not a literal share of the gap.")
    recovery_cols = ["MACHINE_ID", f"Avg_UPH_{PERIOD0}", f"Avg_UPH_{PERIOD1}",
                      "Recovered_UPH_Availability", "Recovered_UPH_Performance",
                      "UPH_Gain_Availability", "UPH_Gain_Performance", "Root_Cause", "Verdict"]
    lines.append(scored[recovery_cols].sort_values("UPH_Gain_Performance", ascending=False).to_string(index=False))
    lines.append("")

    lines.append("STEP 5 - MACHINE DETAILS (all machines, including skipped)")
    lines.append("-" * 140)
    details_cols = ["MACHINE_ID", "Performance_Gap_%", "Performance_ZScore",
                     "Availability_Gap_%", "Availability_ZScore",
                     "Availability_Share_%", "Performance_Share_%",
                     "Gap_Closed_Availability_%", "Gap_Closed_Performance_%",
                     "Randomness_Assessment", "Root_Cause", "Verdict"]
    lines.append(final_df[details_cols].to_string(index=False))

    return "\n".join(lines) + "\n"


# ============================================================
# MAIN
# ============================================================

def main(csv_path=INPUT_FILE, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_validate(csv_path)
    df = add_kpi_evolution(df)
    df, zscore_stats = add_zscores(df)
    final_df = build_final_df(df)

    csv_out = os.path.join(output_dir, "unified_machine_analysis.csv")
    final_df.to_csv(csv_out, index=False)

    report_text = build_report_text(final_df, csv_path, zscore_stats)
    report_out = os.path.join(output_dir, "unified_root_cause_report.txt")
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print("Generated files:")
    print(f"- {csv_out}")
    print(f"- {report_out}")


if __name__ == "__main__":
    import sys

    csv_arg = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    outdir_arg = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    main(csv_arg, outdir_arg)
