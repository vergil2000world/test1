"""
MASTER UPH / OEE ROOT-CAUSE SCRIPT - POOLED (all machines = same type)
===================================================================
All machines are the same type, so every UPH/OEE record - regardless
of which MACHINE_ID it came from - is treated as one more sample of
the SAME underlying process. Records are pooled within each period
(Jul, Aug) rather than grouped per machine. This turns Step 3 from a
cross-machine proxy into a REAL two-sample significance test: each
period now has a genuine sample (n = number of records) with its own
mean and std, which is exactly what a z-test needs.

  STEP 1  KPI evolution        - pooled Jul vs Aug record-level stats
  STEP 2  Log decomposition    - on the pooled means (one calculation)
  STEP 3  Significance         - two-sample z-test (Welch-style SE)
                                  on Performance_% and Availability_%,
                                  computed PER RECORD then compared
                                  as two real distributions
  STEP 4  Counterfactual       - signed comparison, no abs() (same
                                  bug-fix as before)
  STEP 5  Pareto (real cause)  - unchanged: independent downtime log,
                                  park-level (see note in that section)

INPUTS
------
1. MACHINE_XLSX  - two-sheet workbook (Jul/Aug), columns:
                   MACHINE_ID, UPH, OEE   (one row per production record)
                   MACHINE_ID is read but NOT used to group - it is
                   only reported as a record count / sanity check.
2. DOWNTIME_LOG  - tab-separated, UTF-16 CSV export, columns:
                   Fiscal Year, Fiscal Month, Reason Code, Dur (seconds)

KEY ASSUMPTION (confirmed by the user): Quality = 100% throughout, and
all machines are the same type -> pooling their records is valid.
Per record:
    Performance_% = UPH / UPH_THEORETICAL * 100
    Availability_% = OEE / Performance_% * 100   (from OEE = A x P)

OUTPUTS
-------
Output/Master_Root_Cause_Pooled/
    pooled_records.csv          (every record, with Performance_%/Availability_% added)
    period_summary.csv          (Step 1-4 result, one row per period + the comparison)
    downtime_pareto.csv         (Step 5, real reason-code ranking)
    master_report.txt           (Q&A-oriented report + tables)
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ============================================================
# CONFIG - adjust these paths/labels to your files
# ============================================================

MACHINE_XLSX = r"Resources\Machine_UPH_JUL_AUG.xlsx"
SHEET0, SHEET1 = "Jul26", "Aug26"

DOWNTIME_LOG = r"Resources\CDM_Reports_Bouskoura - Reason Codes 2025sept_2026sept.csv"
DOWNTIME_ENCODING = "utf-16"
DOWNTIME_SEP = "\t"
FISCAL_YEAR = 2026
MONTH0_NAME, MONTH1_NAME = "July", "August"
MONTH0_DAYS, MONTH1_DAYS = 31, 31

OUTPUT_DIR = "Output/Master_Root_Cause_Pooled"
PERIOD0, PERIOD1 = "Jul", "Aug"
UPH_THEORETICAL = 9455

Z_SIGNIFICANT = 2.0
DOMINANCE_GAP_CLOSED_MIN = 50.0


# ============================================================
# STEP 1 - LOAD + POOL RECORDS, PER-RECORD KPI
# ============================================================

def load_and_enrich(xlsx_path, sheet_name, uph_theoretical, period_label):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    for col in ("MACHINE_ID", "UPH", "OEE"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in sheet '{sheet_name}'")

    df = df.copy()
    df["Period"] = period_label
    df["Performance_%"] = df["UPH"] / uph_theoretical * 100
    # guard against Performance_% == 0 (would divide by zero below)
    df["Availability_%"] = np.where(
        df["Performance_%"] != 0, df["OEE"] / df["Performance_%"] * 100, np.nan
    )
    return df


def period_stats(df, label):
    n_machines = df["MACHINE_ID"].nunique()
    n_records = len(df)
    stats = {"Period": label, "N_Records": n_records, "N_Machines": n_machines}
    for col in ["UPH", "OEE", "Performance_%", "Availability_%"]:
        stats[f"{col}_mean"] = df[col].mean()
        stats[f"{col}_median"] = df[col].median()
        stats[f"{col}_std"] = df[col].std(ddof=1)
    return stats


def step1_kpi_pooled(xlsx_path, sheet0, sheet1, uph_theoretical):
    df0 = load_and_enrich(xlsx_path, sheet0, uph_theoretical, PERIOD0)
    df1 = load_and_enrich(xlsx_path, sheet1, uph_theoretical, PERIOD1)
    pooled = pd.concat([df0, df1], ignore_index=True)

    s0 = period_stats(df0, PERIOD0)
    s1 = period_stats(df1, PERIOD1)
    return pooled, s0, s1


# ============================================================
# STEP 2 - LOG DECOMPOSITION (on pooled period means)
# ============================================================

def step2_log_decomposition(s0, s1):
    A0, A1 = s0["Availability_%_mean"] / 100, s1["Availability_%_mean"] / 100
    P0, P1 = s0["Performance_%_mean"] / 100, s1["Performance_%_mean"] / 100

    ln_A = np.log(A1 / A0)
    ln_P = np.log(P1 / P0)
    terms = {"Availability": ln_A, "Performance": ln_P}
    loss = {k: v for k, v in terms.items() if v < 0}
    offset = {k: v for k, v in terms.items() if v >= 0}
    loss_sum, offset_sum = sum(loss.values()), sum(offset.values())

    def share(term):
        group = loss_sum if term < 0 else offset_sum
        return round(term / group * 100, 2) if group else 0.0

    return {
        "ln_Availability": round(ln_A, 6),
        "ln_Performance": round(ln_P, 6),
        "ln_Total": round(ln_A + ln_P, 6),
        "OEE_Change_%": round((np.exp(ln_A + ln_P) - 1) * 100, 2),
        "Availability_Share_%": share(ln_A),
        "Performance_Share_%": share(ln_P),
        "Dominant_Loss_Factor": min(loss, key=loss.get) if loss else None,
    }


# ============================================================
# STEP 3 - TRUE TWO-SAMPLE SIGNIFICANCE (Welch z/t approximation)
# ============================================================

def significance_label(z):
    if z is None or pd.isna(z):
        return "N/A"
    z = abs(z)
    if z >= 3:
        return "VERY_SIGNIFICANT"
    if z >= 2:
        return "SIGNIFICANT"
    if z >= 1:
        return "MODERATE"
    return "LIKELY_RANDOM"


def two_sample_z(mean0, std0, n0, mean1, std1, n1):
    """Welch-style standard error - valid two-sample test since each
    period now has a real sample (n records), not a single point."""
    se = np.sqrt((std0 ** 2) / n0 + (std1 ** 2) / n1)
    if se == 0 or np.isnan(se):
        return None
    return (mean1 - mean0) / se


def step3_significance_pooled(s0, s1):
    results = {}
    for metric in ["Performance_%", "Availability_%"]:
        z = two_sample_z(
            s0[f"{metric}_mean"], s0[f"{metric}_std"], s0["N_Records"],
            s1[f"{metric}_mean"], s1[f"{metric}_std"], s1["N_Records"],
        )
        results[metric] = {"z": z, "label": significance_label(z)}
    return results


def randomness_label(sig_results):
    perf_unlikely = sig_results["Performance_%"]["z"] is not None and abs(sig_results["Performance_%"]["z"]) >= Z_SIGNIFICANT
    avail_unlikely = sig_results["Availability_%"]["z"] is not None and abs(sig_results["Availability_%"]["z"]) >= Z_SIGNIFICANT
    if perf_unlikely and avail_unlikely:
        return "BOTH_UNLIKELY_RANDOM"
    if perf_unlikely:
        return "PERFORMANCE_UNLIKELY_RANDOM"
    if avail_unlikely:
        return "AVAILABILITY_UNLIKELY_RANDOM"
    return "COMPATIBLE_WITH_RANDOM_VARIATION"


# ============================================================
# STEP 4 - COUNTERFACTUAL (signed comparison, no abs())
# ============================================================

def step4_counterfactual_pooled(s0, s1):
    A0, A1 = s0["Availability_%_mean"] / 100, s1["Availability_%_mean"] / 100
    P0, P1 = s0["Performance_%_mean"] / 100, s1["Performance_%_mean"] / 100
    UPH0, UPH1 = s0["UPH_mean"], s1["UPH_mean"]

    OEE0, OEE1 = A0 * P0, A1 * P1
    gap = OEE0 - OEE1
    OEE_cf_A = A0 * P1
    OEE_cf_P = A1 * P0

    if gap != 0:
        gap_closed_A = (OEE_cf_A - OEE1) / gap * 100
        gap_closed_P = (OEE_cf_P - OEE1) / gap * 100
    else:
        gap_closed_A = gap_closed_P = 0.0

    recovered_uph_A = UPH1 + (UPH0 - UPH1) * (gap_closed_A / 100)
    recovered_uph_P = UPH1 + (UPH0 - UPH1) * (gap_closed_P / 100)

    # SIGNED comparison - never abs(). A factor that already improved
    # (negative gap_closed) must never outrank a real loss driver.
    if gap_closed_A >= gap_closed_P:
        root, contribution = "Availability", gap_closed_A
    else:
        root, contribution = "Performance", gap_closed_P

    return {
        "Gap_Closed_Availability_%": round(gap_closed_A, 2),
        "Gap_Closed_Performance_%": round(gap_closed_P, 2),
        "Recovered_UPH_Availability": round(recovered_uph_A, 2),
        "Recovered_UPH_Performance": round(recovered_uph_P, 2),
        "UPH_Gain_Availability": round(recovered_uph_A - UPH1, 2),
        "UPH_Gain_Performance": round(recovered_uph_P - UPH1, 2),
        "Root_Cause": root,
        "Root_Cause_Contribution_%": round(contribution, 2),
    }


def verdict(root, contribution, sig_results):
    if root is None:
        return "NO_ROOT_CAUSE"
    z = sig_results[f"{root}_%"]["z"]
    if z is not None and abs(z) >= Z_SIGNIFICANT and contribution >= DOMINANCE_GAP_CLOSED_MIN:
        return f"CONFIRMED_ROOT_CAUSE_{root.upper()}"
    if contribution >= DOMINANCE_GAP_CLOSED_MIN:
        return f"LIKELY_ROOT_CAUSE_{root.upper()}"
    return "NO_CONFIRMED_ROOT_CAUSE"


# ============================================================
# STEP 5 - PARETO FROM REAL DOWNTIME REASON CODES (park level)
# ============================================================

def step5_pareto(downtime_csv, encoding, sep, fiscal_year, month0_name, month1_name,
                  month0_days, month1_days):
    df = pd.read_csv(downtime_csv, encoding=encoding, sep=sep, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Fiscal Year"] = pd.to_numeric(df["Fiscal Year"], errors="coerce")
    df["Dur"] = pd.to_numeric(df["Dur"], errors="coerce")
    df["Fiscal Month"] = df["Fiscal Month"].astype(str).str.strip()

    df_year = df[df["Fiscal Year"] == fiscal_year]

    def month_stats(name, calendar_days):
        m = df_year[df_year["Fiscal Month"] == name]
        downtime_sec = m["Dur"].fillna(0).sum()
        calendar_sec = calendar_days * 24 * 3600
        available_sec = calendar_sec - downtime_sec
        availability = available_sec / calendar_sec * 100
        return m, downtime_sec, available_sec, availability

    m0, dt0_sec, avail0_sec, avail0_pct = month_stats(month0_name, month0_days)
    m1, dt1_sec, avail1_sec, avail1_pct = month_stats(month1_name, month1_days)

    park_availability = pd.DataFrame([
        {"Month": PERIOD0, "Downtime_Hours": round(dt0_sec / 3600, 2),
         "Available_Hours": round(avail0_sec / 3600, 2), "Availability_%": round(avail0_pct, 4),
         "Occurrences": len(m0)},
        {"Month": PERIOD1, "Downtime_Hours": round(dt1_sec / 3600, 2),
         "Available_Hours": round(avail1_sec / 3600, 2), "Availability_%": round(avail1_pct, 4),
         "Occurrences": len(m1)},
    ])

    reason_summary = (
        df_year[df_year["Fiscal Month"].isin([month0_name, month1_name])]
        .groupby(["Fiscal Month", "Reason Code"])
        .agg(Occurrences=("Dur", "count"), Downtime_Hours=("Dur", lambda x: x.sum() / 3600))
        .reset_index()
    )

    pivot = reason_summary.pivot_table(
        index="Reason Code", columns="Fiscal Month", values="Downtime_Hours", fill_value=0
    ).reset_index()
    if month0_name in pivot.columns and month1_name in pivot.columns:
        pivot["Downtime_Gap_Hours"] = (pivot[month1_name] - pivot[month0_name]).round(2)
        pivot = pivot.sort_values("Downtime_Gap_Hours", ascending=False)

    return park_availability, reason_summary, pivot


# ============================================================
# REPORT - Q&A oriented, with supporting tables
# ============================================================

def build_report(s0, s1, decomp, sig_results, cf, verdict_str, park_availability, reason_pivot):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    lines = []

    def h1(title):
        lines.append(title); lines.append("=" * 140); lines.append("")

    def h2(title):
        lines.append(title); lines.append("-" * 140)

    h1("MASTER UPH / OEE ROOT-CAUSE REPORT (POOLED - all machines = same type)")
    lines.append(f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Periods   : {PERIOD0} -> {PERIOD1}")
    lines.append(f"{PERIOD0}: {s0['N_Records']} records across {s0['N_Machines']} machines (pooled as one sample)")
    lines.append(f"{PERIOD1}: {s1['N_Records']} records across {s1['N_Machines']} machines (pooled as one sample)")
    lines.append("Assumption: all machines are the same type (pooling valid); Quality = 100% throughout; "
                 "OEE = Availability x Performance.")
    lines.append("")

    # -------------------- Q1 --------------------
    h2("Q1 - DID THE PROCESS ACTUALLY CHANGE, OR COULD THIS BE RANDOM SAMPLE NOISE?")
    lines.append(f"OEE change: {decomp['OEE_Change_%']:.2f}%  "
                 f"(Availability {s0['Availability_%_mean']:.2f}% -> {s1['Availability_%_mean']:.2f}%, "
                 f"Performance {s0['Performance_%_mean']:.2f}% -> {s1['Performance_%_mean']:.2f}%)")
    lines.append("Two-sample z-test (real historical samples - pooled records, not a proxy):")
    for metric in ["Performance_%", "Availability_%"]:
        z = sig_results[metric]["z"]
        z_str = f"{z:+.2f}" if z is not None else "N/A"
        lines.append(f"  {metric:<15} z = {z_str:>8}  -> {sig_results[metric]['label']}")
    lines.append(f"Randomness assessment: {randomness_label(sig_results)}")
    lines.append("")

    # -------------------- Q2 --------------------
    h2("Q2 - WHICH FACTOR EXPLAINS MOST OF THE CHANGE: AVAILABILITY OR PERFORMANCE?")
    lines.append(f"Dominant loss factor: {decomp['Dominant_Loss_Factor']} "
                 f"(ln_Availability={decomp['ln_Availability']:+.4f} [{decomp['Availability_Share_%']:.1f}% of its group], "
                 f"ln_Performance={decomp['ln_Performance']:+.4f} [{decomp['Performance_Share_%']:.1f}% of its group])")
    lines.append("")

    # -------------------- Q3 --------------------
    h2("Q3 - IF WE RESTORED ONE FACTOR TO JULY'S LEVEL, HOW MUCH WOULD WE RECOVER?")
    lines.append(f"Restoring Availability alone -> gap closed {cf['Gap_Closed_Availability_%']:.1f}%, "
                 f"UPH gain = {cf['UPH_Gain_Availability']:+.1f}")
    lines.append(f"Restoring Performance alone  -> gap closed {cf['Gap_Closed_Performance_%']:.1f}%, "
                 f"UPH gain = {cf['UPH_Gain_Performance']:+.1f}")
    lines.append("(A value outside 0-100%, or negative, means the OTHER factor moved strongly enough "
                 "to distort the single-factor overshoot/undershoot - read as directional, not literal share.)")
    lines.append(f"Root cause (signed comparison, never abs()): {cf['Root_Cause']} "
                 f"({cf['Root_Cause_Contribution_%']:.1f}% gap closed)")
    lines.append(f"Verdict: {verdict_str}")
    lines.append("")

    # -------------------- Q4 --------------------
    h2("Q4 - WHICH SPECIFIC DOWNTIME REASONS EXPLAIN THE AVAILABILITY CHANGE? (real log)")
    if reason_pivot is not None and "Downtime_Gap_Hours" in reason_pivot.columns:
        lines.append("Reason codes ranked by increase in downtime hours (Jul -> Aug):")
        lines.append(reason_pivot.to_string(index=False))
    else:
        lines.append("Downtime log not available/processed - Step 5 skipped.")
    lines.append("")

    # -------------------- Synthesis --------------------
    h2("SYNTHESIS")
    if verdict_str.startswith("CONFIRMED_ROOT_CAUSE"):
        lines.append(f"{cf['Root_Cause']} is the CONFIRMED root cause: statistically significant "
                     f"(two-sample z-test on {s0['N_Records']}+{s1['N_Records']} pooled records) "
                     f"AND dominant in the counterfactual (>={DOMINANCE_GAP_CLOSED_MIN:.0f}% gap closed).")
    else:
        lines.append(f"Leading hypothesis: {cf['Root_Cause']} ({cf['Root_Cause_Contribution_%']:.1f}% gap closed), "
                     "but it does not clear both the significance and dominance bar simultaneously.")
    lines.append("")

    # -------------------- Supporting tables --------------------
    h1("SUPPORTING TABLES")

    h2("TABLE 1 - PERIOD SUMMARY (STEP 1: pooled record-level distributions)")
    summary_rows = []
    for s in (s0, s1):
        summary_rows.append({
            "Period": s["Period"], "N_Records": s["N_Records"], "N_Machines": s["N_Machines"],
            "UPH_mean": round(s["UPH_mean"], 2), "UPH_std": round(s["UPH_std"], 2),
            "OEE_mean": round(s["OEE_mean"], 2), "OEE_std": round(s["OEE_std"], 2),
            "Performance_%_mean": round(s["Performance_%_mean"], 2), "Performance_%_std": round(s["Performance_%_std"], 2),
            "Availability_%_mean": round(s["Availability_%_mean"], 2), "Availability_%_std": round(s["Availability_%_std"], 2),
        })
    lines.append(pd.DataFrame(summary_rows).to_string(index=False))
    lines.append("")

    h2("TABLE 2 - LOG DECOMPOSITION (STEP 2)")
    lines.append(pd.DataFrame([decomp]).to_string(index=False))
    lines.append("")

    h2("TABLE 3 - SIGNIFICANCE (STEP 3, two-sample z-test)")
    sig_table = pd.DataFrame([
        {"Metric": m, "Z_Score": sig_results[m]["z"], "Label": sig_results[m]["label"]}
        for m in ["Performance_%", "Availability_%"]
    ])
    lines.append(sig_table.to_string(index=False))
    lines.append("")

    h2("TABLE 4 - COUNTERFACTUAL / UPH RECOVERY (STEP 4)")
    lines.append(pd.DataFrame([cf]).to_string(index=False))
    lines.append("")

    h2("TABLE 5 - PARK AVAILABILITY FROM REAL DOWNTIME LOG (STEP 5)")
    lines.append(park_availability.to_string(index=False) if park_availability is not None else "(not available)")
    lines.append("")

    h2("TABLE 6 - FULL REASON-CODE PARETO (STEP 5)")
    lines.append(reason_pivot.to_string(index=False) if reason_pivot is not None else "(not available)")

    return "\n".join(lines) + "\n"


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pooled, s0, s1 = step1_kpi_pooled(MACHINE_XLSX, SHEET0, SHEET1, UPH_THEORETICAL)
    decomp = step2_log_decomposition(s0, s1)
    sig_results = step3_significance_pooled(s0, s1)
    cf = step4_counterfactual_pooled(s0, s1)
    verdict_str = verdict(cf["Root_Cause"], cf["Root_Cause_Contribution_%"], sig_results)

    park_availability = reason_summary = reason_pivot = None
    try:
        park_availability, reason_summary, reason_pivot = step5_pareto(
            DOWNTIME_LOG, DOWNTIME_ENCODING, DOWNTIME_SEP,
            FISCAL_YEAR, MONTH0_NAME, MONTH1_NAME, MONTH0_DAYS, MONTH1_DAYS
        )
    except FileNotFoundError:
        print(f"[WARN] Downtime log not found at {DOWNTIME_LOG} - Step 5 (Pareto) skipped.")

    pooled.to_csv(os.path.join(OUTPUT_DIR, "pooled_records.csv"), index=False)
    period_summary_df = pd.DataFrame([s0, s1])
    period_summary_df.to_csv(os.path.join(OUTPUT_DIR, "period_summary.csv"), index=False)
    if reason_pivot is not None:
        reason_pivot.to_csv(os.path.join(OUTPUT_DIR, "downtime_pareto.csv"), index=False)

    report_text = build_report(s0, s1, decomp, sig_results, cf, verdict_str, park_availability, reason_pivot)
    report_path = os.path.join(OUTPUT_DIR, "master_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print("Generated files:")
    print(f"- {os.path.join(OUTPUT_DIR, 'pooled_records.csv')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'period_summary.csv')}")
    if reason_pivot is not None:
        print(f"- {os.path.join(OUTPUT_DIR, 'downtime_pareto.csv')}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
