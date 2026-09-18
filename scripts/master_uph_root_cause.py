"""
MASTER UPH / OEE ROOT-CAUSE SCRIPT
===================================================================
Consolidates the six separate scripts (10, 11, 12, 010, 13, 14) into
one pipeline, in the same 5-step order used throughout this analysis:

  STEP 1  KPI evolution        (was: 11_uph_performance_analysis)
  STEP 2  Log decomposition    (was: 12_process_01_check_shares +
                                       the per-machine version in 010)
  STEP 3  Significance/z-score (was: 14_process_step2_step3 / 010)
  STEP 4  Counterfactual       (was: 010 - the BUG-FIXED version.
                                 14_process_step2_step3 picks the
                                 dominant factor with abs(gap_closed),
                                 which can crown a factor that actually
                                 IMPROVED as the "cause" just because
                                 its counterfactual magnitude is large.
                                 010 fixed this with a signed comparison
                                 and that is the version kept here.)
  STEP 5  Pareto (real cause)  (was: 13_calculate_availability_jul_aug)

INPUTS (three independent sources, exactly as in the original scripts)
-------------------------------------------------------------------
1. MACHINE_XLSX  - two-sheet workbook (Jul/Aug), columns:
                   MACHINE_ID, UPH, OEE   (one row per production record)
2. DOWNTIME_LOG  - tab-separated, UTF-16 CSV export, columns:
                   Fiscal Year, Fiscal Month, Reason Code, Dur (seconds)
                   This is a REAL, independently measured downtime log -
                   not derived from UPH/OEE. It has NO machine ID column
                   in the original export, so Step 5 (Pareto) is at the
                   PARK level only. If your export ever adds a machine
                   column, group PARETO_BY below on it too.

KEY ASSUMPTION (confirmed by the user): Quality = 100% throughout.
So OEE = Availability x Performance, and Performance is computed as
    Performance_% = Avg_UPH / UPH_THEORETICAL * 100
(this is the run-rate factor, not overall throughput - it already
 excludes downtime, which is exactly the P in UPH = A x P x Q x 1/C_ideal)
and Availability is backed out as
    Availability_% = Avg_OEE / Performance_% * 100

OUTPUTS
-------
Output/Master_Root_Cause/
    machine_level_analysis.csv     (Steps 1-4, per machine)
    park_level_analysis.csv        (Steps 1-4, park aggregate)
    downtime_pareto.csv            (Step 5, real reason-code ranking)
    master_report.txt              (Q&A-oriented report + tables)
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

OUTPUT_DIR = "Output/Master_Root_Cause"
PERIOD0, PERIOD1 = "Jul", "Aug"
UPH_THEORETICAL = 9455

Z_SIGNIFICANT = 2.0          # |z| threshold for "not random"
DOMINANCE_GAP_CLOSED_MIN = 50.0   # gap_closed_% needed to call a factor dominant


# ============================================================
# STEP 1 - KPI EVOLUTION (per machine + park level)
# ============================================================

def step1_kpi(xlsx_path, sheet0, sheet1, uph_theoretical):
    df0 = pd.read_excel(xlsx_path, sheet_name=sheet0)
    df1 = pd.read_excel(xlsx_path, sheet_name=sheet1)

    for label, d in [(PERIOD0, df0), (PERIOD1, df1)]:
        for col in ("MACHINE_ID", "UPH", "OEE"):
            if col not in d.columns:
                raise ValueError(f"Column '{col}' missing in sheet for {label}")

    def summarize(df, suffix):
        g = df.groupby("MACHINE_ID").agg(
            **{
                f"Avg_UPH_{suffix}": ("UPH", "mean"),
                f"Median_UPH_{suffix}": ("UPH", "median"),
                f"Avg_OEE_{suffix}": ("OEE", "mean"),
                f"Median_OEE_{suffix}": ("OEE", "median"),
                f"Records_{suffix}": ("UPH", "count"),
            }
        ).reset_index()
        g[f"Performance_%_{suffix}"] = (g[f"Avg_UPH_{suffix}"] / uph_theoretical * 100).round(2)
        g[f"Availability_%_{suffix}"] = (
            g[f"Avg_OEE_{suffix}"] / g[f"Performance_%_{suffix}"] * 100
        ).round(2)
        return g

    s0 = summarize(df0, PERIOD0)
    s1 = summarize(df1, PERIOD1)
    machine_df = pd.merge(s0, s1, on="MACHINE_ID", how="outer")

    machine_df["Performance_Gap_%"] = (
        machine_df[f"Performance_%_{PERIOD1}"] - machine_df[f"Performance_%_{PERIOD0}"]
    ).round(2)
    machine_df["Availability_Gap_%"] = (
        machine_df[f"Availability_%_{PERIOD1}"] - machine_df[f"Availability_%_{PERIOD0}"]
    ).round(2)
    machine_df["Avg_UPH_Gap"] = (
        machine_df[f"Avg_UPH_{PERIOD1}"] - machine_df[f"Avg_UPH_{PERIOD0}"]
    ).round(2)
    machine_df["Avg_OEE_Gap"] = (
        machine_df[f"Avg_OEE_{PERIOD1}"] - machine_df[f"Avg_OEE_{PERIOD0}"]
    ).round(2)

    # Park level: computed from RAW records (not the mean of per-machine
    # numbers), same as the original 11_uph_performance_analysis script.
    def park_summary(df, suffix):
        avg_uph = df["UPH"].mean()
        avg_oee = df["OEE"].mean()
        perf = avg_uph / uph_theoretical * 100
        avail = avg_oee / perf * 100
        return {
            f"Avg_UPH_{suffix}": round(avg_uph, 2),
            f"Median_UPH_{suffix}": round(df["UPH"].median(), 2),
            f"Avg_OEE_{suffix}": round(avg_oee, 2),
            f"Median_OEE_{suffix}": round(df["OEE"].median(), 2),
            f"Performance_%_{suffix}": round(perf, 2),
            f"Availability_%_{suffix}": round(avail, 2),
        }

    park = {**park_summary(df0, PERIOD0), **park_summary(df1, PERIOD1)}
    park["Performance_Gap_%"] = round(park[f"Performance_%_{PERIOD1}"] - park[f"Performance_%_{PERIOD0}"], 2)
    park["Availability_Gap_%"] = round(park[f"Availability_%_{PERIOD1}"] - park[f"Availability_%_{PERIOD0}"], 2)
    park["Avg_UPH_Gap"] = round(park[f"Avg_UPH_{PERIOD1}"] - park[f"Avg_UPH_{PERIOD0}"], 2)
    park["Avg_OEE_Gap"] = round(park[f"Avg_OEE_{PERIOD1}"] - park[f"Avg_OEE_{PERIOD0}"], 2)
    park_df = pd.DataFrame([park])

    return machine_df, park_df


# ============================================================
# STEP 2 - LOG DECOMPOSITION (mixed-sign, works for one row or many)
# ============================================================

def step2_log_decomposition(df):
    out = df.copy()
    ln_A, ln_P, ln_total = [], [], []
    share_A, share_P, dominant = [], [], []

    for _, r in out.iterrows():
        A0 = r[f"Availability_%_{PERIOD0}"] / 100
        A1 = r[f"Availability_%_{PERIOD1}"] / 100
        P0 = r[f"Performance_%_{PERIOD0}"] / 100
        P1 = r[f"Performance_%_{PERIOD1}"] / 100

        if 0 in (A0, A1, P0, P1) or pd.isna(A0) or pd.isna(A1) or pd.isna(P0) or pd.isna(P1):
            ln_A.append(None); ln_P.append(None); ln_total.append(None)
            share_A.append(None); share_P.append(None); dominant.append(None)
            continue

        a = np.log(A1 / A0)
        p = np.log(P1 / P0)
        terms = {"Availability": a, "Performance": p}
        loss = {k: v for k, v in terms.items() if v < 0}
        offset = {k: v for k, v in terms.items() if v >= 0}
        loss_sum, offset_sum = sum(loss.values()), sum(offset.values())

        def share(term):
            group = loss_sum if term < 0 else offset_sum
            return round(term / group * 100, 2) if group else 0.0

        ln_A.append(round(a, 6)); ln_P.append(round(p, 6)); ln_total.append(round(a + p, 6))
        share_A.append(share(a)); share_P.append(share(p))
        dominant.append(min(loss, key=loss.get) if loss else None)

    out["ln_Availability"] = ln_A
    out["ln_Performance"] = ln_P
    out["ln_Total"] = ln_total
    out["OEE_Change_%"] = [round((np.exp(x) - 1) * 100, 2) if x is not None else None for x in ln_total]
    out["Availability_Share_%"] = share_A
    out["Performance_Share_%"] = share_P
    out["Dominant_Loss_Factor"] = dominant
    return out


# ============================================================
# STEP 3 - SIGNIFICANCE (z-score)
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


def randomness_label(perf_z, avail_z):
    perf_unlikely = perf_z is not None and not pd.isna(perf_z) and abs(perf_z) >= Z_SIGNIFICANT
    avail_unlikely = avail_z is not None and not pd.isna(avail_z) and abs(avail_z) >= Z_SIGNIFICANT
    if perf_unlikely and avail_unlikely:
        return "BOTH_UNLIKELY_RANDOM"
    if perf_unlikely:
        return "PERFORMANCE_UNLIKELY_RANDOM"
    if avail_unlikely:
        return "AVAILABILITY_UNLIKELY_RANDOM"
    return "COMPATIBLE_WITH_RANDOM_VARIATION"


def step3_significance(machine_df, hist_std_p=None, hist_std_a=None):
    """
    Cross-machine z-score (proxy, see module docstring) by default.
    Pass hist_std_p / hist_std_a (in percentage points, from a real
    multi-month log) to switch to a true historical significance test
    for the PARK-level gap instead of the cross-machine proxy.
    """
    out = machine_df.copy()
    stats = {}
    for label, col in [("Performance", "Performance_Gap_%"), ("Availability", "Availability_Gap_%")]:
        mean = out[col].mean()
        std = out[col].std(ddof=1)
        if not std or np.isnan(std):
            std = 1.0
        out[f"{label}_ZScore"] = (out[col] - mean) / std
        stats[label] = {"mean": mean, "std": std}

    out["Performance_Significance"] = out["Performance_ZScore"].apply(significance_label)
    out["Availability_Significance"] = out["Availability_ZScore"].apply(significance_label)
    out["Randomness_Assessment"] = out.apply(
        lambda r: randomness_label(r["Performance_ZScore"], r["Availability_ZScore"]), axis=1
    )

    park_z = {}
    if hist_std_p is not None:
        gap_p = out["Performance_Gap_%"].mean()  # park gap approximated by mean of machine gaps
        park_z["Performance"] = gap_p / hist_std_p
    if hist_std_a is not None:
        gap_a = out["Availability_Gap_%"].mean()
        park_z["Availability"] = gap_a / hist_std_a

    return out, stats, park_z


# ============================================================
# STEP 4 - COUNTERFACTUAL (bug-fixed: SIGNED comparison, no abs())
# ============================================================

def step4_counterfactual(df):
    out = df.copy()
    gap_A_l, gap_P_l = [], []
    recov_A_l, recov_P_l, gain_A_l, gain_P_l = [], [], [], []
    root_l, contrib_l = [], []

    has_uph = f"Avg_UPH_{PERIOD0}" in out.columns

    for _, r in out.iterrows():
        A0 = r[f"Availability_%_{PERIOD0}"] / 100
        A1 = r[f"Availability_%_{PERIOD1}"] / 100
        P0 = r[f"Performance_%_{PERIOD0}"] / 100
        P1 = r[f"Performance_%_{PERIOD1}"] / 100

        if 0 in (A0, A1, P0, P1) or pd.isna(A0) or pd.isna(A1) or pd.isna(P0) or pd.isna(P1):
            gap_A_l.append(None); gap_P_l.append(None)
            recov_A_l.append(None); recov_P_l.append(None)
            gain_A_l.append(None); gain_P_l.append(None)
            root_l.append(None); contrib_l.append(None)
            continue

        OEE0, OEE1 = A0 * P0, A1 * P1
        gap = OEE0 - OEE1
        OEE_cf_A = A0 * P1
        OEE_cf_P = A1 * P0

        if gap != 0:
            gap_closed_A = (OEE_cf_A - OEE1) / gap * 100
            gap_closed_P = (OEE_cf_P - OEE1) / gap * 100
        else:
            gap_closed_A = gap_closed_P = 0.0

        gap_A_l.append(round(gap_closed_A, 2))
        gap_P_l.append(round(gap_closed_P, 2))

        if has_uph:
            UPH0, UPH1 = r[f"Avg_UPH_{PERIOD0}"], r[f"Avg_UPH_{PERIOD1}"]
            recov_A = UPH1 + (UPH0 - UPH1) * (gap_closed_A / 100)
            recov_P = UPH1 + (UPH0 - UPH1) * (gap_closed_P / 100)
            recov_A_l.append(round(recov_A, 2)); recov_P_l.append(round(recov_P, 2))
            gain_A_l.append(round(recov_A - UPH1, 2)); gain_P_l.append(round(recov_P - UPH1, 2))

        # SIGNED comparison - never abs(). A factor that already improved
        # (negative gap_closed) must never outrank a real loss driver.
        if gap_closed_A >= gap_closed_P:
            root, contribution = "Availability", gap_closed_A
        else:
            root, contribution = "Performance", gap_closed_P
        root_l.append(root)
        contrib_l.append(round(contribution, 2))

    out["Gap_Closed_Availability_%"] = gap_A_l
    out["Gap_Closed_Performance_%"] = gap_P_l
    if has_uph:
        out["Recovered_UPH_Availability"] = recov_A_l
        out["Recovered_UPH_Performance"] = recov_P_l
        out["UPH_Gain_Availability"] = gain_A_l
        out["UPH_Gain_Performance"] = gain_P_l
    out["Root_Cause"] = root_l
    out["Root_Cause_Contribution_%"] = contrib_l
    return out


def verdict_for(row, has_zscore):
    root = row["Root_Cause"]
    if root is None or pd.isna(root):
        return "SKIPPED_ZERO_BASELINE"
    contribution = row["Root_Cause_Contribution_%"]
    if has_zscore:
        z = abs(row[f"{root}_ZScore"])
        if z >= Z_SIGNIFICANT and contribution >= DOMINANCE_GAP_CLOSED_MIN:
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

def build_report(machine_final, park_final, zstats, park_z, park_availability, reason_pivot):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    lines = []

    def h1(title):
        lines.append(title); lines.append("=" * 140); lines.append("")

    def h2(title):
        lines.append(title); lines.append("-" * 140)

    h1("MASTER UPH / OEE ROOT-CAUSE REPORT")
    lines.append(f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Periods   : {PERIOD0} -> {PERIOD1}")
    lines.append(f"Machines analyzed : {len(machine_final)}")
    lines.append("Assumption: Quality = 100% throughout (confirmed by user); "
                  "OEE = Availability x Performance.")
    lines.append("")

    park = park_final.iloc[0]

    # -------------------- Q1 --------------------
    h2("Q1 - DID THE PARK'S OEE ACTUALLY CHANGE, OR COULD THIS BE NOISE?")
    lines.append(f"Park OEE change: {park['OEE_Change_%']:.2f}%  "
                 f"(Availability {park[f'Availability_%_{PERIOD0}']:.2f}% -> {park[f'Availability_%_{PERIOD1}']:.2f}%, "
                 f"Performance {park[f'Performance_%_{PERIOD0}']:.2f}% -> {park[f'Performance_%_{PERIOD1}']:.2f}%)")
    if park_z:
        for k, v in park_z.items():
            lines.append(f"Park-level {k} z-score (vs supplied historical std): {v:+.2f} "
                          f"-> {significance_label(v)}")
    else:
        lines.append("No historical std supplied for the park-level gap -> park-level significance "
                      "not computed. Per-machine significance below uses the cross-machine spread as a proxy "
                      "(\"is this machine unusual vs its peers this month\", not vs its own history).")
    n_sig_perf = (machine_final["Performance_ZScore"].abs() >= Z_SIGNIFICANT).sum()
    n_sig_avail = (machine_final["Availability_ZScore"].abs() >= Z_SIGNIFICANT).sum()
    lines.append(f"Machines with |z|>={Z_SIGNIFICANT:.0f} on Performance : {n_sig_perf} / {len(machine_final)}")
    lines.append(f"Machines with |z|>={Z_SIGNIFICANT:.0f} on Availability: {n_sig_avail} / {len(machine_final)}")
    lines.append("")

    # -------------------- Q2 --------------------
    h2("Q2 - WHICH FACTOR EXPLAINS MOST OF THE CHANGE: AVAILABILITY OR PERFORMANCE?")
    lines.append(f"Park-level dominant loss factor: {park['Dominant_Loss_Factor']} "
                 f"(ln_Availability={park['ln_Availability']:+.4f}, ln_Performance={park['ln_Performance']:+.4f})")
    dom_counts = machine_final["Dominant_Loss_Factor"].value_counts(dropna=True)
    lines.append("Per-machine dominant loss factor counts:")
    lines.append(dom_counts.to_string() if len(dom_counts) else "(no machine has a loss factor)")
    lines.append("")

    # -------------------- Q3 --------------------
    h2("Q3 - IF WE RESTORED ONE FACTOR TO JULY'S LEVEL, HOW MUCH WOULD WE RECOVER?")
    lines.append(f"Park-level: restoring Availability alone -> gap closed {park['Gap_Closed_Availability_%']:.1f}%; "
                 f"restoring Performance alone -> gap closed {park['Gap_Closed_Performance_%']:.1f}%.")
    lines.append("(A value outside 0-100%, or negative, means the OTHER factor moved strongly enough "
                 "to distort the single-factor overshoot/undershoot - read as directional, not literal share.)")
    root_counts = machine_final["Root_Cause"].value_counts(dropna=True)
    lines.append("Per-machine root cause counts (signed comparison, never abs()):")
    lines.append(root_counts.to_string() if len(root_counts) else "(no machine scored)")
    lines.append("")

    # -------------------- Q4 --------------------
    h2("Q4 - WHICH SPECIFIC DOWNTIME REASONS EXPLAIN THE AVAILABILITY CHANGE? (real log, park-level)")
    if reason_pivot is not None and "Downtime_Gap_Hours" in reason_pivot.columns:
        lines.append("Top 10 reason codes by increase in downtime hours (Jul -> Aug):")
        lines.append(reason_pivot.head(10).to_string(index=False))
    else:
        lines.append("Downtime log not available/processed - Step 5 skipped.")
    lines.append("")

    # -------------------- Q5 --------------------
    h2("Q5 - WHICH MACHINES ARE THE OUTLIERS DRIVING THE PARK RESULT?")
    top_perf = machine_final.sort_values("Performance_Share_%", ascending=False).head(5)
    top_avail = machine_final.sort_values("Availability_Share_%", ascending=False).head(5)
    lines.append("Top 5 machines by Performance loss share:")
    lines.append(top_perf[["MACHINE_ID", "Performance_Share_%", "Performance_ZScore", "Verdict"]].to_string(index=False))
    lines.append("")
    lines.append("Top 5 machines by Availability loss share:")
    lines.append(top_avail[["MACHINE_ID", "Availability_Share_%", "Availability_ZScore", "Verdict"]].to_string(index=False))
    lines.append("")

    # -------------------- Synthesis --------------------
    h2("SYNTHESIS")
    verdict_counts = machine_final["Verdict"].value_counts()
    confirmed = [v for v in verdict_counts.index if v.startswith("CONFIRMED_ROOT_CAUSE")]
    if confirmed:
        lines.append(f"{confirmed[0]}: statistically significant AND dominant in the counterfactual "
                     f"on {verdict_counts[confirmed[0]]} machine(s).")
    else:
        lines.append("No machine reaches both the significance and dominance bar simultaneously; "
                     "treat the park-level dominant factor above as the leading hypothesis, not a confirmed cause.")
    lines.append("")

    # -------------------- Supporting tables --------------------
    h1("SUPPORTING TABLES")

    h2("TABLE 1 - PARK-LEVEL KPI (STEP 1-4)")
    lines.append(park_final.to_string(index=False))
    lines.append("")

    h2("TABLE 2 - Z-SCORE STATS USED (STEP 3, cross-machine proxy)")
    lines.append(pd.DataFrame({
        "Metric": list(zstats.keys()),
        "Mean Gap": [round(v["mean"], 2) for v in zstats.values()],
        "Std Dev": [round(v["std"], 2) for v in zstats.values()],
    }).to_string(index=False))
    lines.append("")

    h2("TABLE 3 - PER-MACHINE FULL RESULTS (STEPS 1-4)")
    cols = ["MACHINE_ID", "Performance_Gap_%", "Availability_Gap_%",
            "Performance_ZScore", "Availability_ZScore",
            "Dominant_Loss_Factor", "Gap_Closed_Availability_%", "Gap_Closed_Performance_%",
            "Root_Cause", "Root_Cause_Contribution_%", "Verdict"]
    cols = [c for c in cols if c in machine_final.columns]
    lines.append(machine_final[cols].to_string(index=False))
    lines.append("")

    h2("TABLE 4 - PARK AVAILABILITY FROM REAL DOWNTIME LOG (STEP 5)")
    if park_availability is not None:
        lines.append(park_availability.to_string(index=False))
    else:
        lines.append("(not available)")
    lines.append("")

    h2("TABLE 5 - FULL REASON-CODE PARETO (STEP 5)")
    if reason_pivot is not None:
        lines.append(reason_pivot.to_string(index=False))
    else:
        lines.append("(not available)")

    return "\n".join(lines) + "\n"


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # STEP 1
    machine_df, park_df = step1_kpi(MACHINE_XLSX, SHEET0, SHEET1, UPH_THEORETICAL)

    # STEP 2
    machine_df = step2_log_decomposition(machine_df)
    park_df = step2_log_decomposition(park_df)

    # STEP 3
    machine_df, zstats, park_z = step3_significance(machine_df)

    # STEP 4
    machine_df = step4_counterfactual(machine_df)
    park_df = step4_counterfactual(park_df)

    machine_df["Verdict"] = machine_df.apply(lambda r: verdict_for(r, has_zscore=True), axis=1)
    park_df["Verdict"] = park_df.apply(lambda r: verdict_for(r, has_zscore=False), axis=1)

    # STEP 5 (independent source - real downtime log)
    park_availability = reason_summary = reason_pivot = None
    try:
        park_availability, reason_summary, reason_pivot = step5_pareto(
            DOWNTIME_LOG, DOWNTIME_ENCODING, DOWNTIME_SEP,
            FISCAL_YEAR, MONTH0_NAME, MONTH1_NAME, MONTH0_DAYS, MONTH1_DAYS
        )
    except FileNotFoundError:
        print(f"[WARN] Downtime log not found at {DOWNTIME_LOG} - Step 5 (Pareto) skipped.")

    # EXPORT
    machine_df.to_csv(os.path.join(OUTPUT_DIR, "machine_level_analysis.csv"), index=False)
    park_df.to_csv(os.path.join(OUTPUT_DIR, "park_level_analysis.csv"), index=False)
    if reason_pivot is not None:
        reason_pivot.to_csv(os.path.join(OUTPUT_DIR, "downtime_pareto.csv"), index=False)

    report_text = build_report(machine_df, park_df, zstats, park_z, park_availability, reason_pivot)
    report_path = os.path.join(OUTPUT_DIR, "master_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print("Generated files:")
    print(f"- {os.path.join(OUTPUT_DIR, 'machine_level_analysis.csv')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'park_level_analysis.csv')}")
    if reason_pivot is not None:
        print(f"- {os.path.join(OUTPUT_DIR, 'downtime_pareto.csv')}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
