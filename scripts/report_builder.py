"""
report_builder.py — turns the phase1/phase2 results into the final
Q&A-oriented text report with supporting tables. No computation happens
here; this module only formats what the phases already produced.
"""

from datetime import datetime, timezone

import pandas as pd

import config


def fmt(x, spec=".2f"):
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else format(x, spec)


def significance_table(sig, metrics):
    return pd.DataFrame([
        {"Metric": m, "Observed_Diff": sig[m]["observed_diff"], "Historical_STD": sig[m]["historical_std"],
         "Z_Score": sig[m]["z"], "Label": sig[m]["label"]}
        for m in metrics
    ])


def build(s0, s1, p1, p2, phase2_relevant):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    lines = []

    def h1(title):
        lines.extend([title, "=" * 140, ""])

    def h2(title):
        lines.extend([title, "-" * 140])

    h1("MASTER UPH / AVAILABILITY ROOT-CAUSE REPORT")
    lines.append(f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Periods   : {config.PERIOD0} -> {config.PERIOD1}")
    lines.append(f"{config.PERIOD0}: {s0['N_Records']} records / {s0['N_Machines']} machines   |   "
                 f"{config.PERIOD1}: {s1['N_Records']} records / {s1['N_Machines']} machines")
    lines.append("Source 'OEE' column = Availability. UPH used as-is. MTTR = F2/TEUD (temporary).")
    lines.append(f"Significant if |z| >= {config.Z_SIGNIFICANT}; dominant if gap/share >= "
                 f"{config.DOMINANCE_GAP_CLOSED_MIN:.0f}%; CONFIRMED needs both, LIKELY needs dominance alone.")
    lines.append("")

    root1 = p1["cf"]["Root_Cause"]
    h1("EXECUTIVE SUMMARY")
    lines.append(f"-> Primary root cause (OEE drop): {root1}  [{p1['verdict']}]  "
                 f"({p1['cf']['Root_Cause_Contribution_%']:.1f}% of the gap)")
    if phase2_relevant:
        lines.append(f"-> Within Availability, the deeper driver is: {p2['decomp']['Root_Cause']}  [{p2['verdict']}]")
    else:
        lines.append("-> Phase 2 (MTTR/MTBF) is SECONDARY: Phase 1's root cause is Performance, "
                      "not Availability, and MTTR/MTBF only explain Availability losses.")
    lines.append("")

    h1("PHASE 1 - PERFORMANCE VS AVAILABILITY")
    h2("Q1 - Did the process actually change, or could this be random sample noise?")
    lines.append(f"OEE change: {p1['decomp']['OEE_Change_%']:.2f}%  "
                 f"(Availability {s0['Availability_%_mean']:.2f}% -> {s1['Availability_%_mean']:.2f}%, "
                 f"Performance {s0['Performance_%_mean']:.2f}% -> {s1['Performance_%_mean']:.2f}%)")
    lines.append(significance_table(p1["sig"], config.PHASE1_METRICS).to_string(index=False))
    lines.append(f"Randomness assessment: {p1['randomness']}")
    lines.append("")

    h2("Q2 - Which factor explains most of the change?")
    d = p1["decomp"]
    lines.append(f"Dominant loss factor: {d['Dominant_Loss_Factor']}  "
                 f"(Availability share {d['Availability_Share_%']:.1f}%, Performance share {d['Performance_Share_%']:.1f}%)")
    lines.append("")

    h2("Q3 - If we restored one factor to the baseline level, how much would we recover?")
    cf = p1["cf"]
    lines.append(f"Restore Availability only -> gap closed {cf['Gap_Closed_Availability_%']:.1f}%, "
                 f"UPH gain {cf['UPH_Gain_Availability']:+.1f}")
    lines.append(f"Restore Performance only  -> gap closed {cf['Gap_Closed_Performance_%']:.1f}%, "
                 f"UPH gain {cf['UPH_Gain_Performance']:+.1f}")
    lines.append(f"Root cause (signed comparison, never abs()): {cf['Root_Cause']} "
                 f"({cf['Root_Cause_Contribution_%']:.1f}% gap closed) -> Verdict: {p1['verdict']}")
    lines.append("")

    h1("PHASE 2 - AVAILABILITY DRIVER: MTTR VS MTBF" +
       ("" if phase2_relevant else "  [SECONDARY - Phase 1 root cause is Performance]"))
    h2("Q4 - Is the Availability loss explained more by MTTR or by MTBF?")
    d2 = p2["decomp"]
    lines.append(f"Real Availability (from data, unchanged): {s0['Availability_%_mean']:.2f}% -> "
                 f"{s1['Availability_%_mean']:.2f}% (see Phase 1 for the full analysis of this change)")
    lines.append(f"MTTR: {fmt(d2['MTTR_Jul'], '.4f')} -> {fmt(d2['MTTR_Aug'], '.4f')}   |   "
                 f"MTBF: {fmt(d2['MTBF_Jul'], '.4f')} -> {fmt(d2['MTBF_Aug'], '.4f')}")
    lines.append(f"Root cause via rho=MTTR/MTBF log decomposition: {d2['Root_Cause']} "
                 f"({fmt(d2['Root_Cause_Contribution_%'])}% of its group)")
    lines.append(d2["Note"])
    lines.append("")

    h2("Q5 - Are the MTTR/MTBF shifts large enough to be unlikely random?")
    lines.append(significance_table(p2["sig"], config.PHASE2_METRICS).to_string(index=False))
    lines.append(f"Randomness assessment: {p2['randomness']}   |   Verdict: {p2['verdict']}")
    lines.append("")

    h1("SUPPORTING TABLES")
    h2("Table 1 - Period summary")
    summary = pd.DataFrame([{
        "Period": s["Period"], "N_Records": s["N_Records"], "N_Machines": s["N_Machines"],
        **{f"{col}_mean": round(s.get(f"{col}_mean", float("nan")), 4) for col in config.STAT_COLUMNS}
    } for s in (s0, s1)])
    lines.append(summary.to_string(index=False))
    lines.append("")

    h2("Table 2 - Phase 1 log decomposition")
    lines.append(pd.DataFrame([p1["decomp"]]).to_string(index=False))
    lines.append("")

    h2("Table 3 - Phase 1 counterfactual")
    lines.append(pd.DataFrame([p1["cf"]]).to_string(index=False))
    lines.append("")

    h2("Table 4 - Phase 2 rho decomposition (MTTR vs MTBF)")
    lines.append(pd.DataFrame([{k: v for k, v in p2["decomp"].items() if k != "Note"}]).to_string(index=False))

    return "\n".join(lines) + "\n"
