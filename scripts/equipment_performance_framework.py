"""
EQUIPMENT PERFORMANCE ASSESSMENT FRAMEWORK — single-file implementation
====================================================================
Implements the framework described in Framework.txt end to end:

  STEP 1  Build the data foundation (mock: Event, Production, Operator,
          Reference tables)
  STEP 2  Analysis levels: Equipment / Operator / Operator group
  STEP 3  KPI dimensions (throughput, stability, utilization, manpower,
          interference, balance)
  STEP 4  Base KPIs (Appendix A formulas, Phase 1 MVP list)
  STEP 5  Composite indices: EPI, OGPI, CSEPI (Appendix A.7)
  STEP 6  Pattern-based interpretation (framework Step 6 table +
          Appendix B thresholds)

Then renders one self-contained HTML report: framework walkthrough,
KPI map (equipment x KPI, colour-coded), SVG radar chart per equipment
(no external CDN - works fully offline), and the reference
pattern/interpretation table.

Run:
    python equipment_performance_framework.py
Output:
    Output/equipment_performance_report.html
"""

import os
import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

RNG_SEED = 42
SHIFT_START = datetime(2026, 9, 1, 6, 0, 0)
SHIFT_HOURS = 8.0
SCHEDULED_TIME_HOURS = SHIFT_HOURS

MAJOR_EVENT_THRESHOLD_S = 120  # >=120s = major interruption, <120s = micro
DPI_DELTA_MINUTES = 10          # propagation window for Delay Propagation Index

OUTPUT_DIR = "Output"
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "equipment_performance_report.html")

# --- Operator groups: operator -> list of machines it supervises this shift
OPERATOR_GROUPS = {
    "OP-1": ["M01", "M02", "M03"],   # overloaded group (3 machines)
    "OP-2": ["M04", "M05"],          # well-staffed group (2 machines)
}
PLANNED_MMR = {"OP-1": 1 / 2.0, "OP-2": 1 / 2.0}   # operators needed per machine -> PSC = 1/MMR

REFERENCE_TUPH = {"M01": 120, "M02": 120, "M03": 120, "M04": 100, "M05": 100}
TARGET_MTBI_MIN = {"M01": 45, "M02": 45, "M03": 45, "M04": 60, "M05": 60}

# --- Composite index weights (Appendix A.7) ---
EPI_WEIGHTS = {"throughput": 0.30, "stability": 0.30, "utilization": 0.20, "operator": 0.20}
STABILITY_WEIGHTS = {"mtbi": 0.6, "mir": 0.4}
OGPI_WEIGHTS = {"epi_group": 0.40, "load": 0.20, "interference": 0.25, "balance": 0.15}
CSEPI_WEIGHTS = {"epi": 0.60, "ogpi": 0.40}   # "simpler first version"

# --- Score model limits (engineering assumptions, Appendix A.7 / B) ---
MIR_LIMIT = 6.0          # micro-interruptions per hour treated as "fully bad" beyond this
UR_TARGET = 0.90
OWTR_LIMIT = 0.30
OLR_CRIT = 1.5
EII_LIMIT = 0.30
DPI_LIMIT = 0.20
EPSILON = 1e-6

# --- Appendix B thresholds: (green_op, threshold_green, threshold_amber) ---
# op 'ge' = green when value >= threshold; 'le' = green when value <= threshold
THRESHOLDS = {
    "UPH Adherence": ("ge", 0.95, 0.85),
    "AMTBI": ("ge", 1.00, 0.80),
    "OLR": ("le", 1.00, 1.15),
    "OWTR": ("le", 0.10, 0.25),
    "EII": ("le", 0.15, 0.30),
    "DPI": ("le", 0.10, 0.20),
    "WBI": ("ge", 0.85, 0.70),
    "TBR": ("ge", 0.85, 0.70),
}


# ============================================================
# STEP 1 — MOCK DATA FOUNDATION
# ============================================================

def machine_to_operator(machine):
    for op, machines in OPERATOR_GROUPS.items():
        if machine in machines:
            return op
    raise ValueError(f"Machine {machine} not assigned to any operator group")


def generate_events(machine, rng, overload_bias=False):
    """Generate a shift's worth of interruption events for one machine.
    overload_bias=True simulates a machine suffering from operator
    unavailability (longer operator-dependent recovery, more events)."""
    events = []
    t = SHIFT_START
    end = SHIFT_START + timedelta(hours=SHIFT_HOURS)

    major_rate_per_hour = rng.uniform(1.2, 2.2) if overload_bias else rng.uniform(0.6, 1.3)
    micro_rate_per_hour = rng.uniform(3.0, 5.0)

    while t < end:
        gap_min = rng.expovariate(1.0 / 18.0)  # avg 18 min between events
        t = t + timedelta(minutes=gap_min)
        if t >= end:
            break
        is_major = rng.random() < (major_rate_per_hour / (major_rate_per_hour + micro_rate_per_hour))
        if is_major:
            base_duration = rng.uniform(150, 420)
            operator_dependent = rng.random() < (0.75 if overload_bias else 0.35)
            if operator_dependent and overload_bias:
                base_duration *= rng.uniform(1.4, 2.2)  # waiting for a busy operator
            duration = base_duration
        else:
            duration = rng.uniform(20, 115)
            operator_dependent = False

        events.append({
            "Equipment": machine, "Operator": machine_to_operator(machine),
            "EventStart": t, "Duration_s": duration,
            "IsMajor": duration >= MAJOR_EVENT_THRESHOLD_S,
            "OperatorDependent": operator_dependent,
        })
        t = t + timedelta(seconds=duration)

    return pd.DataFrame(events)


def generate_production_trace(machine, events_df, rng):
    """Build track-in/track-out segments so ProductionTime_e,s = shift time
    minus total interruption time, and quantity is consistent with AUPH."""
    total_int_s = events_df["Duration_s"].sum() if len(events_df) else 0.0
    production_time_h = max(SHIFT_HOURS - total_int_s / 3600.0, 0.5)

    true_rate = REFERENCE_TUPH[machine] * rng.uniform(0.80, 1.02)
    quantity = int(round(true_rate * production_time_h))

    return {
        "Equipment": machine, "ProductionTime_h": production_time_h,
        "Quantity": quantity, "TrackIn": SHIFT_START, "TrackOut": SHIFT_START + timedelta(hours=SHIFT_HOURS),
    }


def build_step1_tables():
    rng = random.Random(RNG_SEED)
    overload_map = {"M01": False, "M02": False, "M03": True, "M04": False, "M05": False}

    event_frames, prod_rows = [], []
    for machine, overloaded in overload_map.items():
        ev = generate_events(machine, rng, overload_bias=overloaded)
        event_frames.append(ev)
        prod_rows.append(generate_production_trace(machine, ev, rng))

    events_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    production_df = pd.DataFrame(prod_rows)

    operator_assignment_df = pd.DataFrame([
        {"Operator": op, "Equipment": m} for op, machines in OPERATOR_GROUPS.items() for m in machines
    ])

    reference_df = pd.DataFrame([
        {"Equipment": m, "TUPH": REFERENCE_TUPH[m], "TargetMTBI_min": TARGET_MTBI_MIN[m],
         "Operator": machine_to_operator(m), "PlannedMMR": PLANNED_MMR[machine_to_operator(m)]}
        for m in REFERENCE_TUPH
    ])

    return events_df, production_df, operator_assignment_df, reference_df


# ============================================================
# STEP 4 — BASE KPIs (equipment level)
# ============================================================

def equipment_kpis(machine, events_df, production_df, reference_df):
    ev = events_df[events_df["Equipment"] == machine]
    prod = production_df[production_df["Equipment"] == machine].iloc[0]
    ref = reference_df[reference_df["Equipment"] == machine].iloc[0]

    production_time_h = prod["ProductionTime_h"]
    quantity = prod["Quantity"]

    major = ev[ev["IsMajor"]]
    micro = ev[~ev["IsMajor"]]
    int_count = len(major)
    occ_count = len(micro)
    all_event_count = int_count + occ_count
    int_dur_s = major["Duration_s"].sum()
    operator_dependent_dur_s = major[major["OperatorDependent"]]["Duration_s"].sum()

    # --- 4.1 Throughput ---
    auph = quantity / production_time_h if production_time_h > 0 else 0.0
    tuph = ref["TUPH"]
    uph_adh = auph / tuph if tuph else 0.0

    # --- 4.2 Stability ---
    mtbi_major_min = (production_time_h * 60.0) / int_count if int_count else float("inf")
    mtbi_all_min = (production_time_h * 60.0) / all_event_count if all_event_count else float("inf")
    target_mtbi = ref["TargetMTBI_min"]
    amtbi = mtbi_major_min / target_mtbi if target_mtbi else 0.0
    major_int_rate = int_count / production_time_h if production_time_h > 0 else 0.0
    mir = occ_count / production_time_h if production_time_h > 0 else 0.0
    mrt_min = (int_dur_s / 60.0) / int_count if int_count else 0.0
    spr = mtbi_all_min / mtbi_major_min if mtbi_major_min not in (0, float("inf")) else 0.0

    # --- 4.3 Utilization ---
    ur = production_time_h / SCHEDULED_TIME_HOURS
    oer = ur * uph_adh

    # --- 4.5 (equipment-facing) Operator Waiting Time Ratio ---
    owtr = operator_dependent_dur_s / int_dur_s if int_dur_s > 0 else 0.0

    return {
        "Equipment": machine, "Operator": ref["Operator"],
        "AUPH": auph, "TUPH": tuph, "UPH Adherence": uph_adh, "Quantity": quantity,
        "Major MTBI (min)": mtbi_major_min, "All-event MTBI (min)": mtbi_all_min, "AMTBI": amtbi,
        "Major Int Rate (/h)": major_int_rate, "MIR (/h)": mir,
        "Total Int Duration (min)": int_dur_s / 60.0, "MRT (min)": mrt_min, "SPR": spr,
        "Utilization Rate": ur, "OER": oer, "OWTR": owtr,
        "ProductionTime_h": production_time_h,
        "IntCount": int_count, "OccCount": occ_count, "IntDur_s": int_dur_s,
        "OperatorDependentDur_s": operator_dependent_dur_s,
    }


# ============================================================
# STEP 4.4 — MANPOWER KPIs (operator level)
# ============================================================

def operator_kpis(operator, operator_assignment_df):
    machines = operator_assignment_df[operator_assignment_df["Operator"] == operator]["Equipment"].tolist()
    psc = 1.0 / PLANNED_MMR[operator]
    asc = len(machines)
    olr = asc / psc if psc else 0.0
    das = 1.0 / asc if asc else 0.0
    return {"Operator": operator, "PSC": psc, "ASC": asc, "OLR": olr, "DAS": das, "Machines": machines}


# ============================================================
# STEP 4.5 — INTER-EQUIPMENT (group level, real overlap detection)
# ============================================================

def _intervals(df):
    return [(row["EventStart"], row["EventStart"] + timedelta(seconds=row["Duration_s"]))
            for _, row in df.iterrows()]


def _overlap_seconds(intervals_a, intervals_b):
    total = 0.0
    for a_start, a_end in intervals_a:
        for b_start, b_end in intervals_b:
            latest_start = max(a_start, b_start)
            earliest_end = min(a_end, b_end)
            if latest_start < earliest_end:
                total += (earliest_end - latest_start).total_seconds()
    return total


def group_kpis(operator, machines, events_df, equip_kpi_df):
    group_ev = events_df[events_df["Equipment"].isin(machines)]
    major_ev = group_ev[group_ev["IsMajor"]]

    # --- Equipment Interference Index: overlap time between DIFFERENT
    #     machines' major interruptions, over total group interrupt time ---
    total_group_int_s = major_ev["Duration_s"].sum()
    overlap_s = 0.0
    for i, m1 in enumerate(machines):
        for m2 in machines[i + 1:]:
            iv1 = _intervals(major_ev[major_ev["Equipment"] == m1])
            iv2 = _intervals(major_ev[major_ev["Equipment"] == m2])
            overlap_s += _overlap_seconds(iv1, iv2)
    eii = overlap_s / total_group_int_s if total_group_int_s > 0 else 0.0

    # --- Delay Propagation Index: fraction of group major events followed
    #     by an event on ANOTHER machine within DPI_DELTA_MINUTES ---
    starts_by_machine = {m: sorted(major_ev[major_ev["Equipment"] == m]["EventStart"].tolist()) for m in machines}
    propagated = 0
    total_major_events = len(major_ev)
    for _, row in major_ev.iterrows():
        window_end = row["EventStart"] + timedelta(minutes=DPI_DELTA_MINUTES)
        other_machines = [m for m in machines if m != row["Equipment"]]
        hit = any(any(row["EventStart"] < s <= window_end for s in starts_by_machine[m]) for m in other_machines)
        if hit:
            propagated += 1
    dpi = propagated / total_major_events if total_major_events > 0 else 0.0

    # --- Supervision Conflict Rate: simultaneous operator-requiring events,
    #     counted as overlapping OperatorDependent intervals across machines,
    #     normalized per shift hour ---
    dep_ev = major_ev[major_ev["OperatorDependent"]]
    conflict_events = 0
    for i, m1 in enumerate(machines):
        for m2 in machines[i + 1:]:
            iv1 = _intervals(dep_ev[dep_ev["Equipment"] == m1])
            iv2 = _intervals(dep_ev[dep_ev["Equipment"] == m2])
            for a_start, a_end in iv1:
                for b_start, b_end in iv2:
                    if max(a_start, b_start) < min(a_end, b_end):
                        conflict_events += 1
    scr = conflict_events / SHIFT_HOURS

    # --- Workload Balance Index & Throughput Balance Ratio ---
    sub = equip_kpi_df[equip_kpi_df["Equipment"].isin(machines)]
    x = (0.4 * sub["IntCount"] + 0.2 * sub["OccCount"]
         + 0.3 * sub["OperatorDependentDur_s"] + 0.1 * sub["Quantity"] / sub["Quantity"].max())
    wbi = 1 - (x.std(ddof=0) / (x.mean() + EPSILON))
    tbr = 1 - (sub["UPH Adherence"].std(ddof=0) / (sub["UPH Adherence"].mean() + EPSILON))

    # --- Cluster Stability Index ---
    csi = sub["ProductionTime_h"].sum() / (sub["ProductionTime_h"].sum() + sub["IntDur_s"].sum() / 3600.0)

    return {
        "Operator": operator, "EII": eii, "DPI": dpi, "SCR": scr,
        "WBI": max(0.0, wbi), "TBR": max(0.0, tbr), "CSI": csi,
    }


def equipment_owtr_group_view(equip_kpi_df):
    """OWTR is computed per equipment (Step 4.5 formula is equipment-level
    despite living in the inter-equipment section); already in equip_kpis."""
    return equip_kpi_df[["Equipment", "Operator", "OWTR"]]


# ============================================================
# STEP 5 — COMPOSITE INDICES
# ============================================================

def equipment_performance_index(row):
    s_throughput = 100 * min(1.0, row["UPH Adherence"] / 1.00)
    s_mtbi = 100 * min(1.0, row["AMTBI"])
    s_mir = 100 * max(0.0, 1 - row["MIR (/h)"] / MIR_LIMIT)
    s_stability = STABILITY_WEIGHTS["mtbi"] * s_mtbi + STABILITY_WEIGHTS["mir"] * s_mir
    s_utilization = 100 * min(1.0, row["Utilization Rate"] / UR_TARGET)
    s_operator = 100 * (1 - min(1.0, row["OWTR"] / OWTR_LIMIT))

    epi = (EPI_WEIGHTS["throughput"] * s_throughput + EPI_WEIGHTS["stability"] * s_stability
           + EPI_WEIGHTS["utilization"] * s_utilization + EPI_WEIGHTS["operator"] * s_operator)

    return {
        "S_throughput": s_throughput, "S_stability": s_stability, "S_mtbi": s_mtbi, "S_mir": s_mir,
        "S_utilization": s_utilization, "S_operator": s_operator, "EPI": epi,
    }


def operator_group_performance_index(op_kpi, grp_kpi, epi_values):
    s_epi_group = 0.8 * np.mean(epi_values) + 0.2 * np.min(epi_values)

    olr = op_kpi["OLR"]
    s_load = 100.0 if olr <= 1 else 100 * max(0.0, 1 - (olr - 1) / (OLR_CRIT - 1))

    s_interference = 100 * (1 - 0.5 * grp_kpi["EII"] / EII_LIMIT - 0.5 * grp_kpi["DPI"] / DPI_LIMIT)
    s_interference = max(0.0, s_interference)

    s_balance = 100 * (grp_kpi["WBI"] + grp_kpi["TBR"]) / 2

    ogpi = (OGPI_WEIGHTS["epi_group"] * s_epi_group + OGPI_WEIGHTS["load"] * s_load
            + OGPI_WEIGHTS["interference"] * s_interference + OGPI_WEIGHTS["balance"] * s_balance)

    return {"S_epi_group": s_epi_group, "S_load": s_load, "S_interference": s_interference,
            "S_balance": s_balance, "OGPI": ogpi}


def composite_supervised_equipment_index(epi_all, ogpi_all):
    return CSEPI_WEIGHTS["epi"] * np.mean(epi_all) + CSEPI_WEIGHTS["ogpi"] * np.mean(ogpi_all)


# ============================================================
# STEP 6 — INTERPRETATION (pattern table + thresholds)
# ============================================================

PATTERN_TABLE = [
    ("Low UPH + low UR", "Loading / starvation / scheduling problem",
     lambda r: r["UPH Adherence"] < 0.85 and r["Utilization Rate"] < UR_TARGET * 0.85),
    ("Low UPH + poor MTBI / high MIR", "Machine or process instability",
     lambda r: r["UPH Adherence"] < 0.85 and (r["AMTBI"] < 0.80 or r["MIR (/h)"] > MIR_LIMIT * 0.7)),
    ("Low UPH + high OLR + high OWTR", "Manpower constraint",
     lambda r: r["UPH Adherence"] < 0.85 and r["OLR"] > 1.0 and r["OWTR"] > 0.25),
    ("Low UPH + high OLR + high OWTR + high EII/DPI/SCR", "Shared-operator interference",
     lambda r: (r["UPH Adherence"] < 0.85 and r["OLR"] > 1.0 and r["OWTR"] > 0.25
                and (r["EII"] > 0.30 or r["DPI"] > 0.20))),
    ("Good average UPH + weak WBI/TBR", "Imbalance inside operator group",
     lambda r: r["UPH Adherence"] >= 0.85 and (r["WBI"] < 0.70 or r["TBR"] < 0.70)),
    ("Good UPH + weak manpower/interference scores", "Performance achieved but not sustainably",
     lambda r: r["UPH Adherence"] >= 0.85 and (r["OLR"] > 1.0 or r["EII"] > 0.30 or r["DPI"] > 0.20)),
]


def interpret(row):
    matches = [(pattern, meaning) for pattern, meaning, cond in PATTERN_TABLE if cond(row)]
    if not matches:
        return "No red-flag pattern matched", "Performance within expected bounds"
    # Most specific / severe pattern first (interference > manpower > instability > loading)
    return matches[-1] if len(matches) > 1 else matches[0]


def threshold_status(kpi_name, value):
    if kpi_name not in THRESHOLDS or value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    op, green, amber = THRESHOLDS[kpi_name]
    if op == "ge":
        if value >= green:
            return "green"
        if value >= amber:
            return "amber"
        return "red"
    else:
        if value <= green:
            return "green"
        if value <= amber:
            return "amber"
        return "red"


# ============================================================
# RADAR CHART — pure inline SVG, no external dependency
# ============================================================

def radar_svg(labels, values, size=260, max_val=100, color="#b5541e"):
    n = len(labels)
    cx = cy = size / 2
    radius = size * 0.36
    angle_step = 2 * math.pi / n

    def point(i, val):
        angle = -math.pi / 2 + i * angle_step
        r = radius * max(0.0, min(1.0, val / max_val))
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    def axis_point(i, r):
        angle = -math.pi / 2 + i * angle_step
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    grid = ""
    for frac in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{axis_point(i, radius * frac)[0]:.1f},{axis_point(i, radius * frac)[1]:.1f}" for i in range(n))
        grid += f'<polygon points="{pts}" fill="none" stroke="#d8d0c4" stroke-width="1"/>'

    axes = ""
    labels_svg = ""
    for i, lab in enumerate(labels):
        x, y = axis_point(i, radius)
        axes += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#d8d0c4" stroke-width="1"/>'
        lx, ly = axis_point(i, radius * 1.28)
        anchor = "middle"
        if lx < cx - 5:
            anchor = "end"
        elif lx > cx + 5:
            anchor = "start"
        labels_svg += f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="10.5" text-anchor="{anchor}" fill="#6b6255">{lab}</text>'

    poly_pts = " ".join(f"{point(i, v)[0]:.1f},{point(i, v)[1]:.1f}" for i, v in enumerate(values))

    return f"""<svg viewBox="0 0 {size} {size}" width="100%" height="{size}">
      {grid}{axes}
      <polygon points="{poly_pts}" fill="{color}33" stroke="{color}" stroke-width="2"/>
      {labels_svg}
    </svg>"""


# ============================================================
# HTML REPORT
# ============================================================

def status_dot(status):
    color = {"green": "#1f7a52", "amber": "#c98a1a", "red": "#c0392b", "n/a": "#999"}[status]
    return f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{color};margin-right:5px;"></span>'


def build_html(equip_kpi_df, epi_df, op_kpi_df, grp_kpi_df, ogpi_df, csepi):
    css = """
    body{font-family:'IBM Plex Sans',system-ui,sans-serif;background:#f7f5f2;color:#1f1b16;margin:0;}
    .wrap{max-width:1080px;margin:0 auto;padding:28px 18px 80px;}
    h1{font-size:1.6rem;} h2{font-size:1.2rem;margin-top:36px;border-bottom:2px solid #e3ddd3;padding-bottom:6px;}
    .step{background:#fff;border:1px solid #e3ddd3;border-radius:12px;padding:16px 20px;margin:10px 0;}
    table{border-collapse:collapse;width:100%;font-size:0.85rem;margin:10px 0;}
    th,td{border:1px solid #e3ddd3;padding:6px 9px;text-align:right;} th:first-child,td:first-child{text-align:left;}
    th{background:#f0dcc8;}
    .radar-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;}
    .radar-card{background:#fff;border:1px solid #e3ddd3;border-radius:12px;padding:14px;text-align:center;}
    .interp{font-size:0.85rem;background:#f0dcc8;border-radius:8px;padding:8px 10px;margin-top:8px;}
    .csepi{font-size:2rem;font-weight:700;color:#b5541e;text-align:center;margin:10px 0;}
    """

    steps_html = """
    <div class="step"><b>Step 1 — Data foundation.</b> Event table (interruptions with timestamps),
    production trace (track-in/out, quantity), operator assignment, reference table (TUPH, target MTBI, planned MMR).
    Below, all four are mocked for one 8h shift, 2 operator groups, 5 machines.</div>
    <div class="step"><b>Step 2 — Analysis levels.</b> Equipment (machine), Operator (span of control),
    Operator group (cross-machine interaction), Site (roll-up). This report covers levels A-C.</div>
    <div class="step"><b>Step 3 — KPI dimensions.</b> Throughput, Stability, Utilization, Manpower coverage,
    Interference resistance, Balance/fairness — the 6 radar axes below.</div>
    <div class="step"><b>Step 4 — Base KPIs.</b> Computed per equipment/operator/group from the mock tables
    (Appendix A formulas) — see the KPI map.</div>
    <div class="step"><b>Step 5 — Composite indices.</b> Equipment Performance Index (EPI), Operator Group
    Performance Index (OGPI), Composite Supervised Equipment Performance Index (CSEPI).</div>
    <div class="step"><b>Step 6 — Interpretation.</b> Each equipment's KPI profile is matched against the
    reference pattern table to produce a plain-language diagnosis.</div>
    """

    # KPI map table
    kpi_cols = ["UPH Adherence", "AMTBI", "OWTR"]
    rows_html = ""
    for _, r in equip_kpi_df.iterrows():
        cells = "".join(f"<td>{status_dot(threshold_status(c, r[c]))}{r[c]:.2f}</td>" for c in kpi_cols)
        op_row = op_kpi_df[op_kpi_df["Operator"] == r["Operator"]].iloc[0]
        grp_row = grp_kpi_df[grp_kpi_df["Operator"] == r["Operator"]].iloc[0]
        rows_html += (f"<tr><td>{r['Equipment']}</td><td>{r['Operator']}</td>{cells}"
                      f"<td>{status_dot(threshold_status('OLR', op_row['OLR']))}{op_row['OLR']:.2f}</td>"
                      f"<td>{status_dot(threshold_status('EII', grp_row['EII']))}{grp_row['EII']:.2f}</td>"
                      f"<td>{status_dot(threshold_status('WBI', grp_row['WBI']))}{grp_row['WBI']:.2f}</td></tr>")

    kpi_map_html = f"""
    <table><tr><th>Equipment</th><th>Operator</th><th>UPH Adh.</th><th>AMTBI</th><th>OWTR</th>
    <th>OLR (op)</th><th>EII (group)</th><th>WBI (group)</th></tr>{rows_html}</table>
    """

    # Radar cards
    radar_html = ""
    for _, r in epi_df.iterrows():
        grp = grp_kpi_df[grp_kpi_df["Operator"] == r["Operator"]].iloc[0]
        labels = ["Throughput", "Stability", "Utilization", "Manpower", "Interference", "Balance"]
        values = [r["S_throughput"], r["S_stability"], r["S_utilization"], r["S_operator"],
                  100 * (1 - grp["EII"]), 100 * (grp["WBI"] + grp["TBR"]) / 2]
        pattern, meaning = interpret({**equip_kpi_df[equip_kpi_df["Equipment"] == r["Equipment"]].iloc[0].to_dict(),
                                       **op_kpi_df[op_kpi_df["Operator"] == r["Operator"]].iloc[0].to_dict(),
                                       **grp.to_dict()})
        radar_html += f"""<div class="radar-card">
          <h3>{r['Equipment']} <span style="font-weight:400;font-size:0.8rem;color:#6b6255">({r['Operator']})</span></h3>
          {radar_svg(labels, values)}
          <div>EPI = <b>{r['EPI']:.1f}</b> / 100</div>
          <div class="interp"><b>{pattern}</b><br>{meaning}</div>
        </div>"""

    # Group table
    grp_rows = ""
    for _, g in ogpi_df.iterrows():
        grp_rows += (f"<tr><td>{g['Operator']}</td><td>{g['OGPI']:.1f}</td>"
                     f"<td>{g['S_load']:.1f}</td><td>{g['S_interference']:.1f}</td><td>{g['S_balance']:.1f}</td></tr>")
    group_table_html = f"""<table><tr><th>Operator group</th><th>OGPI</th><th>S_load</th>
    <th>S_interference</th><th>S_balance</th></tr>{grp_rows}</table>"""

    # Reference pattern table
    pattern_rows = "".join(f"<tr><td>{p}</td><td>{m}</td></tr>" for p, m, _ in PATTERN_TABLE)
    pattern_html = f"<table><tr><th>Pattern</th><th>Main interpretation</th></tr>{pattern_rows}</table>"

    threshold_rows = "".join(
        f"<tr><td>{k}</td><td>{'>=' if op=='ge' else '<='} {g}</td><td>{a}</td><td>{'<' if op=='ge' else '>'} {a}</td></tr>"
        for k, (op, g, a) in THRESHOLDS.items()
    )
    threshold_html = f"""<table><tr><th>KPI</th><th>Green</th><th>Amber</th><th>Red</th></tr>{threshold_rows}</table>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Equipment Performance Assessment Report</title><style>{css}</style></head><body>
    <div class="wrap">
    <h1>Equipment Performance Assessment Report</h1>
    <p style="color:#6b6255">Mock shift, {SHIFT_START.strftime('%Y-%m-%d')} 06:00–14:00 — 2 operator groups, 5 machines.</p>

    <h2>Framework — step by step</h2>{steps_html}

    <h2>KPI map (equipment / operator / group)</h2>{kpi_map_html}

    <h2>Radar charts — performance signature per equipment</h2>
    <div class="radar-grid">{radar_html}</div>

    <h2>Operator Group Performance Index (OGPI)</h2>{group_table_html}

    <h2>Composite Supervised Equipment Performance Index (CSEPI)</h2>
    <div class="csepi">{csepi:.1f} / 100</div>

    <h2>Reference — interpretation pattern map (Step 6)</h2>{pattern_html}

    <h2>Reference — KPI thresholds (Appendix B)</h2>{threshold_html}
    </div></body></html>"""

    return html


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    events_df, production_df, operator_assignment_df, reference_df = build_step1_tables()

    equip_rows = [equipment_kpis(m, events_df, production_df, reference_df) for m in REFERENCE_TUPH]
    equip_kpi_df = pd.DataFrame(equip_rows)

    op_rows = [operator_kpis(op, operator_assignment_df) for op in OPERATOR_GROUPS]
    op_kpi_df = pd.DataFrame(op_rows)

    grp_rows = [group_kpis(op, machines, events_df, equip_kpi_df) for op, machines in OPERATOR_GROUPS.items()]
    grp_kpi_df = pd.DataFrame(grp_rows)

    epi_rows = []
    for _, r in equip_kpi_df.iterrows():
        epi_rows.append({"Equipment": r["Equipment"], "Operator": r["Operator"], **equipment_performance_index(r)})
    epi_df = pd.DataFrame(epi_rows)

    ogpi_rows = []
    for op in OPERATOR_GROUPS:
        op_row = op_kpi_df[op_kpi_df["Operator"] == op].iloc[0]
        grp_row = grp_kpi_df[grp_kpi_df["Operator"] == op].iloc[0]
        epi_values = epi_df[epi_df["Operator"] == op]["EPI"].values
        ogpi_rows.append({"Operator": op, **operator_group_performance_index(op_row, grp_row, epi_values)})
    ogpi_df = pd.DataFrame(ogpi_rows)

    csepi = composite_supervised_equipment_index(epi_df["EPI"].values, ogpi_df["OGPI"].values)

    # Save data outputs
    equip_kpi_df.to_csv(os.path.join(OUTPUT_DIR, "equipment_kpis.csv"), index=False)
    op_kpi_df.to_csv(os.path.join(OUTPUT_DIR, "operator_kpis.csv"), index=False)
    grp_kpi_df.to_csv(os.path.join(OUTPUT_DIR, "group_kpis.csv"), index=False)
    epi_df.to_csv(os.path.join(OUTPUT_DIR, "equipment_performance_index.csv"), index=False)
    ogpi_df.to_csv(os.path.join(OUTPUT_DIR, "operator_group_performance_index.csv"), index=False)

    html = build_html(equip_kpi_df, epi_df, op_kpi_df, grp_kpi_df, ogpi_df, csepi)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"CSEPI = {csepi:.1f}/100")
    print(equip_kpi_df[["Equipment", "Operator", "UPH Adherence", "AMTBI", "OWTR"]].to_string(index=False))
    print(f"\nReport written to: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
