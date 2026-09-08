#!/usr/bin/env python3
"""
Statistical CA / MCA downtime analyzer (uses the `prince` library, 0.14.0).

Input CSV columns: Machine ID, Duration (sec), ReasonCode, Lot ID,
Product Code, Machine Group, and optionally an Int flag column
(0/1 - 1 = counted as an interruption, 0 = not counted).

What it does (the full, one-script process)
--------------------------------------------
1. Loads the CSV and, if an Int/Interruption flag column is present,
   filters to only the rows where it equals 1 (an interruption actually
   occurred) - every table and plot below is built from that filtered
   data only.

2. Adds a derived categorical feature `DurationLevel` (Low/Medium/High/
   Critical) by cutting `Duration (sec)` into quantile bins (pandas.qcut,
   q=4 by default -> quartiles).

3. Treats these 6 columns as the analysis variables:
   Machine ID, Machine Group, ReasonCode, Product Code, Lot ID, DurationLevel

4. CA  (Correspondence Analysis)          -> for EVERY pair of variables
   (all C(6,2)=15 combinations): builds the contingency table, runs
   prince.CA, reports the chi-square test / Cramer's V, eigenvalues
   (explained inertia), row/column coordinates, and the top-N
   statistically most-associated cells (largest standardized residuals).
   A CA biplot (row + column coordinates) is saved as a PNG alongside it.

5. MCA (Multiple Correspondence Analysis) -> for every 3-way combination
   of variables (all C(6,3)=20), for the specific combination the user
   asked for (Machine Group + ReasonCode + DurationLevel), and for all 6
   variables at once: runs prince.MCA, reports eigenvalues, category
   coordinates/cos2, the top-N raw combinations by total downtime, a
   lift-analysis table for those same variables, and the top-N closest
   cross-variable category pairs in the MCA map. A category-map PNG is
   saved alongside each report.

6. Lift analysis -> for every pair (and for every MCA combination): lift
   = observed proportion / expected proportion under independence.
   Lift > 1 means the combination occurs more often than chance would
   predict (a classic market-basket-analysis "association rule" metric,
   complementary to the CA standardized residuals). Combinations with
   fewer than --min-count events are excluded to avoid noise from rare,
   one-off co-occurrences.

7. Plots (PNG, matplotlib) -> a Pareto chart (bars + cumulative-% line)
   per variable, a CA biplot per pair, and an MCA category map per
   multi-way combination - saved next to their corresponding text report.

8. Interpretation -> INTERPRETATION.md at the top of --outdir: a
   plain-language summary of the strongest single-cause contributors,
   the strongest pairwise associations (CA + lift), the multi-cause
   (MCA) highlights, and a few auto-generated recommendations, all
   derived from the tables above.

9. HTML dashboard -> report_data.json + report.html at the top of
   --outdir. report_data.json is a plain JSON file holding every table,
   stat and image path used by the dashboard - it is written to disk
   first, same as the .txt/.csv reports, and then read back to build
   report.html, so the dashboard is built strictly from files in this
   folder, not from anything kept only in memory. report.html is a
   single, self-contained (all images embedded as base64, no external
   network calls) modern HTML page with:
     - an Overview tab: KPIs, the top-5 pairwise associations across all
       15 pairs (by Cramer's V and by lift), the top-5 three-way findings
       across all 20 MCA combinations (by duration and by lift), and a
       Pareto thumbnail grid;
     - one tab per variable (a table + Pareto chart), with a "compare
       with" sub-tab per other variable (CA stats, CA biplot map
       highlighting the two variables' categories, lift table, top-N
       combination table);
     - a Multi-Cause tab (dropdown over the highlighted triple + all 20
       triples + all 6 variables) showing, for the selected combination,
       its category map/coordinates plus EVERY combination it found (not
       just the top N) in the downtime, lift, and closest-category-pair
       tables;
     - an Interpretation tab mirroring INTERPRETATION.md.
   Every table beyond a handful of rows is paginated client-side (10
   rows/page, Prev/Next controls) so large tables (all Lot ID values, all
   MCA combinations, ...) stay scannable instead of dumping hundreds of
   rows on the page at once.
   Open it directly in a browser - no server needed. Pass
   --rebuild-html-only to regenerate report.html purely by reading an
   existing report_data.json back from --outdir, with no CSV, no re-run
   of the analysis, and no network access.

10. Output layout (under --outdir, default "output"):
     output/
       00_Summary.txt
       INTERPRETATION.md                  auto-generated takeaways (step 8)
       report_data.json                   everything report.html is built from
       report.html                        interactive dashboard (step 9)
       <Variable>/                        one folder per variable (6)
         00_CA_univariate_<Variable>.txt    full Pareto detail
         00_CA_univariate_<Variable>_pareto.png
         pairwise/
           CA_<Variable>_x_<Other>.txt       statistical CA (prince)
           CA_<Variable>_x_<Other>_biplot.png
           Lift_<Variable>_x_<Other>.txt     lift/association-rule table
           Top<N>_<Variable>_x_<Other>_combo.txt   raw top-N combo table
           Top<N>_<Other>_per_<Variable>.txt       top-N Other per each Variable value
       MultiWayMCA/
         00_MCA_MachineGroup_ReasonCode_DurationLevel.txt   (the requested combo)
         00_MCA_MachineGroup_ReasonCode_DurationLevel_map.png
         MCA_triple_<A>_<B>_<C>.txt (+ _map.png)  all other 3-way combinations
         MCA_all_variables.txt (+ _map.png)       all 6 variables at once
       pivots/
         <A>_x_<B>_pivot.csv                duration-sum matrix, all 15 pairs

Requirements: pandas, numpy, scipy, matplotlib, scikit-learn==1.5.2,
prince==0.14.0 (see requirements.txt - prince 0.14.0 needs
scikit-learn<1.6, newer scikit-learn removed an internal API it relies on).

Usage:
    python3 analyze_ca_mca.py [INPUT.csv] [--outdir output] [--top 5]
                               [--quantiles 4] [--min-count 5]

    INPUT.csv defaults to "cleaned_file.csv" next to this script, so with a
    file placed there you can just run: python3 analyze_ca_mca.py

    python3 analyze_ca_mca.py --outdir output --rebuild-html-only
        Rebuild output/report.html from output/report_data.json alone -
        no CSV read, no analysis re-run.
"""

import argparse
import base64
import html
import json
import os
import shutil
from datetime import datetime, timezone
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import prince
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# Column resolution (tolerant of minor header naming differences)
# ---------------------------------------------------------------------------

COLUMN_ALIASES = {
    "Machine ID": ["machineid", "machine"],
    "Duration (sec)": ["durationsec", "duration", "durations", "durationseconds"],
    "ReasonCode": ["reasoncode", "reason"],
    "Lot ID": ["lotid", "lot"],
    "Product Code": ["productcode", "product"],
    "Machine Group": ["machinegroup", "group"],
}

# Optional binary flag column: 1 = counted as an interruption/downtime event,
# 0 = not counted. When present, only rows with value 1 feed every table.
INT_FLAG_ALIASES = ["int", "interruption", "isinterruption", "interruptionflag", "interrupt"]

VARIABLES = ["Machine ID", "Machine Group", "ReasonCode", "Product Code", "Lot ID", "DurationLevel"]
HIGHLIGHT_TRIPLE = ("Machine Group", "ReasonCode", "DurationLevel")

# Colorblind-safe categorical palette (Okabe & Ito, 2008), assigned in a
# fixed order - never re-cycled per plot, so a given position always maps
# to the same role (1st series, 2nd series, ...).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
MAX_PARETO_BARS = 15


def _normalize(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def resolve_columns(columns):
    normalized = {_normalize(c): c for c in columns}
    resolved, missing = {}, []
    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for cand in [canonical] + aliases:
            if _normalize(cand) in normalized:
                found = normalized[_normalize(cand)]
                break
        if found is None:
            missing.append(canonical)
        else:
            resolved[canonical] = found
    if missing:
        raise SystemExit(f"ERROR: missing required column(s): {missing}\nHeaders found: {list(columns)}")
    return resolved


def find_int_flag_column(columns):
    normalized = {_normalize(c): c for c in columns}
    for cand in INT_FLAG_ALIASES:
        if _normalize(cand) in normalized:
            return normalized[_normalize(cand)]
    return None


def load_data(csv_path):
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    colmap = resolve_columns(raw.columns)

    int_flag_col = find_int_flag_column(raw.columns)
    df = raw.rename(columns={v: k for k, v in colmap.items()})[list(COLUMN_ALIASES.keys())].copy()
    if int_flag_col is not None:
        df["Int"] = pd.to_numeric(raw[int_flag_col], errors="coerce")

    n_before = len(df)
    df["Duration (sec)"] = pd.to_numeric(
        df["Duration (sec)"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    df = df.dropna(subset=["Duration (sec)"]).copy()
    bad_rows = n_before - len(df)

    int_dropped = 0
    if int_flag_col is not None:
        n_before_flag = len(df)
        df = df[df["Int"] == 1].copy()
        int_dropped = n_before_flag - len(df)
        df = df.drop(columns=["Int"])

    for col in ["Machine ID", "ReasonCode", "Lot ID", "Product Code", "Machine Group"]:
        df[col] = df[col].fillna("(blank)").astype(str).str.strip().replace("", "(blank)")

    return df, bad_rows, int_flag_col, int_dropped


def add_duration_level(df, q=4):
    labels = ["Low", "Medium", "High", "Critical"] if q == 4 else [f"Level{i+1}" for i in range(q)]
    levels, bin_edges = pd.qcut(df["Duration (sec)"], q=q, labels=labels, duplicates="drop", retbins=True)
    df = df.copy()
    df["DurationLevel"] = levels.astype(str)
    used_labels = list(pd.Categorical(levels).categories.astype(str))
    return df, used_labels, bin_edges


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_hms(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_table(headers, rows, aligns=None):
    n = len(headers)
    aligns = aligns or ["<"] * n
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells):
        return "  ".join(f"{str(c):{aligns[i]}{widths[i]}}" for i, c in enumerate(cells))

    lines = [fmt_row(headers), "  ".join("-" * w for w in widths)]
    for r in rows:
        lines.append(fmt_row(r))
    return lines


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class NpEncoder(json.JSONEncoder):
    """Lets json.dump handle numpy scalar types that leak in from pandas records."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_report_data(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NpEncoder, indent=2)


def load_report_data(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe(name):
    return name.replace(" ", "").replace("(", "").replace(")", "")


def cramers_v(chi2, n, r, c):
    k = min(r, c)
    if n <= 0 or k <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))))


def association_strength(v):
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "weak"
    if v < 0.5:
        return "moderate"
    return "strong"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot_pareto(df, var, total_duration, path, top_n_bars=MAX_PARETO_BARS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g = df.groupby(var)["Duration (sec)"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    if len(g) > top_n_bars:
        head = g.iloc[:top_n_bars]
        other = pd.DataFrame({"count": [g.iloc[top_n_bars:]["count"].sum()],
                               "sum": [g.iloc[top_n_bars:]["sum"].sum()]}, index=["(Other)"])
        g = pd.concat([head, other])

    cum_pct = g["sum"].cumsum() / total_duration * 100 if total_duration else g["sum"].cumsum() * 0

    fig, ax1 = plt.subplots(figsize=(max(6.0, len(g) * 0.55), 5.0))
    ax1.bar(range(len(g)), g["sum"], color=PALETTE[0], width=0.65)
    ax1.set_xticks(range(len(g)))
    ax1.set_xticklabels(g.index.astype(str), rotation=45, ha="right")
    ax1.set_ylabel("Total downtime (sec)")
    _style_axes(ax1)

    ax2 = ax1.twinx()
    ax2.plot(range(len(g)), cum_pct, color=PALETTE[1], marker="o", markersize=5, linewidth=2)
    ax2.set_ylabel("Cumulative % of total downtime")
    ax2.set_ylim(0, 105)
    ax2.axhline(80, color="#888888", linestyle="--", linewidth=1)
    ax2.spines["top"].set_visible(False)

    ax1.set_title(f"Pareto - {var}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pad_second_dim(coords):
    if coords.shape[1] < 2:
        coords = coords.copy()
        coords[coords.shape[1]] = 0.0
    return coords


# Cycle of label offsets (points) used to spread out annotations that would
# otherwise stack on top of each other when categories cluster near the origin.
LABEL_OFFSETS = [(5, 5), (5, -11), (-38, 5), (-38, -11), (5, 14), (-38, 14)]


def _offset_for(i):
    return LABEL_OFFSETS[i % len(LABEL_OFFSETS)]


def plot_ca_biplot(row_coords, col_coords, var_a, var_b, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row_coords = _pad_second_dim(row_coords)
    col_coords = _pad_second_dim(col_coords)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.axhline(0, color="#aaaaaa", linewidth=0.8)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)

    label_i = 0
    ax.scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1], color=PALETTE[0], s=55, label=var_a, zorder=3)
    for idx, r in row_coords.iterrows():
        ax.annotate(str(idx), (r.iloc[0], r.iloc[1]), fontsize=8, color=PALETTE[0],
                    xytext=_offset_for(label_i), textcoords="offset points")
        label_i += 1

    ax.scatter(col_coords.iloc[:, 0], col_coords.iloc[:, 1], color=PALETTE[1], s=55, marker="^", label=var_b, zorder=3)
    for idx, r in col_coords.iterrows():
        ax.annotate(str(idx), (r.iloc[0], r.iloc[1]), fontsize=8, color=PALETTE[1],
                    xytext=_offset_for(label_i), textcoords="offset points")
        label_i += 1

    ax.set_xlabel("Dimension 0")
    ax.set_ylabel("Dimension 1")
    ax.set_title(f"CA map - {var_a} x {var_b}")
    ax.legend(loc="best", frameon=False)
    _style_axes(ax)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mca_map(coords, vars_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.axhline(0, color="#aaaaaa", linewidth=0.8)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)

    label_i = 0
    for i, v in enumerate(vars_list):
        color = PALETTE[i % len(PALETTE)]
        sub = coords[coords.index.str.startswith(f"{v}__")]
        ax.scatter(sub.iloc[:, 0], sub.iloc[:, 1], color=color, s=55, label=v, zorder=3)
        for idx, r in sub.iterrows():
            label = idx.split("__", 1)[1]
            ax.annotate(label, (r.iloc[0], r.iloc[1]), fontsize=7, color=color,
                        xytext=_offset_for(label_i), textcoords="offset points")
            label_i += 1

    ax.set_xlabel("Dimension 0")
    ax.set_ylabel("Dimension 1")
    ax.set_title("MCA map - " + " x ".join(vars_list))
    ax.legend(loc="best", frameon=False, fontsize=8)
    _style_axes(ax)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Descriptive (frequency/duration) reports - single dimension & combos
# ---------------------------------------------------------------------------

def univariate_report(df, var, total_duration, total_events):
    g = df.groupby(var)["Duration (sec)"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    lines = [f"CAUSE ANALYSIS (CA) - descriptive Pareto by {var}", "=" * 70,
             f"Total events   : {total_events}",
             f"Total downtime : {total_duration:,.0f} sec ({fmt_hms(total_duration)})",
             f"Unique {var} values: {len(g)}", ""]

    rows, table_records, cum = [], [], 0.0
    for rank, (value, r) in enumerate(g.iterrows(), start=1):
        pct = (r["sum"] / total_duration * 100) if total_duration else 0.0
        cum += pct
        rows.append([rank, value, int(r["count"]), f"{r['sum']:,.0f}", fmt_hms(r["sum"]), f"{pct:.1f}%", f"{cum:.1f}%"])
        table_records.append({"rank": rank, "value": str(value), "count": int(r["count"]),
                               "duration_sec": float(r["sum"]), "duration_hms": fmt_hms(r["sum"]),
                               "pct": round(pct, 1), "cum_pct": round(cum, 1)})
    headers = ["Rank", var, "Count", "Duration(sec)", "Duration(hh:mm:ss)", "% Total", "Cum %"]
    aligns = ["<", "<", ">", ">", ">", ">", ">"]
    lines.extend(fmt_table(headers, rows, aligns))
    lines.append("")

    stats = {"var": var, "n_unique": len(g), "table_records": table_records}
    if len(g):
        top_value, top_row = g.index[0], g.iloc[0]
        top_pct = top_row["sum"] / total_duration * 100 if total_duration else 0.0
        lines.append(f"Top contributor: '{top_value}' -> {top_row['sum']:,.0f} sec "
                      f"({top_pct:.1f}% of total downtime, {int(top_row['count'])} events)")
        cum_running, n80 = 0.0, 0
        for _, r in g.iterrows():
            n80 += 1
            cum_running += r["sum"] / total_duration * 100 if total_duration else 0
            if cum_running >= 80.0:
                break
        lines.append(f"Pareto (80/20): top {n80} of {len(g)} {var} value(s) account for ~80% of total downtime.")
        stats.update({"top_value": top_value, "top_pct": top_pct, "top_count": int(top_row["count"]), "n80": n80})
    lines.append("")
    return "\n".join(lines), stats


def combo_top_report(df, cols, total_duration, top_n):
    g = df.groupby(cols)["Duration (sec)"].agg(["count", "sum"]).sort_values("sum", ascending=False).head(top_n)
    label = " + ".join(cols)
    lines = [f"TOP {top_n} {label} COMBINATIONS (by total downtime)", "=" * 70, ""]
    rows, records = [], []
    for rank, (key, r) in enumerate(g.iterrows(), start=1):
        key = key if isinstance(key, tuple) else (key,)
        pct = (r["sum"] / total_duration * 100) if total_duration else 0.0
        rows.append([rank, *key, int(r["count"]), f"{r['sum']:,.0f}", fmt_hms(r["sum"]), f"{pct:.1f}%"])
        rec = {"rank": rank, "count": int(r["count"]), "duration_sec": float(r["sum"]),
               "duration_hms": fmt_hms(r["sum"]), "pct": round(pct, 1)}
        rec["values"] = {c: str(v) for c, v in zip(cols, key)}
        records.append(rec)
    headers = ["Rank", *cols, "Count", "Duration(sec)", "Duration(hh:mm:ss)", "% of Total"]
    aligns = ["<"] + ["<"] * len(cols) + [">", ">", ">", ">"]
    lines.extend(fmt_table(headers, rows, aligns))
    lines.append("")
    if len(g):
        top_key = g.index[0]
        top_key = top_key if isinstance(top_key, tuple) else (top_key,)
        top_row = g.iloc[0]
        pct = top_row["sum"] / total_duration * 100 if total_duration else 0.0
        combo_str = ", ".join(f"{c}='{v}'" for c, v in zip(cols, top_key))
        lines.append(f"Most associated combination: {combo_str} -> {top_row['sum']:,.0f} sec "
                      f"({pct:.1f}% of total downtime, {int(top_row['count'])} events)")
    lines.append("")
    return "\n".join(lines), records


def top_n_per_group_report(df, group_var, rank_var, top_n):
    lines = [f"TOP {top_n} {rank_var} PER {group_var}", "=" * 70, ""]
    groups = []
    totals = df.groupby(group_var)["Duration (sec)"].sum().sort_values(ascending=False)
    for g_value, g_total in totals.items():
        sub = df[df[group_var] == g_value].groupby(rank_var)["Duration (sec)"].agg(["count", "sum"])
        sub = sub.sort_values("sum", ascending=False).head(top_n)
        lines.append(f"{group_var}: {g_value}  (total downtime {g_total:,.0f} sec / {fmt_hms(g_total)})")
        rows, sub_records = [], []
        for value, r in sub.iterrows():
            pct = (r["sum"] / g_total * 100) if g_total else 0.0
            rows.append([value, int(r["count"]), f"{r['sum']:,.0f}", fmt_hms(r["sum"]), f"{pct:.1f}%"])
            sub_records.append({"value": str(value), "count": int(r["count"]), "duration_sec": float(r["sum"]),
                                 "duration_hms": fmt_hms(r["sum"]), "pct": round(pct, 1)})
        headers = [rank_var, "Count", "Duration(sec)", "Duration(hh:mm:ss)", "% of group total"]
        aligns = ["<", ">", ">", ">", ">"]
        for line in fmt_table(headers, rows, aligns):
            lines.append("    " + line)
        lines.append("")
        groups.append({"group_value": str(g_value), "total_sec": float(g_total),
                        "total_hms": fmt_hms(g_total), "rows": sub_records})
    return "\n".join(lines), groups


def write_pivot_csv(path, df, var_a, var_b):
    pt = pd.pivot_table(df, index=var_a, columns=var_b, values="Duration (sec)", aggfunc="sum", fill_value=0)
    pt["Row Total"] = pt.sum(axis=1)
    pt.loc["Column Total"] = pt.sum(axis=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pt.to_csv(path)


# ---------------------------------------------------------------------------
# Lift analysis (market-basket-style association rules)
# ---------------------------------------------------------------------------

def lift_table(df, cols, min_count):
    n = len(df)
    marginal = {c: df[c].value_counts(normalize=True) for c in cols}
    g = df.groupby(list(cols)).size().reset_index(name="count")
    g = g[g["count"] >= min_count].copy()
    if g.empty:
        return g.assign(support=[], expected_support=[], expected_count=[], lift=[])
    expected_support = np.ones(len(g))
    for c in cols:
        expected_support *= g[c].map(marginal[c]).values
    g["support"] = g["count"] / n
    g["expected_support"] = expected_support
    g["expected_count"] = expected_support * n
    g["lift"] = g["support"] / g["expected_support"]
    return g.sort_values("lift", ascending=False)


def lift_table_records(table, cols, top_n=None):
    subset = table if top_n is None else table.head(top_n)
    records = []
    for rank, (_, r) in enumerate(subset.iterrows(), start=1):
        records.append({
            "rank": rank, "values": {c: str(r[c]) for c in cols}, "count": int(r["count"]),
            "support_pct": round(float(r["support"]) * 100, 2),
            "expected_count": round(float(r["expected_count"]), 1),
            "lift": round(float(r["lift"]), 2),
        })
    return records


def combo_all_records(df, cols, total_duration, min_count=1):
    """Like combo_top_report's records, but every combination meeting
    min_count (not just the top N) - used for the HTML dashboard's "all
    outputs" tables."""
    g = df.groupby(cols)["Duration (sec)"].agg(["count", "sum"])
    g = g[g["count"] >= min_count].sort_values("sum", ascending=False)
    records = []
    for rank, (key, r) in enumerate(g.iterrows(), start=1):
        key = key if isinstance(key, tuple) else (key,)
        pct = (r["sum"] / total_duration * 100) if total_duration else 0.0
        records.append({"rank": rank, "count": int(r["count"]), "duration_sec": float(r["sum"]),
                         "duration_hms": fmt_hms(r["sum"]), "pct": round(pct, 1),
                         "values": {c: str(v) for c, v in zip(cols, key)}})
    return records


def render_lift_report(cols, table, top_n, min_count):
    label = " + ".join(cols)
    lines = [f"LIFT ANALYSIS - {label} (top {top_n} by lift, min count = {min_count})", "=" * 70,
             "Lift = observed proportion of the combination / expected proportion under independence.",
             "  Lift > 1x -> occurs MORE often than random chance predicts (positive association).",
             "  Lift < 1x -> occurs LESS often than random chance predicts (negative association).",
             "  Lift = 1x -> no association (independent).",
             f"Combinations with fewer than {min_count} events are excluded (too rare to trust the ratio).", ""]

    top = table.head(top_n)
    rows = []
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        combo_vals = [r[c] for c in cols]
        rows.append([rank, *combo_vals, int(r["count"]), f"{r['support']*100:.2f}%",
                     f"{r['expected_count']:.1f}", f"{r['lift']:.2f}x"])
    headers = ["Rank", *cols, "Count", "Support %", "Expected Count", "Lift"]
    aligns = ["<"] + ["<"] * len(cols) + [">", ">", ">", ">"]
    lines.extend(fmt_table(headers, rows, aligns))
    lines.append("")

    if len(top):
        r = top.iloc[0]
        combo_str = ", ".join(f"{c}='{r[c]}'" for c in cols)
        lines.append(f"Strongest lift: {combo_str} -> occurs {r['lift']:.2f}x more often than chance would "
                      f"predict ({int(r['count'])} events observed vs ~{r['expected_count']:.1f} expected).")
    else:
        lines.append(f"No combination reached the minimum count of {min_count}.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical CA (pairwise, prince) - the core "Correspondence Analysis"
# ---------------------------------------------------------------------------

def ca_report(df, var_a, var_b, top_n=5, plot_path=None):
    ct = pd.crosstab(df[var_a], df[var_b])
    n = ct.values.sum()
    chi2, p, dof, expected = chi2_contingency(ct)
    v = cramers_v(chi2, n, ct.shape[0], ct.shape[1])

    n_comp = max(1, min(2, ct.shape[0] - 1, ct.shape[1] - 1))
    ca = prince.CA(n_components=n_comp, random_state=42).fit(ct)

    lines = [f"CORRESPONDENCE ANALYSIS (CA) - {var_a}  x  {var_b}", "=" * 70,
             f"Contingency table: {ct.shape[0]} x {ct.shape[1]} categories, N = {int(n)} events", ""]

    lines.append("Chi-square test of association")
    lines.append("-" * 40)
    lines.append(f"  chi2 = {chi2:.2f}, dof = {dof}, p-value = {p:.4g}")
    lines.append(f"  Cramer's V = {v:.3f}  -> {association_strength(v)} association")
    lines.append(f"  {'Statistically significant (p<0.05)' if p < 0.05 else 'Not statistically significant (p>=0.05)'} "
                 f"association between {var_a} and {var_b}.")
    lines.append("")

    eigen_df = ca.eigenvalues_summary.reset_index()
    lines.append("Eigenvalues / explained inertia")
    lines.append("-" * 40)
    lines.append(ca.eigenvalues_summary.to_string())
    lines.append("")

    # Standardized residuals -> the statistically most-associated cells
    resid = (ct - expected) / np.sqrt(expected)
    flat = resid.stack()
    flat = flat.reindex(flat.abs().sort_values(ascending=False).index).head(top_n)
    lines.append(f"Top {top_n} most statistically associated {var_a} x {var_b} cells (largest standardized residuals)")
    lines.append("-" * 40)
    rows, residual_records = [], []
    top_residual = None
    for rank, ((a_val, b_val), resid_val) in enumerate(flat.items(), start=1):
        obs = ct.loc[a_val, b_val]
        exp = expected[ct.index.get_loc(a_val), ct.columns.get_loc(b_val)]
        direction = "over-represented" if resid_val > 0 else "under-represented"
        rows.append([rank, a_val, b_val, int(obs), f"{exp:.1f}", f"{resid_val:+.2f}", direction])
        residual_records.append({"rank": rank, "a": str(a_val), "b": str(b_val), "observed": int(obs),
                                  "expected": round(float(exp), 1), "resid": round(float(resid_val), 2),
                                  "direction": direction})
        if rank == 1:
            top_residual = {"a": a_val, "b": b_val, "resid": float(resid_val), "direction": direction}
    headers = ["Rank", var_a, var_b, "Observed", "Expected", "Std.Resid", "Direction"]
    aligns = ["<", "<", "<", ">", ">", ">", "<"]
    lines.extend(fmt_table(headers, rows, aligns))
    lines.append("")
    lines.append("(A large positive standardized residual means that combination occurs far more often")
    lines.append(" than chance would predict -> the two categories are strongly associated.)")
    lines.append("")

    row_coords = ca.row_coordinates(ct)
    col_coords = ca.column_coordinates(ct)
    lines.append(f"{var_a} coordinates (CA map)")
    lines.append("-" * 40)
    lines.append(row_coords.round(3).to_string())
    lines.append("")
    lines.append(f"{var_b} coordinates (CA map)")
    lines.append("-" * 40)
    lines.append(col_coords.round(3).to_string())
    lines.append("")

    if plot_path:
        plot_ca_biplot(row_coords, col_coords, var_a, var_b, plot_path)

    stats = {
        "var_a": var_a, "var_b": var_b, "n": int(n), "chi2": float(chi2), "p": float(p), "dof": int(dof),
        "cramers_v": v, "strength": association_strength(v), "top_residual": top_residual,
        "eigen_records": eigen_df.to_dict("records"), "residual_records": residual_records,
    }
    return "\n".join(lines), stats


# ---------------------------------------------------------------------------
# Statistical MCA (3+ variables, prince)
# ---------------------------------------------------------------------------

def mca_report(df, vars_list, total_duration, top_n=5, min_count=5, plot_path=None):
    X = df[list(vars_list)].astype(str)
    mca = prince.MCA(n_components=2, random_state=42).fit(X)

    label = " x ".join(vars_list)
    lines = [f"MULTIPLE CORRESPONDENCE ANALYSIS (MCA) - {label}", "=" * 70,
             f"Variables: {', '.join(vars_list)}   N = {len(X)} events", ""]

    eigen_df = mca.eigenvalues_summary.reset_index()
    lines.append("Eigenvalues / explained inertia")
    lines.append("-" * 40)
    lines.append(mca.eigenvalues_summary.to_string())
    lines.append("")
    cum_pct = float(mca.eigenvalues_summary["% of variance (cumulative)"].iloc[-1].strip("%"))

    coords = mca.column_coordinates(X)
    cos2 = mca.column_cosine_similarities(X)
    coords.columns = [f"dim{c}" for c in coords.columns]
    cos2.columns = [f"cos2_dim{c}" for c in cos2.columns]
    combined = pd.concat([coords, cos2], axis=1)
    combined["quality"] = cos2.sum(axis=1)
    combined = combined.sort_values("quality", ascending=False)
    lines.append("Category coordinates & quality of representation (cos2), best-represented first")
    lines.append("-" * 40)
    lines.append(combined.round(3).to_string())
    lines.append("")
    category_df = combined.round(3).reset_index()
    category_df.columns = ["category"] + list(category_df.columns[1:])

    # Raw top-N combination by total downtime (practical / easy to read), plus
    # every combination for the HTML dashboard's "all outputs" table.
    combo_text, combo_records = combo_top_report(df, list(vars_list), total_duration, top_n)
    combo_records_all = combo_all_records(df, list(vars_list), total_duration, min_count)
    lines.append(combo_text)
    top_combo = None
    if combo_records:
        top_combo = {"values": combo_records[0]["values"], "count": combo_records[0]["count"],
                     "duration": combo_records[0]["duration_sec"], "pct": combo_records[0]["pct"]}

    # Lift analysis for the same set of variables (top-N for the text report,
    # every combination meeting --min-count for the HTML dashboard).
    lift = lift_table(df, list(vars_list), min_count)
    lift_records = lift_table_records(lift, list(vars_list), top_n)
    lift_records_all = lift_table_records(lift, list(vars_list))
    lines.append(render_lift_report(list(vars_list), lift, top_n, min_count))
    top_lift = None
    if lift_records:
        top_lift = {"values": lift_records[0]["values"], "lift": lift_records[0]["lift"],
                    "count": lift_records[0]["count"]}

    # Closest cross-variable category pairs in the MCA map = most "associated"
    var_of = {}
    for v in vars_list:
        for cat in X[v].unique():
            var_of[f"{v}__{cat}"] = v
    idx = list(coords.index)
    dist_rows = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            ci, cj = idx[i], idx[j]
            if var_of[ci] == var_of[cj]:
                continue  # skip same-variable categories (mutually exclusive, not a real "association")
            d = np.sqrt(((coords.loc[ci] - coords.loc[cj]) ** 2).sum())
            dist_rows.append((ci, cj, d))
    dist_rows.sort(key=lambda t: t[2])
    lines.append(f"Top {top_n} closest cross-variable category pairs in the MCA map (most associated)")
    lines.append("-" * 40)
    rows, closest_records, closest_records_all = [], [], []
    for rank, (a, b, d) in enumerate(dist_rows, start=1):
        rec = {"rank": rank, "a": a.replace("__", ": "), "b": b.replace("__", ": "), "distance": round(float(d), 3)}
        closest_records_all.append(rec)
        if rank <= top_n:
            rows.append([rank, rec["a"], rec["b"], f"{d:.3f}"])
            closest_records.append(rec)
    headers = ["Rank", "Category A", "Category B", "Distance (smaller = more associated)"]
    aligns = ["<", "<", "<", ">"]
    lines.extend(fmt_table(headers, rows, aligns))
    lines.append("")

    if plot_path:
        plot_mca_map(coords, list(vars_list), plot_path)

    stats = {
        "vars": list(vars_list), "cum_pct": cum_pct, "top_combo": top_combo, "top_lift": top_lift,
        "top_closest_pair": {"a": dist_rows[0][0], "b": dist_rows[0][1], "dist": float(dist_rows[0][2])} if dist_rows else None,
        "eigen_records": eigen_df.to_dict("records"), "category_records": category_df.to_dict("records"),
        "combo_records": combo_records, "lift_records": lift_records, "closest_records": closest_records,
        "combo_records_all": combo_records_all, "lift_records_all": lift_records_all,
        "closest_records_all": closest_records_all,
    }
    return "\n".join(lines), stats


# ---------------------------------------------------------------------------
# Interpretation - shared aggregation, then Markdown and HTML renderers
# ---------------------------------------------------------------------------

def compute_interpretation_summary(univariate_stats, ca_stats_all, lift_stats_all):
    biggest = max((s for s in univariate_stats if "top_pct" in s), key=lambda s: s["top_pct"], default=None)
    sig = [s for s in ca_stats_all if s["p"] < 0.05]
    sig_sorted = sorted(sig, key=lambda s: s["cramers_v"], reverse=True)
    lift_sorted = sorted(lift_stats_all, key=lambda s: s["lift"], reverse=True)
    return {"biggest": biggest, "sig_sorted": sig_sorted, "lift_sorted": lift_sorted}


def build_interpretation_md(csv_path, int_flag_col, int_dropped, total_duration, total_events,
                             level_labels, univariate_stats, summary, mca_highlight_stats, top_n, min_count):
    biggest, sig_sorted, lift_sorted = summary["biggest"], summary["sig_sorted"], summary["lift_sorted"]
    lines = ["# Downtime CA / MCA - Interpretation", "",
             f"*Auto-generated from `{os.path.basename(csv_path)}` - {total_events} events, "
             f"{total_duration:,.0f} sec ({total_duration/3600:,.1f} hr) of total downtime.*", ""]

    lines.append("## Data scope")
    if int_flag_col is not None:
        lines.append(f"- Filtered to rows where **{int_flag_col} = 1** (counted interruptions); "
                      f"**{int_dropped}** row(s) with {int_flag_col}=0 were excluded.")
    else:
        lines.append("- No Int/interruption flag column was found - all rows were used.")
    lines.append(f"- `DurationLevel` categories (quantile bins): {', '.join(level_labels)}.")
    lines.append(f"- Associations below use a minimum count of **{min_count}** events and show the top **{top_n}**.")
    lines.append("")

    lines.append("## Top single-cause contributors (CA)")
    lines.append("")
    lines.append("| Variable | Top value | % of total downtime | # values for 80% of downtime |")
    lines.append("|---|---|---|---|")
    for s in univariate_stats:
        if "top_value" in s:
            lines.append(f"| {s['var']} | {s['top_value']} | {s['top_pct']:.1f}% | {s['n80']} of {s['n_unique']} |")
    lines.append("")
    if biggest:
        lines.append(f"**Single biggest lever:** `{biggest['var']}` = '{biggest['top_value']}' alone accounts for "
                      f"**{biggest['top_pct']:.1f}%** of all downtime - the highest-leverage single-variable fix "
                      f"available in this dataset.")
    lines.append("")

    lines.append("## Strongest pairwise associations (Correspondence Analysis)")
    lines.append("")
    if sig_sorted:
        lines.append("| Pair | Cramer's V | Strength | p-value | Most surprising cell |")
        lines.append("|---|---|---|---|---|")
        for s in sig_sorted[:5]:
            tr = s["top_residual"]
            cell = f"{s['var_a']}='{tr['a']}' & {s['var_b']}='{tr['b']}' ({tr['direction']})" if tr else "n/a"
            lines.append(f"| {s['var_a']} x {s['var_b']} | {s['cramers_v']:.3f} | {s['strength']} | "
                          f"{s['p']:.3g} | {cell} |")
    else:
        lines.append("No pair reached statistical significance (p<0.05) - the variables in this dataset "
                      "appear largely independent of each other at the pairwise level.")
    lines.append("")

    lines.append("## Strongest associations by lift")
    lines.append("")
    if lift_sorted:
        lines.append("| Combination | Lift | Count |")
        lines.append("|---|---|---|")
        for s in lift_sorted[:5]:
            combo_str = ", ".join(f"{c}='{v}'" for c, v in s["values"].items())
            lines.append(f"| {combo_str} | {s['lift']:.2f}x | {s['count']} |")
        top = lift_sorted[0]
        combo_str = ", ".join(f"{c}='{v}'" for c, v in top["values"].items())
        lines.append("")
        lines.append(f"**Strongest co-occurrence:** {combo_str} happens **{top['lift']:.2f}x** more often than "
                      f"random chance would predict - worth investigating as a potential shared root cause.")
    else:
        lines.append(f"No pair had a combination reaching the minimum count of {min_count}.")
    lines.append("")

    lines.append(f"## Multi-cause highlight: {' + '.join(HIGHLIGHT_TRIPLE)}")
    lines.append("")
    tc = mca_highlight_stats.get("top_combo")
    tl = mca_highlight_stats.get("top_lift")
    tp = mca_highlight_stats.get("top_closest_pair")
    if tc:
        combo_str = ", ".join(f"{c}='{v}'" for c, v in tc["values"].items())
        lines.append(f"- Largest raw combination: {combo_str} -> {tc['duration']:,.0f} sec "
                      f"({tc['pct']:.1f}% of total downtime, {tc['count']} events).")
    if tl:
        combo_str = ", ".join(f"{c}='{v}'" for c, v in tl["values"].items())
        lines.append(f"- Strongest lift combination: {combo_str} -> {tl['lift']:.2f}x expected rate "
                      f"({tl['count']} events).")
    if tp:
        lines.append(f"- Closest categories in the MCA map: {tp['a'].replace('__', ': ')} and "
                      f"{tp['b'].replace('__', ': ')} (distance {tp['dist']:.3f}) - these travel together most.")
    lines.append(f"- These 3 variables together explain **{mca_highlight_stats['cum_pct']:.1f}%** of the "
                 f"total inertia (variance) on the first 2 MCA dimensions.")
    lines.append("")

    lines.append("## Notes & recommendations")
    lines.append("")
    lines.append("- Prioritize fixes in this order: the single biggest contributor above, then the strongest "
                  "significant pairwise association, then the strongest lift combination - each is progressively "
                  "more specific (and rarer) but often points to a more precise, fixable root cause.")
    lines.append(f"- Associations built on fewer than {min_count} events were dropped from the lift tables to "
                  "avoid over-reacting to one-off coincidences; lower `--min-count` to see rarer combinations "
                  "(with less statistical confidence).")
    lines.append("- `DurationLevel` thresholds are quantile-based (relative to this dataset) - they will shift "
                  "if the dataset changes substantially, so re-run this script rather than hand-copying the bin edges.")
    lines.append("- See the `<Variable>/` folders (or report.html) for full detail and plots behind every number "
                  "in this summary.")
    lines.append("- A pair with Cramer's V near 1.0 (e.g. Machine ID x Machine Group) can simply reflect a "
                  "structural/hierarchical relationship already known by construction (each machine belongs to "
                  "exactly one group) rather than a new finding - treat those as expected, not actionable.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

STRENGTH_COLORS = {
    "negligible": ("#dbeafe", "#1e3a8a"),
    "weak": ("#93c5fd", "#1e3a8a"),
    "moderate": ("#3b82f6", "#ffffff"),
    "strong": ("#1d4ed8", "#ffffff"),
}


class Raw(str):
    """Marker for a cell that is already-safe HTML (e.g. a badge span) and
    must NOT be escaped again by html_table."""


def esc(s):
    return html.escape(str(s))


# Registry of tables that need client-side pagination, filled in by every
# html_table() call and drained once at the end of build_html_report() into
# a block of pgInit(...) calls - see PAGE_JS.
_TABLE_SEQ = [0]
_PAGINATION_INITS = []
TABLE_PAGE_SIZE = 10


def html_table(headers, rows, page_size=TABLE_PAGE_SIZE):
    _TABLE_SEQ[0] += 1
    table_id = f"tbl-{_TABLE_SEQ[0]}"
    paginate = len(rows) > page_size

    out = [f'<div id="{table_id}" class="table-wrap"><table><thead><tr>']
    for h in headers:
        out.append(f"<th>{esc(h)}</th>")
    out.append("</tr></thead><tbody>")
    if not rows:
        out.append(f'<tr><td colspan="{len(headers)}" class="muted">No data.</td></tr>')
    for r in rows:
        cells = "".join(f"<td>{c if isinstance(c, Raw) else esc(c)}</td>" for c in r)
        out.append(f"<tr>{cells}</tr>")
    out.append("</tbody></table>")
    if paginate:
        out.append('<div class="pg-controls">'
                    f'<button class="pg-prev" onclick="pgNav(\'{table_id}\',-1)">&larr; Prev</button>'
                    '<span class="pg-label"></span>'
                    f'<button class="pg-next" onclick="pgNav(\'{table_id}\',1)">Next &rarr;</button>'
                    "</div>")
        _PAGINATION_INITS.append((table_id, page_size))
    out.append("</div>")
    return "".join(out)


def img_data_uri(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def img_html(path, alt):
    uri = img_data_uri(path)
    if not uri:
        return '<p class="muted">(chart unavailable)</p>'
    return f'<img class="chart" src="{uri}" alt="{esc(alt)}" loading="lazy">'


def strength_badge(strength):
    bg, fg = STRENGTH_COLORS.get(strength, ("#e5e7eb", "#111827"))
    return f'<span class="badge" style="background:{bg};color:{fg}">{esc(strength)}</span>'


def sig_badge(p):
    if p < 0.05:
        return '<span class="badge" style="background:#059669;color:#fff">significant (p&lt;0.05)</span>'
    return '<span class="badge" style="background:#9ca3af;color:#111827">not significant</span>'


def lift_badge(value):
    if value >= 1.2:
        bg, fg, arrow = "#059669", "#fff", "▲"
    elif value <= 0.8:
        bg, fg, arrow = "#d97706", "#fff", "▼"
    else:
        bg, fg, arrow = "#9ca3af", "#111827", "▬"
    return f'<span class="badge" style="background:{bg};color:{fg}">{arrow} {value:.2f}x</span>'


def kpi(label, value):
    return f'<div class="kpi"><div class="value">{esc(value)}</div><div class="label">{esc(label)}</div></div>'


PAGE_CSS = """
:root{--bg:#f5f6f8;--surface:#ffffff;--border:#e3e6eb;--text:#1a1d23;--text-muted:#666e7c;
--accent:#0072B2;--radius:12px;--shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04);
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--surface:#191c22;--border:#2b2f38;--text:#e7e9ee;--text-muted:#98a0ad;}}
*{box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);margin:0;}
.navbar{position:sticky;top:0;background:var(--surface);border-bottom:1px solid var(--border);
display:flex;gap:4px;padding:10px 18px;overflow-x:auto;z-index:20;align-items:center;}
.brand{font-weight:800;margin-right:14px;white-space:nowrap;font-size:1.05rem;}
.navbtn{padding:8px 14px;border-radius:8px;border:none;background:transparent;color:var(--text-muted);
font-weight:600;cursor:pointer;white-space:nowrap;font-size:.88rem;}
.navbtn:hover{background:var(--border);}
.navbtn.active{background:var(--accent);color:#fff;}
.panel{display:none;padding:22px 22px 60px;max-width:1180px;margin:0 auto;}
.panel.active{display:block;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
box-shadow:var(--shadow);padding:18px;margin-bottom:18px;}
.card h2{margin-top:0;font-size:1.15rem;}
.card h3{font-size:.98rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.03em;margin-bottom:10px;}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;}
.kpi{padding:14px 16px;border-radius:var(--radius);background:var(--surface);border:1px solid var(--border);}
.kpi .value{font-size:1.5rem;font-weight:800;}
.kpi .label{color:var(--text-muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;margin-top:2px;}
table{border-collapse:collapse;width:100%;font-size:.83rem;}
th,td{padding:7px 10px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap;}
td{font-family:var(--mono);}
th{color:var(--text-muted);font-weight:700;text-transform:uppercase;font-size:.68rem;letter-spacing:.03em;
font-family:var(--sans);}
.table-wrap{overflow-x:auto;margin-bottom:6px;}
.badge{display:inline-block;padding:3px 9px;border-radius:999px;font-size:.72rem;font-weight:700;white-space:nowrap;}
.subtabs{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 14px;}
.subtabbtn{padding:6px 13px;border-radius:999px;border:1px solid var(--border);background:var(--surface);
color:var(--text);cursor:pointer;font-size:.8rem;font-weight:600;}
.subtabbtn:hover{border-color:var(--accent);}
.subtabbtn.active{background:var(--accent);color:#fff;border-color:var(--accent);}
.subpanel{display:none;}
.subpanel.active{display:block;}
img.chart{max-width:100%;border-radius:8px;border:1px solid var(--border);background:#fff;display:block;margin:0 auto;}
select.mcaselect{padding:9px 12px;border-radius:8px;border:1px solid var(--border);background:var(--surface);
color:var(--text);font-size:.9rem;margin-bottom:16px;min-width:320px;}
h1{margin:.2rem 0 .1rem;font-size:1.5rem;}
.muted{color:var(--text-muted);}
.grid-2{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;align-items:start;}
@media (max-width:860px){.grid-2{grid-template-columns:1fr;}}
.pareto-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;}
.pareto-grid .card img{cursor:pointer;}
.meta-line{color:var(--text-muted);font-size:.85rem;margin:2px 0 18px;}
.stat-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px;}
.foot{text-align:center;color:var(--text-muted);font-size:.78rem;padding:26px 0 40px;}
ul.notes{padding-left:20px;line-height:1.6;}
.pg-controls{display:flex;align-items:center;gap:10px;padding:8px 2px 2px;}
.pg-controls button{padding:5px 12px;border-radius:7px;border:1px solid var(--border);background:var(--surface);
color:var(--text);cursor:pointer;font-size:.78rem;font-weight:600;}
.pg-controls button:hover:not(:disabled){border-color:var(--accent);}
.pg-controls button:disabled{opacity:.4;cursor:default;}
.pg-controls .pg-label{color:var(--text-muted);font-size:.78rem;}
"""

PAGE_JS = """
function showPanel(id, btn){
  document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('active');});
  document.getElementById(id).classList.add('active');
  document.querySelectorAll('.navbtn').forEach(function(b){b.classList.remove('active');});
  if(btn) btn.classList.add('active');
}
function showSub(groupId, subId, btn){
  var group = document.getElementById(groupId);
  group.querySelectorAll('.subpanel').forEach(function(p){p.classList.remove('active');});
  document.getElementById(subId).classList.add('active');
  group.querySelectorAll('.subtabbtn').forEach(function(b){b.classList.remove('active');});
  if(btn) btn.classList.add('active');
}
function showMca(select){
  document.querySelectorAll('.mca-panel').forEach(function(p){p.classList.remove('active');});
  document.getElementById(select.value).classList.add('active');
}
var __pg = {};
function pgInit(id, pageSize){
  var el = document.getElementById(id);
  if(!el) return;
  var rows = Array.prototype.slice.call(el.querySelectorAll('tbody tr'));
  var totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  __pg[id] = {rows: rows, pageSize: pageSize, page: 1, totalPages: totalPages};
  pgRender(id);
}
function pgRender(id){
  var st = __pg[id];
  var el = document.getElementById(id);
  st.rows.forEach(function(r, i){
    r.style.display = (i >= (st.page - 1) * st.pageSize && i < st.page * st.pageSize) ? '' : 'none';
  });
  var lbl = el.querySelector('.pg-label');
  if(lbl) lbl.textContent = 'Page ' + st.page + ' of ' + st.totalPages + ' (' + st.rows.length + ' rows)';
  var prevBtn = el.querySelector('.pg-prev'), nextBtn = el.querySelector('.pg-next');
  if(prevBtn) prevBtn.disabled = st.page <= 1;
  if(nextBtn) nextBtn.disabled = st.page >= st.totalPages;
}
function pgNav(id, delta){
  var st = __pg[id];
  st.page = Math.min(Math.max(1, st.page + delta), st.totalPages);
  pgRender(id);
}
"""


def render_ca_panel(var_a, var_b, ca_stats, ca_png):
    parts = ['<div class="stat-row">']
    parts.append(strength_badge(ca_stats["strength"]))
    parts.append(sig_badge(ca_stats["p"]))
    parts.append(f'<span class="muted">Cramer\'s V = {ca_stats["cramers_v"]:.3f} &middot; '
                  f'chi2 = {ca_stats["chi2"]:.2f} &middot; p = {ca_stats["p"]:.3g} &middot; N = {ca_stats["n"]}</span>')
    parts.append("</div>")
    parts.append('<div class="grid-2">')
    parts.append(f"<div>{img_html(ca_png, f'CA map {var_a} x {var_b}')}</div>")
    parts.append("<div>")
    parts.append("<h3>Most statistically associated cells</h3>")
    rows = [[r["rank"], r["a"], r["b"], r["observed"], r["expected"], f"{r['resid']:+.2f}", r["direction"]]
            for r in ca_stats["residual_records"]]
    parts.append(html_table(["Rank", var_a, var_b, "Observed", "Expected", "Std.Resid", "Direction"], rows))
    parts.append("</div></div>")
    return "".join(parts)


def render_lift_panel(cols, records):
    if not records:
        return '<p class="muted">No combination reached the minimum count.</p>'
    rows = []
    for r in records:
        combo_cells = [r["values"][c] for c in cols]
        rows.append([r["rank"], *combo_cells, r["count"], f"{r['support_pct']}%", r["expected_count"],
                     Raw(lift_badge(r["lift"]))])
    headers = ["Rank", *cols, "Count", "Support", "Expected", "Lift"]
    return html_table(headers, rows)


def render_combo_panel(cols, records):
    if not records:
        return '<p class="muted">No data.</p>'
    rows = []
    for r in records:
        combo_cells = [r["values"][c] for c in cols]
        rows.append([r["rank"], *combo_cells, r["count"], f"{r['duration_sec']:,.0f}", r["duration_hms"], f"{r['pct']}%"])
    headers = ["Rank", *cols, "Count", "Duration(sec)", "Duration(hh:mm:ss)", "% of Total"]
    return html_table(headers, rows)


def render_tpg_panel(rank_var, groups):
    parts = []
    for g in groups:
        parts.append(f'<div style="margin-bottom:14px"><strong>{esc(g["group_value"])}</strong> '
                      f'<span class="muted">({g["total_hms"]} total)</span>')
        rows = [[r["value"], r["count"], f"{r['duration_sec']:,.0f}", r["duration_hms"], f"{r['pct']}%"]
                for r in g["rows"]]
        parts.append(html_table([rank_var, "Count", "Duration(sec)", "Duration(hh:mm:ss)", "% of group"], rows))
        parts.append("</div>")
    return "".join(parts)


def render_variable_panel(var, uni_stats, pareto_png, pairwise, top_n):
    panel_id = f"panel-{safe(var)}"
    parts = [f'<div id="{panel_id}" class="panel"><h1>{esc(var)}</h1>']
    parts.append(f'<p class="meta-line">Univariate cause analysis, and pairwise CA / lift / top-{top_n} '
                  f'combinations against every other variable.</p>')

    parts.append('<div class="card"><h2>Univariate Pareto</h2><div class="grid-2">')
    rows = [[r["rank"], r["value"], r["count"], f"{r['duration_sec']:,.0f}", r["duration_hms"],
             f"{r['pct']}%", f"{r['cum_pct']}%"] for r in uni_stats["table_records"]]
    parts.append("<div>" + html_table(["Rank", var, "Count", "Duration(sec)", "Duration(hh:mm:ss)", "% Total", "Cum %"], rows) + "</div>")
    parts.append(f"<div>{img_html(pareto_png, f'Pareto {var}')}</div>")
    parts.append("</div></div>")

    parts.append('<div class="card"><h2>Compare with</h2>')
    group_id = f"cmp-{safe(var)}"
    others = list(pairwise.keys())
    parts.append('<div class="subtabs">')
    for i, other in enumerate(others):
        active = " active" if i == 0 else ""
        sub_id = f"{group_id}-{safe(other)}"
        parts.append(f'<button class="subtabbtn{active}" onclick="showSub(\'{group_id}\',\'{sub_id}\',this)">{esc(other)}</button>')
    parts.append(f'</div><div id="{group_id}">')
    for i, other in enumerate(others):
        active = " active" if i == 0 else ""
        sub_id = f"{group_id}-{safe(other)}"
        data = pairwise[other]
        parts.append(f'<div id="{sub_id}" class="subpanel{active}">')
        parts.append(render_ca_panel(var, other, data["ca_stats"], data["ca_png"]))
        parts.append(f'<h3 style="margin-top:18px">Lift analysis - {var} + {other}</h3>')
        parts.append(render_lift_panel([var, other], data["lift_records"]))
        parts.append(f'<h3 style="margin-top:18px">Top {top_n} {var} + {other} combinations</h3>')
        parts.append(render_combo_panel([var, other], data["combo_records"]))
        parts.append(f'<h3 style="margin-top:18px">Top {top_n} {other} per {var}</h3>')
        parts.append(render_tpg_panel(other, data["tpg_records"]))
        parts.append("</div>")
    parts.append("</div></div></div>")
    return "".join(parts)


def render_mca_combo(name, stats, png, label, top_n):
    parts = [f'<div id="mca-{name}" class="mca-panel subpanel">']
    parts.append(f'<div class="card"><h2>{esc(label)}</h2>')
    parts.append(f'<p class="meta-line">These variables together explain <strong>{stats["cum_pct"]:.1f}%</strong> '
                  f'of total inertia on the first 2 MCA dimensions.</p>')
    parts.append('<div class="grid-2">')
    parts.append(f"<div>{img_html(png, label)}</div>")
    parts.append("<div>")
    parts.append("<h3>Category coordinates & quality (cos2)</h3>")
    cat_headers = list(stats["category_records"][0].keys()) if stats["category_records"] else []
    cat_rows = [[r[h] for h in cat_headers] for r in stats["category_records"]]
    parts.append(html_table(cat_headers, cat_rows))
    parts.append("</div></div>")

    n_combo = len(stats["combo_records_all"])
    parts.append(f'<h3 style="margin-top:18px">All {n_combo} combinations (by downtime)</h3>')
    parts.append(render_combo_panel(stats["vars"], stats["combo_records_all"]))

    n_lift = len(stats["lift_records_all"])
    parts.append(f'<h3 style="margin-top:18px">All {n_lift} combinations by lift</h3>')
    parts.append(render_lift_panel(stats["vars"], stats["lift_records_all"]))

    n_close = len(stats["closest_records_all"])
    parts.append(f'<h3 style="margin-top:18px">All {n_close} closest cross-variable category pairs</h3>')
    rows = [[r["rank"], r["a"], r["b"], r["distance"]] for r in stats["closest_records_all"]]
    parts.append(html_table(["Rank", "Category A", "Category B", "Distance"], rows))
    parts.append("</div></div>")
    return "".join(parts)


def render_ca_ranking_table(sig_sorted, limit=5):
    if not sig_sorted:
        return '<p class="muted">No pair reached statistical significance (p&lt;0.05).</p>'
    rows = []
    for s in sig_sorted[:limit]:
        tr = s["top_residual"]
        cell = f"{s['var_a']}='{tr['a']}' & {s['var_b']}='{tr['b']}' ({tr['direction']})" if tr else "n/a"
        rows.append([f"{s['var_a']} x {s['var_b']}", f"{s['cramers_v']:.3f}", Raw(strength_badge(s["strength"])),
                     f"{s['p']:.3g}", cell])
    return html_table(["Pair", "Cramer's V", "Strength", "p-value", "Most surprising cell"], rows)


def render_lift_ranking_table(lift_sorted, limit=5):
    if not lift_sorted:
        return '<p class="muted">No combination reached the minimum count.</p>'
    rows = [[", ".join(f"{c}={v}" for c, v in s["values"].items()), Raw(lift_badge(s["lift"])), s["count"]]
            for s in lift_sorted[:limit]]
    return html_table(["Combination", "Lift", "Count"], rows)


def render_top3way_combo_table(combo_pool, limit=5):
    if not combo_pool:
        return '<p class="muted">No data.</p>'
    rows = []
    for rank, c in enumerate(combo_pool[:limit], start=1):
        combo_str = ", ".join(f"{k}={v}" for k, v in c["values"].items())
        rows.append([rank, c["label"], combo_str, f"{c['duration']:,.0f}", f"{c['pct']}%", c["count"]])
    return html_table(["Rank", "Analysis", "Combination", "Duration(sec)", "% of Total", "Count"], rows)


def render_top3way_lift_table(lift_pool, limit=5):
    if not lift_pool:
        return '<p class="muted">No data.</p>'
    rows = []
    for rank, c in enumerate(lift_pool[:limit], start=1):
        combo_str = ", ".join(f"{k}={v}" for k, v in c["values"].items())
        rows.append([rank, c["label"], combo_str, Raw(lift_badge(c["lift"])), c["count"]])
    return html_table(["Rank", "Analysis", "Combination", "Lift", "Count"], rows)


def render_interpretation_panel(csv_path, int_flag_col, int_dropped, total_duration, total_events,
                                 level_labels, univariate_stats, summary, mca_highlight_stats, top_n, min_count):
    biggest, sig_sorted, lift_sorted = summary["biggest"], summary["sig_sorted"], summary["lift_sorted"]
    parts = ['<div id="panel-interpretation" class="panel"><h1>Interpretation</h1>']
    parts.append(f'<p class="meta-line">Auto-generated from <code>{esc(os.path.basename(csv_path))}</code> - '
                  f'{total_events} events, {total_duration:,.0f} sec ({total_duration/3600:,.1f} hr) total downtime.</p>')

    parts.append('<div class="card"><h2>Data scope</h2><ul class="notes">')
    if int_flag_col is not None:
        parts.append(f"<li>Filtered to rows where <strong>{esc(int_flag_col)} = 1</strong> (counted interruptions); "
                      f"<strong>{int_dropped}</strong> row(s) with {esc(int_flag_col)}=0 excluded.</li>")
    else:
        parts.append("<li>No Int/interruption flag column was found - all rows were used.</li>")
    parts.append(f"<li><code>DurationLevel</code> categories (quantile bins): {esc(', '.join(level_labels))}.</li>")
    parts.append(f"<li>Associations use a minimum count of <strong>{min_count}</strong> events, top <strong>{top_n}</strong> shown.</li>")
    parts.append("</ul></div>")

    parts.append('<div class="card"><h2>Top single-cause contributors</h2>')
    rows = [[s["var"], s["top_value"], f"{s['top_pct']:.1f}%", f"{s['n80']} of {s['n_unique']}"]
            for s in univariate_stats if "top_value" in s]
    parts.append(html_table(["Variable", "Top value", "% of total downtime", "# values for 80%"], rows))
    if biggest:
        parts.append(f'<p style="margin-top:10px"><strong>Single biggest lever:</strong> '
                      f'<code>{esc(biggest["var"])}</code> = \'{esc(biggest["top_value"])}\' alone accounts for '
                      f'<strong>{biggest["top_pct"]:.1f}%</strong> of all downtime.</p>')
    parts.append("</div>")

    parts.append('<div class="card"><h2>Strongest pairwise associations (CA)</h2>')
    parts.append(render_ca_ranking_table(sig_sorted))
    parts.append("</div>")

    parts.append('<div class="card"><h2>Strongest associations by lift</h2>')
    if lift_sorted:
        parts.append(render_lift_ranking_table(lift_sorted))
        top = lift_sorted[0]
        combo_str = ", ".join(f"{c}='{v}'" for c, v in top["values"].items())
        parts.append(f'<p style="margin-top:10px"><strong>Strongest co-occurrence:</strong> {esc(combo_str)} happens '
                      f'<strong>{top["lift"]:.2f}x</strong> more often than random chance would predict.</p>')
    else:
        parts.append(f'<p class="muted">No pair had a combination reaching the minimum count of {min_count}.</p>')
    parts.append("</div>")

    parts.append(f'<div class="card"><h2>Multi-cause highlight: {esc(" + ".join(HIGHLIGHT_TRIPLE))}</h2><ul class="notes">')
    tc, tl, tp = mca_highlight_stats.get("top_combo"), mca_highlight_stats.get("top_lift"), mca_highlight_stats.get("top_closest_pair")
    if tc:
        combo_str = ", ".join(f"{c}='{v}'" for c, v in tc["values"].items())
        parts.append(f"<li>Largest raw combination: {esc(combo_str)} &rarr; {tc['duration']:,.0f} sec "
                      f"({tc['pct']:.1f}% of total downtime, {tc['count']} events).</li>")
    if tl:
        combo_str = ", ".join(f"{c}='{v}'" for c, v in tl["values"].items())
        parts.append(f"<li>Strongest lift combination: {esc(combo_str)} &rarr; {tl['lift']:.2f}x expected rate "
                      f"({tl['count']} events).</li>")
    if tp:
        parts.append(f"<li>Closest categories in the MCA map: {esc(tp['a'].replace('__', ': '))} and "
                      f"{esc(tp['b'].replace('__', ': '))} (distance {tp['dist']:.3f}).</li>")
    parts.append(f"<li>These 3 variables together explain <strong>{mca_highlight_stats['cum_pct']:.1f}%</strong> "
                 f"of total inertia on the first 2 MCA dimensions.</li>")
    parts.append("</ul></div>")

    parts.append('<div class="card"><h2>Notes &amp; recommendations</h2><ul class="notes">')
    parts.append("<li>Prioritize fixes in this order: the single biggest contributor, then the strongest "
                  "significant pairwise association, then the strongest lift combination.</li>")
    parts.append(f"<li>Associations built on fewer than {min_count} events were dropped from lift tables to avoid "
                  "over-reacting to one-off coincidences.</li>")
    parts.append("<li><code>DurationLevel</code> thresholds are quantile-based (relative to this dataset) - "
                  "re-run the script if the dataset changes substantially.</li>")
    parts.append("<li>A pair with Cramer's V near 1.0 between hierarchically related columns (e.g. Machine ID and "
                  "Machine Group) reflects a structural relationship, not a new finding.</li>")
    parts.append("</ul></div>")

    parts.append("</div>")
    return "".join(parts)


def build_html_report(path, meta, variables_ctx, mca_ctx, interpretation_ctx, top_n):
    _TABLE_SEQ[0] = 0
    _PAGINATION_INITS.clear()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    nav = ['<div class="navbar"><span class="brand">Downtime CA / MCA</span>']
    nav.append('<button class="navbtn active" onclick="showPanel(\'panel-overview\', this)">Overview</button>')
    for var in VARIABLES:
        nav.append(f'<button class="navbtn" onclick="showPanel(\'panel-{safe(var)}\', this)">{esc(var)}</button>')
    nav.append('<button class="navbtn" onclick="showPanel(\'panel-mca\', this)">Multi-Cause (MCA)</button>')
    nav.append('<button class="navbtn" onclick="showPanel(\'panel-interpretation\', this)">Interpretation</button>')
    nav.append("</div>")

    # ---- Overview ----
    overview = ['<div id="panel-overview" class="panel active"><h1>Overview</h1>']
    overview.append(f'<p class="meta-line">Source: <code>{esc(meta["csv_name"])}</code> &middot; generated {now}</p>')
    overview.append('<div class="card"><div class="kpi-grid">')
    overview.append(kpi("Total events", f'{meta["total_events"]:,}'))
    overview.append(kpi("Total downtime", f'{meta["total_duration"]/3600:,.1f} hr'))
    overview.append(kpi("Avg per event", f'{meta["total_duration"]/meta["total_events"]:,.0f} sec'))
    overview.append(kpi("Variables analyzed", str(len(VARIABLES))))
    if meta["int_flag_col"]:
        overview.append(kpi(f'Rows dropped ({meta["int_flag_col"]}=0)', str(meta["int_dropped"])))
    overview.append("</div></div>")

    summary = interpretation_ctx["summary"]
    overview.append('<div class="card"><h2>Top 5 pairwise associations (all 15 pairs)</h2>')
    overview.append("<h3>By statistical significance (Correspondence Analysis)</h3>")
    overview.append(render_ca_ranking_table(summary["sig_sorted"], limit=5))
    overview.append('<h3 style="margin-top:16px">By lift</h3>')
    overview.append(render_lift_ranking_table(summary["lift_sorted"], limit=5))
    overview.append("</div>")

    three_way = [mca_ctx["highlight"]] + mca_ctx["triples"]
    combo_pool = [{"label": t["label"], **t["stats"]["top_combo"]} for t in three_way if t["stats"].get("top_combo")]
    lift_pool = [{"label": t["label"], **t["stats"]["top_lift"]} for t in three_way if t["stats"].get("top_lift")]
    combo_pool.sort(key=lambda c: c["duration"], reverse=True)
    lift_pool.sort(key=lambda c: c["lift"], reverse=True)
    overview.append(f'<div class="card"><h2>Top 5 three-way (MCA) findings (across all {len(three_way)} combinations)</h2>')
    overview.append("<h3>By total downtime</h3>")
    overview.append(render_top3way_combo_table(combo_pool, limit=5))
    overview.append('<h3 style="margin-top:16px">By lift</h3>')
    overview.append(render_top3way_lift_table(lift_pool, limit=5))
    overview.append("</div>")

    overview.append('<div class="card"><h2>Pareto overview - click a variable to open its tab</h2><div class="pareto-grid">')
    for var in VARIABLES:
        img = img_html(variables_ctx[var]["pareto_png"], f"Pareto {var}")
        overview.append(f'<div class="card" onclick="showPanel(\'panel-{safe(var)}\')"><h3>{esc(var)}</h3>{img}</div>')
    overview.append("</div></div>")
    overview.append("</div>")

    # ---- Variable panels ----
    var_panels = [render_variable_panel(var, variables_ctx[var]["uni_stats"], variables_ctx[var]["pareto_png"],
                                         variables_ctx[var]["pairwise"], top_n)
                  for var in VARIABLES]

    # ---- MCA panel ----
    mca_options = [("mca-" + mca_ctx["highlight"]["name"], mca_ctx["highlight"]["label"] + "  (highlighted)")]
    mca_panels = [render_mca_combo(mca_ctx["highlight"]["name"], mca_ctx["highlight"]["stats"],
                                    mca_ctx["highlight"]["png"], mca_ctx["highlight"]["label"], top_n)]
    for t in mca_ctx["triples"]:
        mca_options.append(("mca-" + t["name"], t["label"]))
        mca_panels.append(render_mca_combo(t["name"], t["stats"], t["png"], t["label"], top_n))
    mca_options.append(("mca-" + mca_ctx["all"]["name"], mca_ctx["all"]["label"] + "  (all 6 variables)"))
    mca_panels.append(render_mca_combo(mca_ctx["all"]["name"], mca_ctx["all"]["stats"], mca_ctx["all"]["png"],
                                        mca_ctx["all"]["label"], top_n))
    mca_panels[0] = mca_panels[0].replace('class="mca-panel subpanel"', 'class="mca-panel subpanel active"', 1)

    mca_panel = ['<div id="panel-mca" class="panel"><h1>Multi-Cause Analysis (MCA)</h1>']
    mca_panel.append('<p class="meta-line">Pick any 3-way combination (or all 6 variables at once) to see its '
                      'category map, coordinates, top combinations, lift table, and closest category pairs.</p>')
    mca_panel.append('<select class="mcaselect" onchange="showMca(this)">')
    for i, (opt_id, opt_label) in enumerate(mca_options):
        sel = " selected" if i == 0 else ""
        mca_panel.append(f'<option value="{opt_id}"{sel}>{esc(opt_label)}</option>')
    mca_panel.append("</select>")
    mca_panel.extend(mca_panels)
    mca_panel.append("</div>")

    # ---- Interpretation panel ----
    interp_panel = render_interpretation_panel(**interpretation_ctx)

    body = "".join(nav) + "".join(overview) + "".join(var_panels) + "".join(mca_panel) + interp_panel
    body += f'<div class="foot">Generated by analyze_ca_mca.py on {now} &middot; fully offline, no data leaves this file.</div>'

    pg_init_script = "".join(f"pgInit('{tid}',{size});" for tid, size in _PAGINATION_INITS)

    doc = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Downtime CA / MCA Dashboard</title>"
        f"<style>{PAGE_CSS}</style></head><body>{body}"
        f"<script>{PAGE_JS}{pg_init_script}</script></body></html>"
    )
    write(path, doc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "cleaned_file.csv")


def main():
    parser = argparse.ArgumentParser(description="Statistical CA/MCA downtime analyzer (prince 0.14)")
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV,
                         help=f"Input CSV file (default: {DEFAULT_CSV})")
    parser.add_argument("--outdir", default="output", help="Output directory (default: output)")
    parser.add_argument("--top", type=int, default=5, help="Top-N for association/lift reports (default: 5)")
    parser.add_argument("--quantiles", type=int, default=4, help="Number of quantile bins for DurationLevel (default: 4)")
    parser.add_argument("--min-count", type=int, default=5, dest="min_count",
                         help="Minimum event count for a combination to appear in lift tables (default: 5)")
    parser.add_argument("--rebuild-html-only", action="store_true", dest="rebuild_html_only",
                         help="Skip re-running the analysis entirely; just rebuild report.html by reading back "
                              "<outdir>/report_data.json and the already-saved PNGs from a previous run.")
    args = parser.parse_args()

    if args.rebuild_html_only:
        json_path = os.path.join(args.outdir, "report_data.json")
        if not os.path.isfile(json_path):
            raise SystemExit(f"ERROR: {json_path} not found - run the script normally first to produce it.")
        data = load_report_data(json_path)
        html_path = os.path.join(args.outdir, "report.html")
        build_html_report(html_path, data["meta"], data["variables"], data["mca"], data["interpretation"], data["top_n"])
        print(f"Rebuilt {html_path} by reading {json_path} (no analysis re-run, no CSV touched).")
        return

    if not os.path.isfile(args.csv_path):
        raise SystemExit(f"ERROR: CSV file not found: {args.csv_path}")

    df, bad_rows, int_flag_col, int_dropped = load_data(args.csv_path)
    if df.empty:
        raise SystemExit("ERROR: no valid data rows found in the CSV.")

    df, level_labels, bin_edges = add_duration_level(df, q=args.quantiles)

    total_duration = df["Duration (sec)"].sum()
    total_events = len(df)
    top_n = args.top
    min_count = args.min_count
    outdir = args.outdir

    # ---------------- Summary ----------------
    summary = ["DOWNTIME CA / MCA ANALYSIS SUMMARY", "=" * 70,
               f"Input file           : {os.path.abspath(args.csv_path)}"]
    if int_flag_col is not None:
        summary.append(f"Int flag column      : '{int_flag_col}' found -> kept only rows where {int_flag_col}=1 "
                        f"(interruption counted); dropped {int_dropped} row(s) where it was 0.")
    else:
        summary.append("Int flag column      : none found - using all rows (no interruption filter applied).")
    summary.append(f"Total events (rows)  : {total_events}  (after Int filter and duration cleanup)")
    if bad_rows:
        summary.append(f"Rows skipped (bad/missing duration): {bad_rows}")
    summary.append(f"Total downtime       : {total_duration:,.0f} sec "
                    f"({total_duration/60:,.1f} min / {total_duration/3600:,.2f} hr) = {fmt_hms(total_duration)}")
    summary.append(f"Average per event    : {total_duration/total_events:,.1f} sec")
    summary.append("")
    summary.append(f"DurationLevel : quantile bins (q={args.quantiles}) on Duration (sec) -> {level_labels}")
    summary.append(f"Bin edges (sec): {[round(float(x), 1) for x in bin_edges]}")
    summary.append("")
    for v in VARIABLES:
        summary.append(f"Unique {v:<14}: {df[v].nunique()}")
    summary.append("")
    summary.append("Variables analyzed: " + ", ".join(VARIABLES))
    summary.append("")
    summary.append("Output layout:")
    summary.append("  INTERPRETATION.md          auto-generated takeaways and notes")
    summary.append("  report_data.json            everything report.html is built from (rebuild with --rebuild-html-only)")
    summary.append("  report.html                 interactive HTML dashboard (tabs, tables, maps)")
    summary.append("  <Variable>/                 descriptive CA + pairwise CA/lift/top-N vs every other variable")
    summary.append("  MultiWayMCA/                3-way MCA for every combination + all-6 MCA")
    summary.append(f"                              (highlighted: {' + '.join(HIGHLIGHT_TRIPLE)})")
    summary.append("  pivots/                     duration-sum pivot CSVs for every pair")
    write(os.path.join(outdir, "00_Summary.txt"), "\n".join(summary) + "\n")

    # ---------------- Per-variable folders ----------------
    ca_cache = {}
    combo_cache = {}
    lift_cache = {}
    univariate_stats = []
    variables_ctx = {}

    def get_ca(a, b):
        key = tuple(sorted((a, b)))
        if key not in ca_cache:
            plot_path = os.path.join(outdir, key[0], "pairwise", f"CA_{safe(key[0])}_x_{safe(key[1])}_biplot.png")
            text, stats = ca_report(df, key[0], key[1], top_n, plot_path=plot_path)
            ca_cache[key] = (text, stats, plot_path)
        return ca_cache[key]

    def get_combo(a, b):
        key = tuple(sorted((a, b)))
        if key not in combo_cache:
            combo_cache[key] = combo_top_report(df, list(key), total_duration, top_n)
        return combo_cache[key]

    def get_lift(a, b):
        key = tuple(sorted((a, b)))
        if key not in lift_cache:
            table = lift_table(df, list(key), min_count)
            text = render_lift_report(list(key), table, top_n, min_count)
            records = lift_table_records(table, list(key), top_n)
            top = {"values": records[0]["values"], "lift": records[0]["lift"], "count": records[0]["count"]} if records else None
            lift_cache[key] = (text, top, records)
        return lift_cache[key]

    for var in VARIABLES:
        var_dir = os.path.join(outdir, var)
        uni_text, uni_stats = univariate_report(df, var, total_duration, total_events)
        write(os.path.join(var_dir, f"00_CA_univariate_{safe(var)}.txt"), uni_text)
        pareto_png = os.path.join(var_dir, f"00_CA_univariate_{safe(var)}_pareto.png")
        plot_pareto(df, var, total_duration, pareto_png)
        univariate_stats.append(uni_stats)
        variables_ctx[var] = {"uni_stats": uni_stats, "pareto_png": pareto_png, "pairwise": {}}

        for other in VARIABLES:
            if other == var:
                continue
            ca_text, ca_stats, ca_plot_src = get_ca(var, other)
            ca_dest_txt = os.path.join(var_dir, "pairwise", f"CA_{safe(var)}_x_{safe(other)}.txt")
            ca_dest_png = os.path.join(var_dir, "pairwise", f"CA_{safe(var)}_x_{safe(other)}_biplot.png")
            write(ca_dest_txt, ca_text)
            if os.path.abspath(ca_dest_png) != os.path.abspath(ca_plot_src):
                os.makedirs(os.path.dirname(ca_dest_png), exist_ok=True)
                shutil.copyfile(ca_plot_src, ca_dest_png)

            lift_text, _, lift_records = get_lift(var, other)
            write(os.path.join(var_dir, "pairwise", f"Lift_{safe(var)}_x_{safe(other)}.txt"), lift_text)

            combo_text, combo_records = get_combo(var, other)
            write(os.path.join(var_dir, "pairwise", f"Top{top_n}_{safe(var)}_x_{safe(other)}_combo.txt"), combo_text)

            tpg_text, tpg_records = top_n_per_group_report(df, var, other, top_n)
            write(os.path.join(var_dir, "pairwise", f"Top{top_n}_{safe(other)}_per_{safe(var)}.txt"), tpg_text)

            # For the HTML "Compare with" panel, combo/lift columns are always [var, other]
            # regardless of the cache's canonical (sorted) key order - flip if needed.
            def _reorder(records, first, second):
                if not records or list(records[0]["values"].keys())[0] == first:
                    return records
                return [{**r, "values": {first: r["values"][first], second: r["values"][second]}} for r in records]

            variables_ctx[var]["pairwise"][other] = {
                "ca_stats": ca_stats if ca_stats["var_a"] == var else {
                    **ca_stats, "var_a": var, "var_b": other,
                    "residual_records": [{**r, "a": r["b"], "b": r["a"]} for r in ca_stats["residual_records"]],
                },
                "ca_png": ca_dest_png,
                "lift_records": _reorder(lift_records, var, other),
                "combo_records": _reorder(combo_records, var, other),
                "tpg_records": tpg_records,
            }

    ca_stats_all = [entry[1] for entry in ca_cache.values()]
    lift_stats_all = [entry[1] for entry in lift_cache.values() if entry[1] is not None]

    # ---------------- Pivot CSVs (all pairs) ----------------
    for a, b in combinations(VARIABLES, 2):
        write_pivot_csv(os.path.join(outdir, "pivots", f"{safe(a)}_x_{safe(b)}_pivot.csv"), df, a, b)

    # ---------------- Multi-way MCA ----------------
    triples = [t for t in combinations(VARIABLES, 3) if set(t) != set(HIGHLIGHT_TRIPLE)]
    mca_ctx = {"triples": []}

    highlight_name = "00_MCA_" + "_".join(safe(v) for v in HIGHLIGHT_TRIPLE)
    highlight_png = os.path.join(outdir, "MultiWayMCA", f"{highlight_name}_map.png")
    highlight_text, highlight_stats = mca_report(df, HIGHLIGHT_TRIPLE, total_duration, top_n, min_count,
                                                  plot_path=highlight_png)
    write(os.path.join(outdir, "MultiWayMCA", f"{highlight_name}.txt"), highlight_text)
    mca_ctx["highlight"] = {"name": highlight_name, "stats": highlight_stats, "png": highlight_png,
                             "label": " x ".join(HIGHLIGHT_TRIPLE)}

    for t in triples:
        name = "MCA_triple_" + "_".join(safe(v) for v in t)
        png = os.path.join(outdir, "MultiWayMCA", f"{name}_map.png")
        text, stats = mca_report(df, t, total_duration, top_n, min_count, plot_path=png)
        write(os.path.join(outdir, "MultiWayMCA", f"{name}.txt"), text)
        mca_ctx["triples"].append({"name": name, "stats": stats, "png": png, "label": " x ".join(t)})

    all_png = os.path.join(outdir, "MultiWayMCA", "MCA_all_variables_map.png")
    all_text, all_stats = mca_report(df, VARIABLES, total_duration, top_n, min_count, plot_path=all_png)
    write(os.path.join(outdir, "MultiWayMCA", "MCA_all_variables.txt"), all_text)
    mca_ctx["all"] = {"name": "MCA_all_variables", "stats": all_stats, "png": all_png, "label": " x ".join(VARIABLES)}

    # ---------------- Interpretation ----------------
    summary_stats = compute_interpretation_summary(univariate_stats, ca_stats_all, lift_stats_all)
    interpretation = build_interpretation_md(
        args.csv_path, int_flag_col, int_dropped, total_duration, total_events, level_labels,
        univariate_stats, summary_stats, highlight_stats, top_n, min_count)
    write(os.path.join(outdir, "INTERPRETATION.md"), interpretation)

    # ---------------- HTML dashboard ----------------
    # Every number/table the HTML needs is written to report_data.json (a plain
    # file in --outdir, next to the txt/csv reports) and then read back from
    # disk before rendering - report.html is built strictly from what is on
    # disk in this folder, the same as the .txt/.csv files, not from anything
    # kept only in memory during the analysis run.
    meta = {"csv_name": os.path.basename(args.csv_path), "total_events": total_events,
            "total_duration": total_duration, "int_flag_col": int_flag_col, "int_dropped": int_dropped}
    interpretation_ctx = dict(csv_path=args.csv_path, int_flag_col=int_flag_col, int_dropped=int_dropped,
                              total_duration=total_duration, total_events=total_events, level_labels=level_labels,
                              univariate_stats=univariate_stats, summary=summary_stats,
                              mca_highlight_stats=highlight_stats, top_n=top_n, min_count=min_count)

    report_data_path = os.path.join(outdir, "report_data.json")
    save_report_data(report_data_path, {
        "meta": meta, "variables": variables_ctx, "mca": mca_ctx,
        "interpretation": interpretation_ctx, "top_n": top_n,
    })
    data = load_report_data(report_data_path)
    build_html_report(os.path.join(outdir, "report.html"), data["meta"], data["variables"], data["mca"],
                       data["interpretation"], data["top_n"])

    n_pairs = len(list(combinations(VARIABLES, 2)))
    n_triples = len(list(combinations(VARIABLES, 3)))
    print(f"Done. Wrote reports to: {os.path.abspath(outdir)}")
    print(f"  - {len(VARIABLES)} per-variable folders (univariate + {n_pairs} pairwise CA/Lift reports + plots)")
    print(f"  - {n_triples} 3-way MCA reports (incl. highlighted {HIGHLIGHT_TRIPLE}) + 1 all-variables MCA (+ maps)")
    print(f"  - {n_pairs} pivot CSVs")
    print("  - report_data.json  (the file report.html is built from - inspect it, or re-run with")
    print("                       --rebuild-html-only to regenerate report.html from it without recomputing)")
    print("  - INTERPRETATION.md")
    print(f"  - report.html  <- open this in a browser")


if __name__ == "__main__":
    main()
