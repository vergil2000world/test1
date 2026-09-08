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

9. Output layout (under --outdir, default "output"):
     output/
       00_Summary.txt
       INTERPRETATION.md                  auto-generated takeaways (step 8)
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
"""

import argparse
import os
import shutil
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

    rows, cum = [], 0.0
    for rank, (value, r) in enumerate(g.iterrows(), start=1):
        pct = (r["sum"] / total_duration * 100) if total_duration else 0.0
        cum += pct
        rows.append([rank, value, int(r["count"]), f"{r['sum']:,.0f}", fmt_hms(r["sum"]), f"{pct:.1f}%", f"{cum:.1f}%"])
    headers = ["Rank", var, "Count", "Duration(sec)", "Duration(hh:mm:ss)", "% Total", "Cum %"]
    aligns = ["<", "<", ">", ">", ">", ">", ">"]
    lines.extend(fmt_table(headers, rows, aligns))
    lines.append("")

    stats = {"var": var, "n_unique": len(g)}
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
    rows = []
    for rank, (key, r) in enumerate(g.iterrows(), start=1):
        key = key if isinstance(key, tuple) else (key,)
        pct = (r["sum"] / total_duration * 100) if total_duration else 0.0
        rows.append([rank, *key, int(r["count"]), f"{r['sum']:,.0f}", fmt_hms(r["sum"]), f"{pct:.1f}%"])
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
    return "\n".join(lines)


def top_n_per_group_report(df, group_var, rank_var, top_n):
    lines = [f"TOP {top_n} {rank_var} PER {group_var}", "=" * 70, ""]
    totals = df.groupby(group_var)["Duration (sec)"].sum().sort_values(ascending=False)
    for g_value, g_total in totals.items():
        sub = df[df[group_var] == g_value].groupby(rank_var)["Duration (sec)"].agg(["count", "sum"])
        sub = sub.sort_values("sum", ascending=False).head(top_n)
        lines.append(f"{group_var}: {g_value}  (total downtime {g_total:,.0f} sec / {fmt_hms(g_total)})")
        rows = []
        for value, r in sub.iterrows():
            pct = (r["sum"] / g_total * 100) if g_total else 0.0
            rows.append([value, int(r["count"]), f"{r['sum']:,.0f}", fmt_hms(r["sum"]), f"{pct:.1f}%"])
        headers = [rank_var, "Count", "Duration(sec)", "Duration(hh:mm:ss)", "% of group total"]
        aligns = ["<", ">", ">", ">", ">"]
        for line in fmt_table(headers, rows, aligns):
            lines.append("    " + line)
        lines.append("")
    return "\n".join(lines)


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
    rows = []
    top_residual = None
    for rank, ((a_val, b_val), resid_val) in enumerate(flat.items(), start=1):
        obs = ct.loc[a_val, b_val]
        exp = expected[ct.index.get_loc(a_val), ct.columns.get_loc(b_val)]
        direction = "over-represented" if resid_val > 0 else "under-represented"
        rows.append([rank, a_val, b_val, int(obs), f"{exp:.1f}", f"{resid_val:+.2f}", direction])
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

    # Raw top-N combination by total downtime (practical / easy to read)
    combo_text = combo_top_report(df, list(vars_list), total_duration, top_n)
    lines.append(combo_text)
    g = df.groupby(list(vars_list))["Duration (sec)"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    top_combo = None
    if len(g):
        key = g.index[0]
        key = key if isinstance(key, tuple) else (key,)
        top_combo = {"values": dict(zip(vars_list, key)), "count": int(g.iloc[0]["count"]),
                     "duration": float(g.iloc[0]["sum"]),
                     "pct": float(g.iloc[0]["sum"] / total_duration * 100) if total_duration else 0.0}

    # Lift analysis for the same set of variables
    lift = lift_table(df, list(vars_list), min_count)
    lines.append(render_lift_report(list(vars_list), lift, top_n, min_count))
    top_lift = None
    if len(lift):
        r = lift.iloc[0]
        top_lift = {"values": {c: r[c] for c in vars_list}, "lift": float(r["lift"]), "count": int(r["count"])}

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
    rows = []
    for rank, (a, b, d) in enumerate(dist_rows[:top_n], start=1):
        rows.append([rank, a.replace("__", ": "), b.replace("__", ": "), f"{d:.3f}"])
    headers = ["Rank", "Category A", "Category B", "Distance (smaller = more associated)"]
    aligns = ["<", "<", "<", ">"]
    lines.extend(fmt_table(headers, rows, aligns))
    lines.append("")

    if plot_path:
        plot_mca_map(coords, list(vars_list), plot_path)

    stats = {
        "vars": list(vars_list), "cum_pct": cum_pct, "top_combo": top_combo, "top_lift": top_lift,
        "top_closest_pair": {"a": dist_rows[0][0], "b": dist_rows[0][1], "dist": float(dist_rows[0][2])} if dist_rows else None,
    }
    return "\n".join(lines), stats


# ---------------------------------------------------------------------------
# Interpretation (auto-generated markdown takeaways)
# ---------------------------------------------------------------------------

def build_interpretation_md(csv_path, int_flag_col, int_dropped, total_duration, total_events,
                             level_labels, univariate_stats, ca_stats_all, lift_stats_all,
                             mca_highlight_stats, top_n, min_count):
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
    if univariate_stats:
        biggest = max((s for s in univariate_stats if "top_pct" in s), key=lambda s: s["top_pct"], default=None)
        if biggest:
            lines.append(f"**Single biggest lever:** `{biggest['var']}` = '{biggest['top_value']}' alone accounts for "
                          f"**{biggest['top_pct']:.1f}%** of all downtime - the highest-leverage single-variable fix "
                          f"available in this dataset.")
    lines.append("")

    lines.append("## Strongest pairwise associations (Correspondence Analysis)")
    lines.append("")
    sig = [s for s in ca_stats_all if s["p"] < 0.05]
    sig_sorted = sorted(sig, key=lambda s: s["cramers_v"], reverse=True)
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
    lift_sorted = sorted(lift_stats_all, key=lambda s: s["lift"], reverse=True)
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
    lines.append("- See the `<Variable>/` folders for full detail and PNG plots (Pareto charts, CA biplots, "
                  "MCA category maps) behind every number in this summary.")
    lines.append("- A pair with Cramer's V near 1.0 (e.g. Machine ID x Machine Group) can simply reflect a "
                  "structural/hierarchical relationship already known by construction (each machine belongs to "
                  "exactly one group) rather than a new finding - treat those as expected, not actionable.")
    lines.append("")

    return "\n".join(lines)


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
    args = parser.parse_args()

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
            top = None
            if len(table):
                r = table.iloc[0]
                top = {"values": {key[0]: r[key[0]], key[1]: r[key[1]]}, "lift": float(r["lift"]), "count": int(r["count"])}
            lift_cache[key] = (text, top)
        return lift_cache[key]

    for var in VARIABLES:
        var_dir = os.path.join(outdir, var)
        uni_text, uni_stats = univariate_report(df, var, total_duration, total_events)
        write(os.path.join(var_dir, f"00_CA_univariate_{safe(var)}.txt"), uni_text)
        plot_pareto(df, var, total_duration, os.path.join(var_dir, f"00_CA_univariate_{safe(var)}_pareto.png"))
        univariate_stats.append(uni_stats)

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

            lift_text, _ = get_lift(var, other)
            write(os.path.join(var_dir, "pairwise", f"Lift_{safe(var)}_x_{safe(other)}.txt"), lift_text)

            write(os.path.join(var_dir, "pairwise", f"Top{top_n}_{safe(var)}_x_{safe(other)}_combo.txt"), get_combo(var, other))
            write(os.path.join(var_dir, "pairwise", f"Top{top_n}_{safe(other)}_per_{safe(var)}.txt"),
                  top_n_per_group_report(df, var, other, top_n))

    ca_stats_all = [entry[1] for entry in ca_cache.values()]
    lift_stats_all = [entry[1] for entry in lift_cache.values() if entry[1] is not None]

    # ---------------- Pivot CSVs (all pairs) ----------------
    for a, b in combinations(VARIABLES, 2):
        write_pivot_csv(os.path.join(outdir, "pivots", f"{safe(a)}_x_{safe(b)}_pivot.csv"), df, a, b)

    # ---------------- Multi-way MCA ----------------
    triples = [t for t in combinations(VARIABLES, 3) if set(t) != set(HIGHLIGHT_TRIPLE)]

    highlight_name = "00_MCA_" + "_".join(safe(v) for v in HIGHLIGHT_TRIPLE)
    highlight_text, highlight_stats = mca_report(
        df, HIGHLIGHT_TRIPLE, total_duration, top_n, min_count,
        plot_path=os.path.join(outdir, "MultiWayMCA", f"{highlight_name}_map.png"))
    write(os.path.join(outdir, "MultiWayMCA", f"{highlight_name}.txt"), highlight_text)

    for t in triples:
        name = "MCA_triple_" + "_".join(safe(v) for v in t)
        text, _ = mca_report(df, t, total_duration, top_n, min_count,
                              plot_path=os.path.join(outdir, "MultiWayMCA", f"{name}_map.png"))
        write(os.path.join(outdir, "MultiWayMCA", f"{name}.txt"), text)

    all_text, _ = mca_report(df, VARIABLES, total_duration, top_n, min_count,
                              plot_path=os.path.join(outdir, "MultiWayMCA", "MCA_all_variables_map.png"))
    write(os.path.join(outdir, "MultiWayMCA", "MCA_all_variables.txt"), all_text)

    # ---------------- Interpretation ----------------
    interpretation = build_interpretation_md(
        args.csv_path, int_flag_col, int_dropped, total_duration, total_events, level_labels,
        univariate_stats, ca_stats_all, lift_stats_all, highlight_stats, top_n, min_count)
    write(os.path.join(outdir, "INTERPRETATION.md"), interpretation)

    n_pairs = len(list(combinations(VARIABLES, 2)))
    n_triples = len(list(combinations(VARIABLES, 3)))
    print(f"Done. Wrote reports to: {os.path.abspath(outdir)}")
    print(f"  - {len(VARIABLES)} per-variable folders (univariate + {n_pairs} pairwise CA/Lift reports + plots)")
    print(f"  - {n_triples} 3-way MCA reports (incl. highlighted {HIGHLIGHT_TRIPLE}) + 1 all-variables MCA (+ maps)")
    print(f"  - {n_pairs} pivot CSVs")
    print(f"  - INTERPRETATION.md")


if __name__ == "__main__":
    main()
