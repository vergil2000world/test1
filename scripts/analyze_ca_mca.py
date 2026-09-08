#!/usr/bin/env python3
"""
Statistical CA / MCA downtime analyzer (uses the `prince` library, 0.14.0).

Input CSV columns: Machine ID, Duration (sec), ReasonCode, Lot ID,
Product Code, Machine Group.

What it does
------------
1. Adds a derived categorical feature `DurationLevel` (Low/Medium/High/
   Critical) by cutting `Duration (sec)` into quantile bins (pandas.qcut,
   q=4 by default -> quartiles).

2. Treats these 6 columns as the analysis variables:
   Machine ID, Machine Group, ReasonCode, Product Code, Lot ID, DurationLevel

3. CA  (Correspondence Analysis)          -> for EVERY pair of variables
   (all C(6,2)=15 combinations): builds the contingency table, runs
   prince.CA, reports the chi-square test / Cramer's V, eigenvalues
   (explained inertia), row/column coordinates, and the top-N
   statistically most-associated cells (largest standardized residuals).

4. MCA (Multiple Correspondence Analysis) -> for every 3-way combination
   of variables (all C(6,3)=20), for the specific combination the user
   asked for (Machine Group + ReasonCode + DurationLevel), and for all 6
   variables at once: runs prince.MCA, reports eigenvalues, category
   coordinates/cos2, the top-N raw combinations by total downtime, and
   the top-N closest cross-variable category pairs in the MCA map (the
   categories that "travel together" most).

5. Output layout (under --outdir, default "output"):
     output/
       00_Summary.txt
       <Variable>/                       one folder per variable (6)
         00_CA_univariate_<Variable>.txt   full Pareto detail
         pairwise/
           CA_<Variable>_x_<Other>.txt     statistical CA (prince)
           Top<N>_<Variable>_x_<Other>_combo.txt   raw top-N combo table
           Top<N>_<Other>_per_<Variable>.txt       top-N Other per each Variable value
       MultiWayMCA/
         00_MCA_MachineGroup_ReasonCode_DurationLevel.txt   (the requested combo)
         MCA_triple_<A>_<B>_<C>.txt         all other 3-way combinations
         MCA_all_variables.txt              all 6 variables at once
       pivots/
         <A>_x_<B>_pivot.csv                duration-sum matrix, all 15 pairs

Requirements: pandas, numpy, scipy, scikit-learn==1.5.2, prince==0.14.0
(see requirements.txt - prince 0.14.0 needs scikit-learn<1.6, newer
scikit-learn removed an internal API prince 0.14.0 relies on).

Usage:
    python3 analyze_ca_mca.py INPUT.csv --outdir output --top 5 --quantiles 4
"""

import argparse
import os
from itertools import combinations

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

VARIABLES = ["Machine ID", "Machine Group", "ReasonCode", "Product Code", "Lot ID", "DurationLevel"]
HIGHLIGHT_TRIPLE = ("Machine Group", "ReasonCode", "DurationLevel")


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


def load_data(csv_path):
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    colmap = resolve_columns(raw.columns)
    df = raw.rename(columns={v: k for k, v in colmap.items()})[list(COLUMN_ALIASES.keys())].copy()

    n_before = len(df)
    df["Duration (sec)"] = pd.to_numeric(
        df["Duration (sec)"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    df = df.dropna(subset=["Duration (sec)"]).copy()
    bad_rows = n_before - len(df)

    for col in ["Machine ID", "ReasonCode", "Lot ID", "Product Code", "Machine Group"]:
        df[col] = df[col].fillna("(blank)").astype(str).str.strip().replace("", "(blank)")

    return df, bad_rows


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
    lines.append("")
    return "\n".join(lines)


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
# Statistical CA (pairwise, prince) - the core "Correspondence Analysis"
# ---------------------------------------------------------------------------

def ca_report(df, var_a, var_b, top_n=5):
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
    for rank, ((a_val, b_val), resid_val) in enumerate(flat.items(), start=1):
        obs = ct.loc[a_val, b_val]
        exp = expected[ct.index.get_loc(a_val), ct.columns.get_loc(b_val)]
        direction = "over-represented" if resid_val > 0 else "under-represented"
        rows.append([rank, a_val, b_val, int(obs), f"{exp:.1f}", f"{resid_val:+.2f}", direction])
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

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical MCA (3+ variables, prince)
# ---------------------------------------------------------------------------

def mca_report(df, vars_list, total_duration, top_n=5):
    X = df[list(vars_list)].astype(str)
    mca = prince.MCA(n_components=2, random_state=42).fit(X)

    label = " x ".join(vars_list)
    lines = [f"MULTIPLE CORRESPONDENCE ANALYSIS (MCA) - {label}", "=" * 70,
             f"Variables: {', '.join(vars_list)}   N = {len(X)} events", ""]

    lines.append("Eigenvalues / explained inertia")
    lines.append("-" * 40)
    lines.append(mca.eigenvalues_summary.to_string())
    lines.append("")

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
    lines.append(combo_top_report(df, list(vars_list), total_duration, top_n))

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

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Statistical CA/MCA downtime analyzer (prince 0.14)")
    parser.add_argument("csv_path", help="Input CSV file")
    parser.add_argument("--outdir", default="output", help="Output directory (default: output)")
    parser.add_argument("--top", type=int, default=5, help="Top-N for association reports (default: 5)")
    parser.add_argument("--quantiles", type=int, default=4, help="Number of quantile bins for DurationLevel (default: 4)")
    args = parser.parse_args()

    df, bad_rows = load_data(args.csv_path)
    if df.empty:
        raise SystemExit("ERROR: no valid data rows found in the CSV.")

    df, level_labels, bin_edges = add_duration_level(df, q=args.quantiles)

    total_duration = df["Duration (sec)"].sum()
    total_events = len(df)
    top_n = args.top
    outdir = args.outdir

    # ---------------- Summary ----------------
    summary = ["DOWNTIME CA / MCA ANALYSIS SUMMARY", "=" * 70,
               f"Input file           : {os.path.abspath(args.csv_path)}",
               f"Total events (rows)  : {total_events}"]
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
    summary.append("  <Variable>/                 descriptive CA + pairwise CA/top-N vs every other variable")
    summary.append("  MultiWayMCA/                3-way MCA for every combination + all-6 MCA")
    summary.append(f"                              (highlighted: {' + '.join(HIGHLIGHT_TRIPLE)})")
    summary.append("  pivots/                     duration-sum pivot CSVs for every pair")
    write(os.path.join(outdir, "00_Summary.txt"), "\n".join(summary) + "\n")

    # ---------------- Per-variable folders ----------------
    ca_cache = {}
    combo_cache = {}

    def get_ca(a, b):
        key = tuple(sorted((a, b)))
        if key not in ca_cache:
            ca_cache[key] = ca_report(df, key[0], key[1], top_n)
        return ca_cache[key]

    def get_combo(a, b):
        key = tuple(sorted((a, b)))
        if key not in combo_cache:
            combo_cache[key] = combo_top_report(df, list(key), total_duration, top_n)
        return combo_cache[key]

    for var in VARIABLES:
        var_dir = os.path.join(outdir, var)
        write(os.path.join(var_dir, f"00_CA_univariate_{safe(var)}.txt"),
              univariate_report(df, var, total_duration, total_events))

        for other in VARIABLES:
            if other == var:
                continue
            write(os.path.join(var_dir, "pairwise", f"CA_{safe(var)}_x_{safe(other)}.txt"), get_ca(var, other))
            write(os.path.join(var_dir, "pairwise", f"Top{top_n}_{safe(var)}_x_{safe(other)}_combo.txt"), get_combo(var, other))
            write(os.path.join(var_dir, "pairwise", f"Top{top_n}_{safe(other)}_per_{safe(var)}.txt"),
                  top_n_per_group_report(df, var, other, top_n))

    # ---------------- Pivot CSVs (all pairs) ----------------
    for a, b in combinations(VARIABLES, 2):
        write_pivot_csv(os.path.join(outdir, "pivots", f"{safe(a)}_x_{safe(b)}_pivot.csv"), df, a, b)

    # ---------------- Multi-way MCA ----------------
    triples = [t for t in combinations(VARIABLES, 3) if set(t) != set(HIGHLIGHT_TRIPLE)]

    write(os.path.join(outdir, "MultiWayMCA", "00_MCA_" + "_".join(safe(v) for v in HIGHLIGHT_TRIPLE) + ".txt"),
          mca_report(df, HIGHLIGHT_TRIPLE, total_duration, top_n))

    for t in triples:
        write(os.path.join(outdir, "MultiWayMCA", "MCA_triple_" + "_".join(safe(v) for v in t) + ".txt"),
              mca_report(df, t, total_duration, top_n))

    write(os.path.join(outdir, "MultiWayMCA", "MCA_all_variables.txt"),
          mca_report(df, VARIABLES, total_duration, top_n))

    n_pairs = len(list(combinations(VARIABLES, 2)))
    n_triples = len(list(combinations(VARIABLES, 3)))
    print(f"Done. Wrote reports to: {os.path.abspath(outdir)}")
    print(f"  - {len(VARIABLES)} per-variable folders (univariate + {n_pairs} pairwise CA reports)")
    print(f"  - {n_triples} 3-way MCA reports (incl. highlighted {HIGHLIGHT_TRIPLE}) + 1 all-variables MCA")
    print(f"  - {n_pairs} pivot CSVs")


if __name__ == "__main__":
    main()
