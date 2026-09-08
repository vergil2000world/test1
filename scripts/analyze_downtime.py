#!/usr/bin/env python3
"""
Downtime CA / MCA analyzer.

Reads a CSV of machine downtime events (columns: Machine ID, Duration (sec),
ReasonCode, Lot ID, Product Code, Machine Group) and produces:

  - CA  (Cause Analysis)       : full Pareto breakdown for each single
                                  dimension (ReasonCode, Machine ID,
                                  Machine Group, Product Code, Lot ID),
                                  ranked by total downtime, with count,
                                  duration, % of total and cumulative %.

  - MCA (Multiple Cause Analysis): top-N most associated combinations
                                  across pairs of dimensions (e.g. which
                                  Machine + ReasonCode pairs cause the most
                                  downtime, which Machine Group + ReasonCode
                                  pairs, etc).

Usage:
    python3 analyze_downtime.py INPUT.csv [--outdir reports] [--top 5]

Only uses the Python standard library - no third-party packages required.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Column handling
# ---------------------------------------------------------------------------

# Canonical dimension names -> list of accepted header aliases (normalized:
# lowercased, spaces/underscores/parentheses stripped).
COLUMN_ALIASES = {
    "Machine ID": ["machineid", "machine"],
    "Duration (sec)": ["durationsec", "duration", "durations", "durationseconds"],
    "ReasonCode": ["reasoncode", "reason"],
    "Lot ID": ["lotid", "lot"],
    "Product Code": ["productcode", "product"],
    "Machine Group": ["machinegroup", "group"],
}

DIMENSIONS = ["ReasonCode", "Machine ID", "Machine Group", "Product Code", "Lot ID"]

COMBO_REPORTS = [
    ("Machine ID", "ReasonCode"),
    ("Machine Group", "ReasonCode"),
    ("Product Code", "ReasonCode"),
    ("Lot ID", "ReasonCode"),
    ("Machine Group", "Machine ID"),
]


def _normalize(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def resolve_columns(fieldnames):
    """Map canonical column names to the actual header found in the CSV."""
    normalized = {_normalize(f): f for f in fieldnames}
    resolved = {}
    missing = []
    for canonical, aliases in COLUMN_ALIASES.items():
        candidates = [canonical] + aliases
        found = None
        for cand in candidates:
            key = _normalize(cand)
            if key in normalized:
                found = normalized[key]
                break
        if found is None:
            missing.append(canonical)
        else:
            resolved[canonical] = found
    if missing:
        raise SystemExit(
            "ERROR: could not find required column(s) in the CSV: "
            + ", ".join(missing)
            + f"\nHeaders found: {fieldnames}"
        )
    return resolved


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("ERROR: CSV appears to have no header row.")
        colmap = resolve_columns(reader.fieldnames)

        rows = []
        bad_duration = 0
        for raw in reader:
            dur_raw = (raw.get(colmap["Duration (sec)"]) or "").strip()
            try:
                duration = float(dur_raw.replace(",", ""))
            except ValueError:
                bad_duration += 1
                continue

            rows.append({
                "Machine ID": (raw.get(colmap["Machine ID"]) or "").strip() or "(blank)",
                "Duration (sec)": duration,
                "ReasonCode": (raw.get(colmap["ReasonCode"]) or "").strip() or "(blank)",
                "Lot ID": (raw.get(colmap["Lot ID"]) or "").strip() or "(blank)",
                "Product Code": (raw.get(colmap["Product Code"]) or "").strip() or "(blank)",
                "Machine Group": (raw.get(colmap["Machine Group"]) or "").strip() or "(blank)",
            })

    return rows, bad_duration


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_hms(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_table(headers, rows, aligns=None):
    """rows: list of list-of-str, already formatted as strings."""
    n = len(headers)
    aligns = aligns or ["<"] * n
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        return "  ".join(f"{c:{aligns[i]}{widths[i]}}" for i, c in enumerate(cells))

    lines = [fmt_row(headers), "  ".join("-" * w for w in widths)]
    for r in rows:
        lines.append(fmt_row(r))
    return lines


# ---------------------------------------------------------------------------
# CA - single-dimension Pareto (Cause Analysis)
# ---------------------------------------------------------------------------

def pareto(rows, dimension, total_duration):
    agg = defaultdict(lambda: [0, 0.0])  # value -> [count, duration]
    for r in rows:
        key = r[dimension]
        agg[key][0] += 1
        agg[key][1] += r["Duration (sec)"]

    items = sorted(agg.items(), key=lambda kv: kv[1][1], reverse=True)
    out = []
    cum = 0.0
    for rank, (value, (count, duration)) in enumerate(items, start=1):
        pct = (duration / total_duration * 100) if total_duration else 0.0
        cum += pct
        out.append({
            "rank": rank,
            "value": value,
            "count": count,
            "duration": duration,
            "pct": pct,
            "cum_pct": cum,
        })
    return out


def render_ca_report(dimension, entries, total_duration, total_events):
    lines = []
    lines.append(f"CAUSE ANALYSIS (CA) - by {dimension}")
    lines.append("=" * 70)
    lines.append(f"Total events   : {total_events}")
    lines.append(f"Total downtime : {total_duration:,.0f} sec ({fmt_hms(total_duration)})")
    lines.append(f"Unique {dimension} values: {len(entries)}")
    lines.append("")

    table_rows = []
    for e in entries:
        table_rows.append([
            str(e["rank"]),
            e["value"],
            str(e["count"]),
            f"{e['duration']:,.0f}",
            fmt_hms(e["duration"]),
            f"{e['pct']:.1f}%",
            f"{e['cum_pct']:.1f}%",
        ])
    headers = ["Rank", dimension, "Count", "Duration(sec)", "Duration(hh:mm:ss)", "% Total", "Cum %"]
    aligns = ["<", "<", ">", ">", ">", ">", ">"]
    lines.extend(fmt_table(headers, table_rows, aligns))
    lines.append("")

    if entries:
        top = entries[0]
        lines.append(
            f"Top contributor: '{top['value']}' -> {top['duration']:,.0f} sec "
            f"({top['pct']:.1f}% of total downtime, {top['count']} events)"
        )
        # 80/20 callout: how many values make up 80% of downtime
        n80 = 0
        for e in entries:
            n80 += 1
            if e["cum_pct"] >= 80.0:
                break
        lines.append(
            f"Pareto (80/20): top {n80} of {len(entries)} {dimension} value(s) "
            f"account for ~80% of total downtime."
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCA - multi-dimension (pairwise) association, top-N
# ---------------------------------------------------------------------------

def combo_top_n(rows, dim_a, dim_b, total_duration, top_n):
    agg = defaultdict(lambda: [0, 0.0])
    for r in rows:
        key = (r[dim_a], r[dim_b])
        agg[key][0] += 1
        agg[key][1] += r["Duration (sec)"]

    items = sorted(agg.items(), key=lambda kv: kv[1][1], reverse=True)[:top_n]
    out = []
    for rank, ((a, b), (count, duration)) in enumerate(items, start=1):
        pct = (duration / total_duration * 100) if total_duration else 0.0
        out.append({"rank": rank, "a": a, "b": b, "count": count, "duration": duration, "pct": pct})
    return out


def render_mca_report(dim_a, dim_b, entries, top_n):
    lines = []
    lines.append(f"MULTIPLE CAUSE ANALYSIS (MCA) - Top {top_n} {dim_a} + {dim_b} combinations")
    lines.append("=" * 70)
    lines.append("")

    table_rows = []
    for e in entries:
        table_rows.append([
            str(e["rank"]),
            e["a"],
            e["b"],
            str(e["count"]),
            f"{e['duration']:,.0f}",
            fmt_hms(e["duration"]),
            f"{e['pct']:.1f}%",
        ])
    headers = ["Rank", dim_a, dim_b, "Count", "Duration(sec)", "Duration(hh:mm:ss)", "% of Total"]
    aligns = ["<", "<", "<", ">", ">", ">", ">"]
    lines.extend(fmt_table(headers, table_rows, aligns))
    lines.append("")

    if entries:
        top = entries[0]
        lines.append(
            f"Most associated pair: {dim_a}='{top['a']}' + {dim_b}='{top['b']}' -> "
            f"{top['duration']:,.0f} sec ({top['pct']:.1f}% of total downtime, {top['count']} events)"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-N per group (e.g. top 5 ReasonCodes for EACH Machine ID)
# ---------------------------------------------------------------------------

def top_n_per_group(rows, group_dim, rank_dim, top_n):
    buckets = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for r in rows:
        g = r[group_dim]
        k = r[rank_dim]
        buckets[g][k][0] += 1
        buckets[g][k][1] += r["Duration (sec)"]

    result = {}
    for g, agg in buckets.items():
        group_total = sum(v[1] for v in agg.values())
        items = sorted(agg.items(), key=lambda kv: kv[1][1], reverse=True)[:top_n]
        result[g] = {
            "total": group_total,
            "entries": [
                {
                    "value": k,
                    "count": v[0],
                    "duration": v[1],
                    "pct": (v[1] / group_total * 100) if group_total else 0.0,
                }
                for k, v in items
            ],
        }
    return result


def render_top_n_per_group(group_dim, rank_dim, data, top_n):
    lines = []
    lines.append(f"TOP {top_n} {rank_dim} PER {group_dim}")
    lines.append("=" * 70)
    lines.append("")
    for g in sorted(data.keys(), key=lambda k: data[k]["total"], reverse=True):
        info = data[g]
        lines.append(f"{group_dim}: {g}  (total downtime {info['total']:,.0f} sec / {fmt_hms(info['total'])})")
        table_rows = []
        for e in info["entries"]:
            table_rows.append([
                e["value"],
                str(e["count"]),
                f"{e['duration']:,.0f}",
                fmt_hms(e["duration"]),
                f"{e['pct']:.1f}%",
            ])
        headers = [rank_dim, "Count", "Duration(sec)", "Duration(hh:mm:ss)", "% of group total"]
        aligns = ["<", ">", ">", ">", ">"]
        table_lines = fmt_table(headers, table_rows, aligns)
        lines.extend("    " + l for l in table_lines)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pivot CSV export (for Excel / further analysis)
# ---------------------------------------------------------------------------

def write_pivot_csv(path, rows, dim_a, dim_b):
    row_values = sorted({r[dim_a] for r in rows})
    col_values = sorted({r[dim_b] for r in rows})
    agg = defaultdict(float)
    for r in rows:
        agg[(r[dim_a], r[dim_b])] += r["Duration (sec)"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([dim_a + " \\ " + dim_b] + col_values + ["Row Total"])
        for rv in row_values:
            row_total = 0.0
            row_cells = []
            for cv in col_values:
                val = agg.get((rv, cv), 0.0)
                row_total += val
                row_cells.append(f"{val:.0f}")
            writer.writerow([rv] + row_cells + [f"{row_total:.0f}"])
        col_totals = [f"{sum(agg.get((rv, cv), 0.0) for rv in row_values):.0f}" for cv in col_values]
        grand_total = sum(agg.values())
        writer.writerow(["Column Total"] + col_totals + [f"{grand_total:.0f}"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CA/MCA downtime analyzer")
    parser.add_argument("csv_path", help="Input CSV file")
    parser.add_argument("--outdir", default="reports", help="Output directory (default: reports)")
    parser.add_argument("--top", type=int, default=5, help="Top-N for MCA/association reports (default: 5)")
    args = parser.parse_args()

    rows, bad_duration = load_rows(args.csv_path)
    if not rows:
        raise SystemExit("ERROR: no valid data rows found in the CSV.")

    os.makedirs(args.outdir, exist_ok=True)

    total_duration = sum(r["Duration (sec)"] for r in rows)
    total_events = len(rows)
    top_n = args.top

    # ---------------- Summary ----------------
    summary_lines = []
    summary_lines.append("DOWNTIME ANALYSIS SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Input file           : {os.path.abspath(args.csv_path)}")
    summary_lines.append(f"Total events (rows)  : {total_events}")
    if bad_duration:
        summary_lines.append(f"Rows skipped (bad/missing duration): {bad_duration}")
    summary_lines.append(f"Total downtime       : {total_duration:,.0f} sec "
                          f"({total_duration/60:,.1f} min / {total_duration/3600:,.2f} hr) "
                          f"= {fmt_hms(total_duration)}")
    summary_lines.append(f"Average per event    : {total_duration/total_events:,.1f} sec")
    for dim in DIMENSIONS:
        summary_lines.append(f"Unique {dim:<14}: {len({r[dim] for r in rows})}")
    summary_lines.append("")
    summary_lines.append("Contents of this report set:")
    summary_lines.append("  CA  (Cause Analysis)          -> full Pareto detail per dimension")
    summary_lines.append("  MCA (Multiple Cause Analysis) -> top {} most associated pair combinations".format(top_n))
    summary_lines.append("  Top-N per group reports       -> e.g. top {} ReasonCodes for each Machine".format(top_n))
    summary_lines.append("  Pivot CSVs                    -> matrices for Excel / further analysis")
    summary_text = "\n".join(summary_lines) + "\n"
    with open(os.path.join(args.outdir, "00_Summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    # ---------------- CA reports (full detail, one per dimension) ----------------
    ca_sections = []
    for i, dim in enumerate(DIMENSIONS, start=1):
        entries = pareto(rows, dim, total_duration)
        text = render_ca_report(dim, entries, total_duration, total_events)
        ca_sections.append(text)
        fname = f"{i:02d}_CA_{dim.replace(' ', '')}.txt"
        with open(os.path.join(args.outdir, fname), "w", encoding="utf-8") as f:
            f.write(text)

    # ---------------- MCA reports (top-N pair associations) ----------------
    mca_sections = []
    for j, (a, b) in enumerate(COMBO_REPORTS, start=1):
        entries = combo_top_n(rows, a, b, total_duration, top_n)
        text = render_mca_report(a, b, entries, top_n)
        mca_sections.append(text)
        fname = f"{len(DIMENSIONS)+j:02d}_MCA_{a.replace(' ', '')}_{b.replace(' ', '')}_Top{top_n}.txt"
        with open(os.path.join(args.outdir, fname), "w", encoding="utf-8") as f:
            f.write(text)

    # ---------------- Top-N per group reports ----------------
    per_group_sections = []
    per_group_specs = [
        ("Machine ID", "ReasonCode"),
        ("Machine Group", "ReasonCode"),
        ("Machine Group", "Machine ID"),
        ("Product Code", "ReasonCode"),
    ]
    base_idx = len(DIMENSIONS) + len(COMBO_REPORTS)
    for k, (g, rdim) in enumerate(per_group_specs, start=1):
        data = top_n_per_group(rows, g, rdim, top_n)
        text = render_top_n_per_group(g, rdim, data, top_n)
        per_group_sections.append(text)
        fname = f"{base_idx+k:02d}_Top{top_n}_{rdim.replace(' ', '')}_per_{g.replace(' ', '')}.txt"
        with open(os.path.join(args.outdir, fname), "w", encoding="utf-8") as f:
            f.write(text)

    # ---------------- Combined full report ----------------
    combined_path = os.path.join(args.outdir, "CA_MCA_Full_Report.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
        f.write("\n\n")
        f.write("\n\n".join(ca_sections))
        f.write("\n\n")
        f.write("\n\n".join(mca_sections))
        f.write("\n\n")
        f.write("\n\n".join(per_group_sections))

    # ---------------- Pivot CSVs ----------------
    write_pivot_csv(os.path.join(args.outdir, "pivot_MachineID_ReasonCode.csv"), rows, "Machine ID", "ReasonCode")
    write_pivot_csv(os.path.join(args.outdir, "pivot_MachineGroup_ReasonCode.csv"), rows, "Machine Group", "ReasonCode")
    write_pivot_csv(os.path.join(args.outdir, "pivot_ProductCode_ReasonCode.csv"), rows, "Product Code", "ReasonCode")

    print(f"Done. Wrote reports to: {os.path.abspath(args.outdir)}")
    print(f"  - Combined report: {combined_path}")
    print(f"  - {len(DIMENSIONS)} CA detail reports, {len(COMBO_REPORTS)} MCA top-{top_n} reports, "
          f"{len(per_group_specs)} per-group top-{top_n} reports, 3 pivot CSVs")


if __name__ == "__main__":
    main()
