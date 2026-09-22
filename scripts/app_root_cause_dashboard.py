"""
DASH APP - EXECUTIVE ROOT-CAUSE DASHBOARD
=========================================
Validated layout (unchanged from spec), redesigned visual layer:
1. Header
2. One row with 4 KPI cards:
   - UPH
   - Availability
   - Performance
   - OEE (derived)
   Each card contains Jul, Aug, % gap, trend bar, and status
3. One row with 2 charts:
   - left  : share of drop dominance
   - right : counterfactual effect
4. One row with 2 charts:
   - left  : z-score chart
   - right : box plots
5. Supporting tables

Run
---
pip install dash plotly pandas openpyxl numpy
python app_root_cause_dashboard.py
"""

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, dash_table
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================

MACHINE_XLSX = r"Resources\Machine_UPH_JUL_AUG.xlsx"
SHEET0, SHEET1 = "Jul26", "Aug26"

HISTORICAL_XLSX = r"Resources\export.xlsx"
HISTORICAL_SHEET = 0

MACHINE_ID_ACTUAL_COLUMN = "MACHINE_ID"
MACHINE_ID_HIST_COLUMN = "MACHINE_LABEL"
UPH_COLUMN = "UPH"
AVAILABILITY_SOURCE_COLUMN = "OEE"
MTBI_COLUMN = "MTBI"
F2_COLUMN = "F2"
MTTR_DENOM_COLUMN = "TEUD"

PERIOD0, PERIOD1 = "Jul", "Aug"
UPH_THEORETICAL = 9455
Z_SIGNIFICANT = 2.0
DOMINANCE_GAP_CLOSED_MIN = 50.0


# ============================================================
# HELPERS
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


def z_from_historical_baseline(observed_diff, hist_std, n0, n1):
    if pd.isna(hist_std) or hist_std == 0 or n0 <= 0 or n1 <= 0:
        return None
    se = hist_std * np.sqrt((1.0 / n0) + (1.0 / n1))
    if se == 0 or np.isnan(se):
        return None
    return observed_diff / se


def safe_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def period_stats(df, label):
    n_machines = df["MACHINE_ID"].nunique() if "MACHINE_ID" in df.columns else np.nan
    n_records = len(df)
    stats = {"Period": label, "N_Records": n_records, "N_Machines": n_machines}

    for col in ["UPH", "Availability_%", "Performance_%", "Derived_OEE_%", "MTBI", "MTTR", "AV_Model_%"]:
        if col in df.columns:
            stats[f"{col}_mean"] = df[col].mean()
            stats[f"{col}_median"] = df[col].median()
            stats[f"{col}_std"] = df[col].std(ddof=1)
    return stats


def pct_gap(old, new):
    if pd.isna(old) or old == 0 or pd.isna(new):
        return np.nan
    return (new - old) / old * 100.0


def fmt_num(v, digits=2):
    return "N/A" if pd.isna(v) else f"{v:,.{digits}f}"


def fmt_pct(v, digits=2):
    return "N/A" if pd.isna(v) else f"{v:+.{digits}f}%"


# ============================================================
# LOADERS
# ============================================================

def load_and_enrich_actual(xlsx_path, sheet_name, uph_theoretical, period_label):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    required_cols = (
        MACHINE_ID_ACTUAL_COLUMN,
        UPH_COLUMN,
        AVAILABILITY_SOURCE_COLUMN,
        MTBI_COLUMN,
        F2_COLUMN,
        MTTR_DENOM_COLUMN,
    )
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in sheet '{sheet_name}'")

    df = df.copy()
    df = safe_numeric(df, [UPH_COLUMN, AVAILABILITY_SOURCE_COLUMN, MTBI_COLUMN, F2_COLUMN, MTTR_DENOM_COLUMN])

    df["MACHINE_ID"] = df[MACHINE_ID_ACTUAL_COLUMN]
    df["UPH"] = df[UPH_COLUMN]

    # Source OEE column is actually Availability
    df["Availability_%"] = df[AVAILABILITY_SOURCE_COLUMN]
    df["Performance_%"] = df["UPH"] / uph_theoretical * 100.0
    df["Derived_OEE_%"] = (df["Availability_%"] * df["Performance_%"]) / 100.0

    df["MTBI"] = df[MTBI_COLUMN]
    df["MTTR"] = np.where(df[MTTR_DENOM_COLUMN] > 0, df[F2_COLUMN] / df[MTTR_DENOM_COLUMN], np.nan)
    df["AV_Model_%"] = np.where(df["MTBI"] > 0, 100.0 / (1.0 + (df["MTTR"] / df["MTBI"])), np.nan)

    df["Period"] = period_label
    return df


def load_historical_baseline(xlsx_path, sheet_name, uph_theoretical):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    required_cols = (
        MACHINE_ID_HIST_COLUMN,
        UPH_COLUMN,
        AVAILABILITY_SOURCE_COLUMN,
        MTBI_COLUMN,
        F2_COLUMN,
        MTTR_DENOM_COLUMN,
    )
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in historical sheet '{sheet_name}'")

    df = df.copy()
    df = safe_numeric(df, [UPH_COLUMN, AVAILABILITY_SOURCE_COLUMN, MTBI_COLUMN, F2_COLUMN, MTTR_DENOM_COLUMN])

    df["MACHINE_ID"] = df[MACHINE_ID_HIST_COLUMN]
    df["UPH"] = df[UPH_COLUMN]
    df["Availability_%"] = df[AVAILABILITY_SOURCE_COLUMN]
    df["Performance_%"] = df["UPH"] / uph_theoretical * 100.0
    df["Derived_OEE_%"] = (df["Availability_%"] * df["Performance_%"]) / 100.0
    df["MTBI"] = df[MTBI_COLUMN]
    df["MTTR"] = np.where(df[MTTR_DENOM_COLUMN] > 0, df[F2_COLUMN] / df[MTTR_DENOM_COLUMN], np.nan)
    df["AV_Model_%"] = np.where(df["MTBI"] > 0, 100.0 / (1.0 + (df["MTTR"] / df["MTBI"])), np.nan)
    return df


# ============================================================
# ANALYSIS
# ============================================================

def step1_kpi_pooled_actual(xlsx_path, sheet0, sheet1, uph_theoretical):
    df0 = load_and_enrich_actual(xlsx_path, sheet0, uph_theoretical, PERIOD0)
    df1 = load_and_enrich_actual(xlsx_path, sheet1, uph_theoretical, PERIOD1)
    pooled = pd.concat([df0, df1], ignore_index=True)
    s0 = period_stats(df0, PERIOD0)
    s1 = period_stats(df1, PERIOD1)
    return pooled, df0, df1, s0, s1


def step2_log_decomposition(s0, s1):
    A0, A1 = s0["Availability_%_mean"] / 100.0, s1["Availability_%_mean"] / 100.0
    P0, P1 = s0["Performance_%_mean"] / 100.0, s1["Performance_%_mean"] / 100.0

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


def step3_significance_phase1(s0, s1, hist_df):
    results = {}
    n0 = s0["N_Records"]
    n1 = s1["N_Records"]

    for metric in ["Performance_%", "Availability_%"]:
        observed_diff = s1[f"{metric}_mean"] - s0[f"{metric}_mean"]
        hist_std = hist_df[metric].std(ddof=1)
        z = z_from_historical_baseline(observed_diff, hist_std, n0, n1)
        results[metric] = {
            "z": z,
            "label": significance_label(z),
            "observed_diff": observed_diff,
            "historical_std": hist_std,
        }
    return results


def step4_counterfactual_pooled(s0, s1):
    A0, A1 = s0["Availability_%_mean"] / 100.0, s1["Availability_%_mean"] / 100.0
    P0, P1 = s0["Performance_%_mean"] / 100.0, s1["Performance_%_mean"] / 100.0

    OEE0, OEE1 = A0 * P0, A1 * P1
    gap = OEE0 - OEE1
    OEE_cf_A = A0 * P1
    OEE_cf_P = A1 * P0

    if gap != 0:
        gap_closed_A = (OEE_cf_A - OEE1) / gap * 100.0
        gap_closed_P = (OEE_cf_P - OEE1) / gap * 100.0
    else:
        gap_closed_A = gap_closed_P = 0.0

    if gap_closed_A >= gap_closed_P:
        root, contribution = "Availability", gap_closed_A
    else:
        root, contribution = "Performance", gap_closed_P

    return {
        "Gap_Closed_Availability_%": round(gap_closed_A, 2),
        "Gap_Closed_Performance_%": round(gap_closed_P, 2),
        "Root_Cause": root,
        "Root_Cause_Contribution_%": round(contribution, 2),
    }


def significance_for_metrics(s0, s1, hist_df, metrics):
    """Historical-baseline z-test, generalized to any list of metrics
    that exist in both the period stats and the historical baseline."""
    n0, n1 = s0["N_Records"], s1["N_Records"]
    results = {}
    for metric in metrics:
        observed_diff = s1[f"{metric}_mean"] - s0[f"{metric}_mean"]
        hist_std = hist_df[metric].std(ddof=1)
        z = z_from_historical_baseline(observed_diff, hist_std, n0, n1)
        results[metric] = {
            "z": z,
            "label": significance_label(z),
            "observed_diff": observed_diff,
            "historical_std": hist_std,
        }
    return results


def availability_driver_decomposition(s0, s1):
    """MTTR vs MTBI counterfactual decomposition of the *model-based*
    availability (AV_Model_% = 100 / (1 + MTTR/MTBI)) — a separate lens
    from the Availability-vs-Performance OEE decomposition above."""
    mttr0, mttr1 = s0["MTTR_mean"], s1["MTTR_mean"]
    mtbi0, mtbi1 = s0["MTBI_mean"], s1["MTBI_mean"]

    av0 = 100.0 / (1.0 + (mttr0 / mtbi0)) if pd.notna(mttr0) and pd.notna(mtbi0) and mtbi0 != 0 else np.nan
    av1 = 100.0 / (1.0 + (mttr1 / mtbi1)) if pd.notna(mttr1) and pd.notna(mtbi1) and mtbi1 != 0 else np.nan

    av_cf_mttr = 100.0 / (1.0 + (mttr0 / mtbi1)) if pd.notna(mttr0) and pd.notna(mtbi1) and mtbi1 != 0 else np.nan
    av_cf_mtbi = 100.0 / (1.0 + (mttr1 / mtbi0)) if pd.notna(mttr1) and pd.notna(mtbi0) and mtbi0 != 0 else np.nan

    gap = av0 - av1
    if gap != 0 and not pd.isna(gap):
        gap_closed_mttr = (av_cf_mttr - av1) / gap * 100.0 if pd.notna(av_cf_mttr) else np.nan
        gap_closed_mtbi = (av_cf_mtbi - av1) / gap * 100.0 if pd.notna(av_cf_mtbi) else np.nan
    else:
        gap_closed_mttr = 0.0
        gap_closed_mtbi = 0.0

    contrib = {"MTTR": gap_closed_mttr, "MTBI": gap_closed_mtbi}
    contrib = {k: v for k, v in contrib.items() if pd.notna(v)}

    if contrib:
        root = max(contrib, key=contrib.get)
        contribution = contrib[root]
    else:
        root = None
        contribution = np.nan

    return {
        "AV_Model_Jul_%": av0,
        "AV_Model_Aug_%": av1,
        "Gap_Closed_MTTR_%": gap_closed_mttr,
        "Gap_Closed_MTBI_%": gap_closed_mtbi,
        "Root_Cause": root,
        "Root_Cause_Contribution_%": contribution,
    }


# ============================================================
# LOAD DATA
# ============================================================

pooled_actual, df_jul, df_aug, s0, s1 = step1_kpi_pooled_actual(
    MACHINE_XLSX, SHEET0, SHEET1, UPH_THEORETICAL
)
hist_df = load_historical_baseline(HISTORICAL_XLSX, HISTORICAL_SHEET, UPH_THEORETICAL)
decomp = step2_log_decomposition(s0, s1)
sig_results = step3_significance_phase1(s0, s1, hist_df)
cf = step4_counterfactual_pooled(s0, s1)

# Source-vs-model availability gap, added post-load (loaders stay untouched)
for _df in (df_jul, df_aug, pooled_actual):
    _df["Gap_Source_vs_Model_%"] = _df["Availability_%"] - _df["AV_Model_%"]
hist_df["Gap_Source_vs_Model_%"] = hist_df["Availability_%"] - hist_df["AV_Model_%"]
s0["Gap_Source_vs_Model_%_mean"] = df_jul["Gap_Source_vs_Model_%"].mean()
s1["Gap_Source_vs_Model_%_mean"] = df_aug["Gap_Source_vs_Model_%"].mean()

avail_sig_results = significance_for_metrics(
    s0, s1, hist_df, ["Availability_%", "MTTR", "MTBI", "AV_Model_%"]
)
av_driver = availability_driver_decomposition(s0, s1)

uph_gap = pct_gap(s0["UPH_mean"], s1["UPH_mean"])
availability_gap = pct_gap(s0["Availability_%_mean"], s1["Availability_%_mean"])
performance_gap = pct_gap(s0["Performance_%_mean"], s1["Performance_%_mean"])
oee_gap = pct_gap(s0["Derived_OEE_%_mean"], s1["Derived_OEE_%_mean"])
mttr_gap = pct_gap(s0["MTTR_mean"], s1["MTTR_mean"])
mtbi_gap = pct_gap(s0["MTBI_mean"], s1["MTBI_mean"])
model_av_gap = pct_gap(s0["AV_Model_%_mean"], s1["AV_Model_%_mean"])

summary_df = pd.DataFrame([
    {
        "Period": s["Period"],
        "N_Records": s["N_Records"],
        "N_Machines": s["N_Machines"],
        "UPH_mean": round(s.get("UPH_mean", np.nan), 4),
        "Availability_%_mean": round(s.get("Availability_%_mean", np.nan), 4),
        "Performance_%_mean": round(s.get("Performance_%_mean", np.nan), 4),
        "Derived_OEE_%_mean": round(s.get("Derived_OEE_%_mean", np.nan), 4),
    }
    for s in [s0, s1]
])

significance_df = pd.DataFrame([
    {
        "Metric": m,
        "Observed_Diff": sig_results[m]["observed_diff"],
        "Historical_STD": sig_results[m]["historical_std"],
        "Z_Score": sig_results[m]["z"],
        "Label": sig_results[m]["label"],
    }
    for m in ["Performance_%", "Availability_%"]
])

drop_share_df = pd.DataFrame({
    "Factor": ["Availability", "Performance"],
    "Share_%": [decomp["Availability_Share_%"], decomp["Performance_Share_%"]],
})

counterfactual_df = pd.DataFrame({
    "Scenario": ["Restore Availability", "Restore Performance"],
    "Gap_Closed_%": [cf["Gap_Closed_Availability_%"], cf["Gap_Closed_Performance_%"]],
})

# ---- Availability deep-dive tables (MTTR / MTBI model view) ----

availability_summary_df = pd.DataFrame([
    {
        "Period": s["Period"],
        "N_Records": s["N_Records"],
        "N_Machines": s["N_Machines"],
        "Availability_%_mean": round(s["Availability_%_mean"], 4),
        "MTTR_mean": round(s["MTTR_mean"], 4),
        "MTBI_mean": round(s["MTBI_mean"], 4),
        "AV_Model_%_mean": round(s["AV_Model_%_mean"], 4),
        "Gap_Source_vs_Model_%_mean": round(s["Gap_Source_vs_Model_%_mean"], 4),
    }
    for s in [s0, s1]
])

avail_significance_df = pd.DataFrame([
    {
        "Metric": m,
        "Observed_Diff": avail_sig_results[m]["observed_diff"],
        "Historical_STD": avail_sig_results[m]["historical_std"],
        "Z_Score": avail_sig_results[m]["z"],
        "Label": avail_sig_results[m]["label"],
    }
    for m in ["Availability_%", "MTTR", "MTBI", "AV_Model_%"]
])

driver_df = pd.DataFrame([{
    "AV_Model_Jul_%": round(av_driver["AV_Model_Jul_%"], 4) if pd.notna(av_driver["AV_Model_Jul_%"]) else np.nan,
    "AV_Model_Aug_%": round(av_driver["AV_Model_Aug_%"], 4) if pd.notna(av_driver["AV_Model_Aug_%"]) else np.nan,
    "Gap_Closed_MTTR_%": round(av_driver["Gap_Closed_MTTR_%"], 2) if pd.notna(av_driver["Gap_Closed_MTTR_%"]) else np.nan,
    "Gap_Closed_MTBI_%": round(av_driver["Gap_Closed_MTBI_%"], 2) if pd.notna(av_driver["Gap_Closed_MTBI_%"]) else np.nan,
    "Root_Cause": av_driver["Root_Cause"],
    "Root_Cause_Contribution_%": round(av_driver["Root_Cause_Contribution_%"], 2) if pd.notna(av_driver["Root_Cause_Contribution_%"]) else np.nan,
}])

source_vs_model_df = pd.DataFrame({
    "Metric": ["Source Availability", "Model Availability"],
    PERIOD0: [s0["Availability_%_mean"], s0["AV_Model_%_mean"]],
    PERIOD1: [s1["Availability_%_mean"], s1["AV_Model_%_mean"]],
})
source_vs_model_melt = source_vs_model_df.melt(id_vars="Metric", var_name="Period", value_name="Value")

driver_contrib_df = pd.DataFrame({
    "Driver": ["MTTR", "MTBI"],
    "Gap_Closed_%": [av_driver["Gap_Closed_MTTR_%"], av_driver["Gap_Closed_MTBI_%"]],
})

avail_z_df = avail_significance_df[avail_significance_df["Metric"].isin(["Availability_%", "MTTR", "MTBI"])]




# ============================================================
# DESIGN TOKENS
# ============================================================
# One small, deliberate system: a toned paper ground, near-black ink for
# text, a single warm accent for emphasis/branding, and a fixed semantic
# pair (teal = good, amber = watch, red = risk) used consistently across
# every card, chip, and chart so status reads the same way everywhere.

INK = "#161A1F"
INK_SOFT = "#4B5560"
INK_FAINT = "#88919B"
PAPER = "#F4F2ED"
SURFACE = "#FFFFFF"
LINE = "#E4E1D8"
ACCENT = "#C9622C"        # warm clay — brand/emphasis only, not status
GOOD = "#1E8A6E"
GOOD_BG = "#E4F3EE"
WATCH = "#B5750A"
WATCH_BG = "#FBF0DD"
RISK = "#BF3B3B"
RISK_BG = "#FBE9E9"
NEUTRAL_BG = "#EEECE4"

FONT_DISPLAY = "'Space Grotesk', 'Segoe UI', sans-serif"
FONT_BODY = "'IBM Plex Sans', 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', 'Consolas', monospace"

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT_BODY, color=INK_SOFT, size=13),
        title=dict(font=dict(family=FONT_DISPLAY, color=INK, size=17)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=48, r=48, t=56, b=44, autoexpand=True),
        xaxis=dict(gridcolor="#EDEAE1", griddash="dot", gridwidth=1, zerolinecolor=LINE,
                    linecolor=LINE, ticks="outside", tickcolor=LINE, tickfont=dict(color=INK_FAINT, size=12),
                    automargin=True),
        yaxis=dict(gridcolor="#EDEAE1", griddash="dot", gridwidth=1, zerolinecolor=LINE,
                    linecolor=LINE, ticks="outside", tickcolor=LINE, tickfont=dict(color=INK_SOFT, size=13),
                    automargin=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(family=FONT_BODY, size=12)),
        colorway=[ACCENT, "#3E6FA6", GOOD, WATCH, RISK],
    )
)


def status_from_gap(v, good_is_positive=True):
    """Map a % gap to a (label, fg, bg) status chip."""
    if pd.isna(v):
        return "NO DATA", INK_FAINT, NEUTRAL_BG
    positive = v >= 0
    is_good = positive if good_is_positive else not positive
    if abs(v) < 1:
        return "STABLE", INK_SOFT, NEUTRAL_BG
    if is_good:
        return "IMPROVING", GOOD, GOOD_BG
    return "DECLINING", RISK, RISK_BG


def significance_chip(label):
    m = {
        "VERY_SIGNIFICANT": (RISK, RISK_BG, "Very significant"),
        "SIGNIFICANT": (WATCH, WATCH_BG, "Significant"),
        "MODERATE": (WATCH, WATCH_BG, "Moderate"),
        "LIKELY_RANDOM": (INK_SOFT, NEUTRAL_BG, "Likely random"),
        "N/A": (INK_FAINT, NEUTRAL_BG, "N/A"),
    }
    return m.get(label, m["N/A"])


# ============================================================
# STYLES
# ============================================================

PAGE_STYLE = {
    "fontFamily": FONT_BODY,
    "backgroundColor": PAPER,
    "padding": "32px 40px 56px",
    "minHeight": "100vh",
    "color": INK,
}

SECTION_STYLE = {
    "backgroundColor": SURFACE,
    "borderRadius": "16px",
    "padding": "24px 26px",
    "marginBottom": "20px",
    "border": f"1px solid {LINE}",
}

SECTION_TITLE_STYLE = {
    "fontFamily": FONT_DISPLAY,
    "fontSize": "13px",
    "fontWeight": "600",
    "letterSpacing": "0.08em",
    "textTransform": "uppercase",
    "color": INK_SOFT,
    "margin": "0 0 18px 0",
}

CARD_STYLE = {
    "backgroundColor": SURFACE,
    "borderRadius": "16px",
    "padding": "22px 22px 20px",
    "border": f"1px solid {LINE}",
    "flex": "1",
    "minWidth": "250px",
}

LABEL_STYLE = {
    "fontFamily": FONT_DISPLAY,
    "fontSize": "12px",
    "fontWeight": "600",
    "letterSpacing": "0.06em",
    "textTransform": "uppercase",
    "color": INK_FAINT,
    "marginBottom": "14px",
}

PERIOD_LABEL_STYLE = {
    "fontSize": "11px",
    "fontWeight": "600",
    "letterSpacing": "0.05em",
    "textTransform": "uppercase",
    "color": INK_FAINT,
    "marginBottom": "4px",
}

VALUE_STYLE = {
    "fontFamily": FONT_MONO,
    "fontSize": "26px",
    "fontWeight": "600",
    "color": INK,
    "margin": "0",
    "lineHeight": "1.1",
}

PREV_VALUE_STYLE = {
    "fontFamily": FONT_MONO,
    "fontSize": "17px",
    "fontWeight": "500",
    "color": INK_FAINT,
    "margin": "0",
    "lineHeight": "1.1",
}

SUBVALUE_STYLE = {"fontSize": "12.5px", "color": INK_FAINT, "fontFamily": FONT_MONO}


def chip(text, fg, bg, size="12px"):
    return html.Span(text, style={
        "display": "inline-block",
        "padding": "5px 11px",
        "borderRadius": "999px",
        "fontSize": size,
        "fontWeight": "700",
        "letterSpacing": "0.02em",
        "color": fg,
        "backgroundColor": bg,
        "whiteSpace": "nowrap",
    })


def meta_pill(text):
    return html.Div(text, style={
        "fontSize": "12.5px",
        "fontFamily": FONT_MONO,
        "color": INK_SOFT,
        "backgroundColor": PAPER,
        "border": f"1px solid {LINE}",
        "borderRadius": "999px",
        "padding": "6px 14px",
    })


def trend_bar(jul_v, aug_v):
    """A tiny two-bar comparison so the Jul->Aug move is visible at a glance."""
    vals = [v for v in (jul_v, aug_v) if not pd.isna(v)]
    hi = max(vals) if vals else 1
    hi = hi if hi > 0 else 1

    def bar(v, color):
        w = max(4, (v / hi) * 100) if not pd.isna(v) else 0
        return html.Div(style={
            "height": "6px", "borderRadius": "3px", "backgroundColor": NEUTRAL_BG,
            "overflow": "hidden", "flex": "1",
        }, children=html.Div(style={
            "height": "100%", "width": f"{w}%", "backgroundColor": color, "borderRadius": "3px",
        }))

    return html.Div(style={"display": "flex", "gap": "6px", "alignItems": "center", "marginTop": "14px"},
                     children=[bar(jul_v, LINE), bar(aug_v, ACCENT)])


# ============================================================
# COMPONENTS
# ============================================================

def kpi_card(title, jul_value, aug_value, gap_value, suffix, status_label, status_fg, status_bg,
             good_is_positive=True):
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"},
                      children=[
                          html.Div(title, style=LABEL_STYLE),
                          chip(status_label, status_fg, status_bg),
                      ]),
            html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "14px", "marginTop": "2px"},
                      children=[
                          html.Div([
                              html.Div(PERIOD1, style=PERIOD_LABEL_STYLE),
                              html.H2(f"{fmt_num(aug_value)}{suffix}", style=VALUE_STYLE),
                          ]),
                          html.Div([
                              html.Div(PERIOD0, style=PERIOD_LABEL_STYLE),
                              html.H3(f"{fmt_num(jul_value)}{suffix}", style=PREV_VALUE_STYLE),
                          ]),
                      ]),
            trend_bar(jul_value, aug_value),
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                              "marginTop": "12px"}, children=[
                html.Span(f"{'▲' if not pd.isna(gap_value) and gap_value >= 0 else '▼'} {fmt_pct(gap_value)}",
                          style={
                              "fontFamily": FONT_MONO, "fontSize": "13px", "fontWeight": "700",
                              "color": GOOD if (not pd.isna(gap_value) and (gap_value >= 0) == good_is_positive)
                              else RISK,
                          }),
                html.Span("vs. Jul", style={"fontSize": "12px", "color": INK_FAINT}),
            ]),
        ]
    )


uph_status_label, uph_status_fg, uph_status_bg = status_from_gap(uph_gap, good_is_positive=True)
avail_status_label, avail_status_fg, avail_status_bg = status_from_gap(availability_gap, good_is_positive=True)
perf_status_label, perf_status_fg, perf_status_bg = status_from_gap(performance_gap, good_is_positive=True)
oee_status_label, oee_status_fg, oee_status_bg = status_from_gap(oee_gap, good_is_positive=True)

kpi_row = html.Div(
    style={"display": "flex", "flexWrap": "wrap", "gap": "16px"},
    children=[
        kpi_card("Units per Hour", s0["UPH_mean"], s1["UPH_mean"], uph_gap, "",
                 uph_status_label, uph_status_fg, uph_status_bg),
        kpi_card("Availability", s0["Availability_%_mean"], s1["Availability_%_mean"], availability_gap, "%",
                 avail_status_label, avail_status_fg, avail_status_bg),
        kpi_card("Performance", s0["Performance_%_mean"], s1["Performance_%_mean"], performance_gap, "%",
                 perf_status_label, perf_status_fg, perf_status_bg),
        kpi_card("OEE (derived)", s0["Derived_OEE_%_mean"], s1["Derived_OEE_%_mean"], oee_gap, "%",
                 oee_status_label, oee_status_fg, oee_status_bg),
    ]
)


# ============================================================
# CHARTS
# ============================================================

AVAIL_BLUE = "#3E6FA6"

max_share = float(drop_share_df["Share_%"].max())
fig_drop_share = px.bar(
    drop_share_df, x="Share_%", y="Factor", orientation="h",
    title="Share of drop dominance", text="Share_%",
    color="Factor", color_discrete_map={"Availability": AVAIL_BLUE, "Performance": RISK},
)
fig_drop_share.update_traces(
    texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False,
    marker_line_width=0, textfont=dict(family=FONT_MONO, size=13, color=INK),
)
fig_drop_share.update_layout(
    template=PLOTLY_TEMPLATE, showlegend=False, height=340,
    margin=dict(l=16, r=64, t=56, b=48),
    bargap=0.55,
    xaxis=dict(title="% share of total decline", ticksuffix="%",
                range=[0, max_share * 1.28], automargin=True),
    yaxis=dict(title=None, automargin=True, ticklabelposition="outside"),
)

max_gap = float(counterfactual_df["Gap_Closed_%"].max())
fig_counterfactual = px.bar(
    counterfactual_df, x="Gap_Closed_%", y="Scenario", orientation="h",
    title="Counterfactual effect — gap closed if restored", text="Gap_Closed_%",
    color="Scenario", color_discrete_sequence=[GOOD, WATCH],
)
fig_counterfactual.update_traces(
    texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False,
    marker_line_width=0, textfont=dict(family=FONT_MONO, size=13, color=INK),
)
fig_counterfactual.update_layout(
    template=PLOTLY_TEMPLATE, showlegend=False, height=340,
    margin=dict(l=16, r=64, t=56, b=48),
    bargap=0.55,
    xaxis=dict(title="% of OEE gap closed", ticksuffix="%",
                range=[0, max_gap * 1.28], automargin=True),
    yaxis=dict(title=None, automargin=True, ticklabelposition="outside"),
)

fig_significance = px.bar(
    significance_df, x="Metric", y="Z_Score", color="Label",
    title=f"Z-score significance vs. historical baseline (±{Z_SIGNIFICANT:.1f} threshold)",
    color_discrete_map={
        "VERY_SIGNIFICANT": RISK, "SIGNIFICANT": WATCH, "MODERATE": WATCH,
        "LIKELY_RANDOM": INK_FAINT, "N/A": INK_FAINT,
    },
)
fig_significance.add_hline(y=Z_SIGNIFICANT, line_dash="dot", line_color=RISK, line_width=1.5)
fig_significance.add_hline(y=-Z_SIGNIFICANT, line_dash="dot", line_color=RISK, line_width=1.5)
fig_significance.update_traces(marker_line_width=0)
fig_significance.update_layout(template=PLOTLY_TEMPLATE, height=380, xaxis_title=None,
                                legend_title_text="", legend=dict(orientation="h", y=-0.18))

fig_box = px.box(
    pooled_actual.melt(id_vars="Period", value_vars=["UPH", "Availability_%", "Performance_%"],
                        var_name="Metric", value_name="Value"),
    x="Metric", y="Value", color="Period", title="Record-level distribution",
    color_discrete_map={PERIOD0: INK_FAINT, PERIOD1: ACCENT},
)
fig_box.update_layout(template=PLOTLY_TEMPLATE, height=380, xaxis_title=None,
                       legend_title_text="", legend=dict(orientation="h", y=-0.18))


# ============================================================
# AVAILABILITY DEEP-DIVE — KPI mini-row (Source / MTTR / MTBI / Model)
# ============================================================

mttr_status_label, mttr_status_fg, mttr_status_bg = status_from_gap(mttr_gap, good_is_positive=False)
mtbi_status_label, mtbi_status_fg, mtbi_status_bg = status_from_gap(mtbi_gap, good_is_positive=True)
model_av_status_label, model_av_status_fg, model_av_status_bg = status_from_gap(model_av_gap, good_is_positive=True)

avail_kpi_row = html.Div(
    style={"display": "flex", "flexWrap": "wrap", "gap": "16px"},
    children=[
        kpi_card("Source Availability", s0["Availability_%_mean"], s1["Availability_%_mean"],
                 availability_gap, "%", avail_status_label, avail_status_fg, avail_status_bg),
        kpi_card("MTTR", s0["MTTR_mean"], s1["MTTR_mean"], mttr_gap, "",
                 mttr_status_label, mttr_status_fg, mttr_status_bg, good_is_positive=False),
        kpi_card("MTBI", s0["MTBI_mean"], s1["MTBI_mean"], mtbi_gap, "",
                 mtbi_status_label, mtbi_status_fg, mtbi_status_bg),
        kpi_card("Model Availability", s0["AV_Model_%_mean"], s1["AV_Model_%_mean"], model_av_gap, "%",
                 model_av_status_label, model_av_status_fg, model_av_status_bg),
    ]
)

# ---- Availability deep-dive charts ----

fig_source_model = px.bar(
    source_vs_model_melt, x="Metric", y="Value", color="Period", barmode="group",
    title="Source vs. modelled availability", text="Value",
    color_discrete_map={PERIOD0: INK_FAINT, PERIOD1: ACCENT},
)
fig_source_model.update_traces(
    texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False,
    marker_line_width=0, textfont=dict(family=FONT_MONO, size=12, color=INK),
)
max_src_model = float(source_vs_model_melt["Value"].max())
fig_source_model.update_layout(
    template=PLOTLY_TEMPLATE, height=340,
    margin=dict(l=16, r=24, t=56, b=48),
    bargap=0.35, bargroupgap=0.08,
    yaxis=dict(title=None, ticksuffix="%", range=[0, max_src_model * 1.22], automargin=True),
    xaxis=dict(title=None, automargin=True),
    legend_title_text="", legend=dict(orientation="h", y=-0.16),
)

max_driver = float(driver_contrib_df["Gap_Closed_%"].max())
fig_availability_driver = px.bar(
    driver_contrib_df, x="Gap_Closed_%", y="Driver", orientation="h",
    title="Availability driver contribution — MTTR vs. MTBI", text="Gap_Closed_%",
    color="Driver", color_discrete_map={"MTTR": WATCH, "MTBI": AVAIL_BLUE},
)
fig_availability_driver.update_traces(
    texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False,
    marker_line_width=0, textfont=dict(family=FONT_MONO, size=13, color=INK),
)
fig_availability_driver.update_layout(
    template=PLOTLY_TEMPLATE, showlegend=False, height=340,
    margin=dict(l=16, r=64, t=56, b=48),
    bargap=0.55,
    xaxis=dict(title="% of model-availability gap closed", ticksuffix="%",
                range=[0, max_driver * 1.28], automargin=True),
    yaxis=dict(title=None, automargin=True, ticklabelposition="outside"),
)

fig_avail_z = px.bar(
    avail_z_df, x="Metric", y="Z_Score", color="Label",
    title=f"Availability-related z-scores (±{Z_SIGNIFICANT:.1f} threshold)",
    color_discrete_map={
        "VERY_SIGNIFICANT": RISK, "SIGNIFICANT": WATCH, "MODERATE": WATCH,
        "LIKELY_RANDOM": INK_FAINT, "N/A": INK_FAINT,
    },
)
fig_avail_z.add_hline(y=Z_SIGNIFICANT, line_dash="dot", line_color=RISK, line_width=1.5)
fig_avail_z.add_hline(y=-Z_SIGNIFICANT, line_dash="dot", line_color=RISK, line_width=1.5)
fig_avail_z.update_traces(marker_line_width=0)
fig_avail_z.update_layout(template=PLOTLY_TEMPLATE, height=380, xaxis_title=None,
                           legend_title_text="", legend=dict(orientation="h", y=-0.18))

fig_avail_box = px.box(
    pooled_actual.melt(id_vars="Period", value_vars=["Availability_%", "MTTR", "MTBI", "AV_Model_%"],
                        var_name="Metric", value_name="Value"),
    x="Metric", y="Value", color="Period", title="Availability, MTTR, MTBI & model distributions",
    color_discrete_map={PERIOD0: INK_FAINT, PERIOD1: ACCENT},
)
fig_avail_box.update_layout(template=PLOTLY_TEMPLATE, height=380, xaxis_title=None,
                             legend_title_text="", legend=dict(orientation="h", y=-0.18))


# ============================================================
# TABLE STYLES
# ============================================================

TABLE_CELL = {
    "textAlign": "left", "padding": "10px 14px", "fontSize": "13px",
    "fontFamily": FONT_MONO, "border": "none", "borderBottom": f"1px solid {LINE}",
    "backgroundColor": SURFACE, "color": INK,
}
TABLE_HEADER = {
    "fontFamily": FONT_DISPLAY, "fontWeight": "600", "fontSize": "11px",
    "letterSpacing": "0.05em", "textTransform": "uppercase",
    "backgroundColor": PAPER, "color": INK_SOFT, "border": "none",
    "borderBottom": f"1px solid {LINE}", "padding": "10px 14px",
}
TABLE_STRIPE = [{"if": {"row_index": "odd"}, "backgroundColor": "#FAF9F5"}]


def data_table(df, page_size=10, filterable=False, sortable=False):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto", "borderRadius": "10px", "border": f"1px solid {LINE}"},
        style_cell=TABLE_CELL,
        style_header=TABLE_HEADER,
        style_data_conditional=TABLE_STRIPE,
        page_size=page_size,
        filter_action="native" if filterable else "none",
        sort_action="native" if sortable else "none",
        css=[{"selector": ".dash-spreadsheet td, .dash-spreadsheet th", "rule": "font-family: " + FONT_MONO}],
    )


# ============================================================
# DASH APP
# ============================================================

app = Dash(__name__)
app.title = "Executive Root-Cause Dashboard"

app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; }
  body { margin: 0; }
  ::-webkit-scrollbar { height: 10px; width: 10px; }
  ::-webkit-scrollbar-thumb { background: #D8D4C8; border-radius: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""

n_machines = max(s0["N_Machines"], s1["N_Machines"])
root_cause_fg, root_cause_bg, root_cause_txt = significance_chip("SIGNIFICANT")

app.layout = html.Div(
    style=PAGE_STYLE,
    children=[

        # ---------------- HEADER ----------------
        html.Div(
            style={**SECTION_STYLE, "display": "flex", "justifyContent": "space-between",
                    "alignItems": "flex-end", "flexWrap": "wrap", "gap": "18px"},
            children=[
                html.Div([
                    html.Div("LINE PERFORMANCE · ROOT-CAUSE ANALYSIS", style={
                        "fontFamily": FONT_DISPLAY, "fontSize": "12px", "fontWeight": "700",
                        "letterSpacing": "0.1em", "color": ACCENT, "marginBottom": "10px",
                    }),
                    html.H1("UPH & Availability Root-Cause Dashboard", style={
                        "fontFamily": FONT_DISPLAY, "fontSize": "30px", "fontWeight": "700",
                        "margin": "0 0 6px 0", "color": INK, "letterSpacing": "-0.01em",
                    }),
                    html.P(f"{PERIOD0} \u2192 {PERIOD1} pooled analysis, benchmarked against historical variance",
                            style={"margin": "0", "color": INK_SOFT, "fontSize": "14.5px"}),
                ]),
                html.Div(
                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                    children=[
                        meta_pill(f"{PERIOD0} · {s0['N_Records']} records"),
                        meta_pill(f"{PERIOD1} · {s1['N_Records']} records"),
                        meta_pill(f"{n_machines} machines"),
                        meta_pill(f"z-threshold ±{Z_SIGNIFICANT:.1f}"),
                    ]
                ),
            ]
        ),

        # ---------------- KPI ROW ----------------
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.Div("Headline KPIs — Jul vs. Aug", style=SECTION_TITLE_STYLE),
                kpi_row,
            ]
        ),

        # ---------------- ROOT CAUSE CHARTS ----------------
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                                  "marginBottom": "4px"}, children=[
                    html.Div("Root-cause decomposition", style=SECTION_TITLE_STYLE),
                    chip(f"Root cause: {cf['Root_Cause']} ({cf['Root_Cause_Contribution_%']:.0f}% of gap)",
                          ACCENT, "#FBEADF"),
                ]),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px", "marginTop": "12px"},
                    children=[
                        dcc.Graph(figure=fig_drop_share, config={"displayModeBar": False}),
                        dcc.Graph(figure=fig_counterfactual, config={"displayModeBar": False}),
                    ]
                )
            ]
        ),

        # ---------------- SIGNIFICANCE / DISTRIBUTION ----------------
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.Div("Statistical significance & distribution", style=SECTION_TITLE_STYLE),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px"},
                    children=[
                        dcc.Graph(figure=fig_significance, config={"displayModeBar": False}),
                        dcc.Graph(figure=fig_box, config={"displayModeBar": False}),
                    ]
                )
            ]
        ),

        # ---------------- AVAILABILITY DEEP-DIVE ----------------
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                                  "flexWrap": "wrap", "gap": "10px", "marginBottom": "4px"}, children=[
                    html.Div([
                        html.Div("Availability deep-dive", style=SECTION_TITLE_STYLE),
                        html.P("Source Availability vs. the MTTR/MTBI reliability model — "
                                "AV_Model_% = 100 / (1 + MTTR / MTBI)",
                                style={"margin": "-12px 0 0 0", "color": INK_FAINT, "fontSize": "13px"}),
                    ]),
                    chip(f"Root driver: {av_driver['Root_Cause']} "
                          f"({fmt_num(av_driver['Root_Cause_Contribution_%'], 0)}% of model gap)",
                          ACCENT, "#FBEADF"),
                ]),

                html.Div(avail_kpi_row, style={"marginTop": "18px", "marginBottom": "22px"}),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px"},
                    children=[
                        dcc.Graph(figure=fig_source_model, config={"displayModeBar": False}),
                        dcc.Graph(figure=fig_availability_driver, config={"displayModeBar": False}),
                    ]
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px", "marginTop": "18px"},
                    children=[
                        dcc.Graph(figure=fig_avail_z, config={"displayModeBar": False}),
                        dcc.Graph(figure=fig_avail_box, config={"displayModeBar": False}),
                    ]
                ),

                html.Div(style={
                    "marginTop": "22px", "padding": "18px 20px", "borderRadius": "14px",
                    "backgroundColor": PAPER, "border": f"1px solid {LINE}",
                }, children=[
                    html.Div("Takeaways", style={**LABEL_STYLE, "marginBottom": "10px"}),
                    html.Ul(style={"margin": "0", "paddingLeft": "18px", "color": INK_SOFT,
                                     "fontSize": "13.5px", "lineHeight": "1.85"}, children=[
                        html.Li([
                            "Source Availability moves ", html.B(f"{fmt_num(s0['Availability_%_mean'])}%"),
                            " \u2192 ", html.B(f"{fmt_num(s1['Availability_%_mean'])}%"),
                            f" ({fmt_pct(availability_gap)}), z-score ",
                            html.B(fmt_num(avail_sig_results['Availability_%']['z'])),
                            f" \u2192 {avail_sig_results['Availability_%']['label'].replace('_', ' ').title()}.",
                        ]),
                        html.Li([
                            "MTTR moves ", html.B(f"{fmt_num(s0['MTTR_mean'], 4)}"), " \u2192 ",
                            html.B(f"{fmt_num(s1['MTTR_mean'], 4)}"),
                            f", z-score {fmt_num(avail_sig_results['MTTR']['z'])}.",
                        ]),
                        html.Li([
                            "MTBI moves ", html.B(f"{fmt_num(s0['MTBI_mean'], 4)}"), " \u2192 ",
                            html.B(f"{fmt_num(s1['MTBI_mean'], 4)}"),
                            f", z-score {fmt_num(avail_sig_results['MTBI']['z'])}.",
                        ]),
                        html.Li([
                            "Model-based dominant driver: ", html.B(av_driver["Root_Cause"] or "N/A"),
                            f" ({fmt_num(av_driver['Root_Cause_Contribution_%'])}% of the model-availability gap closed).",
                        ]),
                        html.Li(
                            "Where source and model Availability diverge, treat the MTTR/MTBI model as "
                            "exploratory — not a validated reconstruction of the source KPI.",
                            style={"color": INK_FAINT},
                        ),
                    ]),
                ]),
            ]
        ),

        # ---------------- TABLES ----------------
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.Div("Supporting data", style=SECTION_TITLE_STYLE),

                html.Div("Period summary", style={**LABEL_STYLE, "marginTop": "4px"}),
                data_table(summary_df),

                html.Div("Significance table", style={**LABEL_STYLE, "marginTop": "26px"}),
                data_table(significance_df),

                html.Div("Availability deep-dive — period summary", style={**LABEL_STYLE, "marginTop": "26px"}),
                data_table(availability_summary_df),

                html.Div("Availability deep-dive — significance table", style={**LABEL_STYLE, "marginTop": "26px"}),
                data_table(avail_significance_df),

                html.Div("Availability deep-dive — driver decomposition", style={**LABEL_STYLE, "marginTop": "26px"}),
                data_table(driver_df),

                html.Div("Record-level data", style={**LABEL_STYLE, "marginTop": "26px"}),
                data_table(pooled_actual.round(4), page_size=15, filterable=True, sortable=True),
            ]
        ),

        html.Div(f"Generated from {MACHINE_XLSX} and {HISTORICAL_XLSX}", style={
            "textAlign": "center", "color": INK_FAINT, "fontSize": "12px", "marginTop": "8px",
        }),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
