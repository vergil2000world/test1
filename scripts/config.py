"""
config.py — single source of truth for every setting the pipeline needs.
Nothing else in this suite hardcodes a path, a column name, or a threshold.
"""

# ============================================================
# INPUT FILES
# ============================================================

# --- Actual month workbook (the two periods being compared) ---
MACHINE_XLSX = r"Resources\Machine_UPH_JUL_AUG.xlsx"
SHEET0, SHEET1 = "Jul26", "Aug26"
PERIOD0, PERIOD1 = "Jul", "Aug"

# --- Historical baseline workbook (every past record, used only to get
#     a real std dev per metric for the significance test) ---
HISTORICAL_XLSX = r"Resources\export.xlsx"
HISTORICAL_SHEET = 0

# ============================================================
# COLUMN MAPPING
# Source column names differ between the actual file and the historical
# export (MACHINE_ID vs MACHINE_LABEL) - everything else downstream uses
# the mapped names below, never the raw source names directly.
# ============================================================

COLUMN_MAP_ACTUAL = {
    "machine_id": "MACHINE_ID",
    "uph": "UPH",
    "availability_source": "OEE",   # source column is named OEE but IS Availability
    "mtbf": "MTBF",
    "teud": "TEUD",                  # MTTR = TEUD / F2 (confirmed by user)
    "f2": "F2",
}

COLUMN_MAP_HISTORICAL = {**COLUMN_MAP_ACTUAL, "machine_id": "MACHINE_LABEL"}

# ============================================================
# METRICS TO ANALYSE
# Add/remove a metric here and every step (stats, significance, report)
# picks it up automatically - nothing else needs editing.
# ============================================================

PHASE1_METRICS = ["Performance_%", "Availability_%"]
PHASE2_METRICS = ["MTTR", "MTBF"]

STAT_COLUMNS = ["UPH", "Availability_%", "Performance_%", "OEE_%", "MTBF", "MTTR"]

# ============================================================
# MODEL CONSTANTS
# ============================================================

UPH_THEORETICAL = 9455

# ============================================================
# DECISION THRESHOLDS
# ============================================================

# abs(z) >= 2.0 is used as a practical approximation of the two-sided 95%
# normal cutoff (1.96). This is a decision policy, not learned from data.
Z_SIGNIFICANT = 2.0

# A factor must close >= 50% of the gap in the counterfactual to be
# considered dominant.
DOMINANCE_GAP_CLOSED_MIN = 50.0

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = "Output/Master_Root_Cause_Pooled"
