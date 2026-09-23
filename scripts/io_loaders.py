"""
io_loaders.py — everything related to READING and ENRICHING source files.
No statistics, no decomposition, no report logic lives here.
"""

import numpy as np
import pandas as pd

import config


def safe_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def enrich(df, column_map, uph_theoretical):
    """Apply the shared KPI model to any raw dataframe (an actual period
    sheet, or the historical export) that has the required raw columns.

    Availability_%  <- source column mapped as 'availability_source'
                        (named OEE in the source files, but IS Availability).
                        This is the ONLY Availability used anywhere in the
                        pipeline - it is never recomputed or modelled from
                        MTTR/MTBF; those two are analysed only to find which
                        one dominates the change (see phase2_mttr_mtbf.py),
                        not to reconstruct Availability itself.
    Performance_%   <- UPH / uph_theoretical * 100
    OEE_%           <- Availability_% * Performance_% / 100
    MTTR            <- TEUD / F2 (confirmed formula - not F2/TEUD)
    MTBF            <- direct input column (mapped as 'mtbf')
    """
    required = list(column_map.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    df = df.copy()
    df = safe_numeric(df, [column_map["uph"], column_map["availability_source"],
                            column_map["mtbf"], column_map["teud"], column_map["f2"]])

    df["MACHINE_ID"] = df[column_map["machine_id"]]
    df["UPH"] = df[column_map["uph"]]
    df["Availability_%"] = df[column_map["availability_source"]]
    df["Performance_%"] = df["UPH"] / uph_theoretical * 100.0
    df["OEE_%"] = df["Availability_%"] * df["Performance_%"] / 100.0
    df["MTBF"] = df[column_map["mtbf"]]
    df["MTTR"] = np.where(df[column_map["f2"]] > 0,
                           df[column_map["teud"]] / df[column_map["f2"]], np.nan)
    return df


def load_actual_period(xlsx_path, sheet_name, period_label, uph_theoretical, column_map):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df = enrich(df, column_map, uph_theoretical)
    df["Period"] = period_label
    return df


def load_historical_baseline(xlsx_path, sheet_name, uph_theoretical, column_map):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    return enrich(df, column_map, uph_theoretical)


def load_all():
    """One call that loads everything main.py needs, using config.py settings."""
    df0 = load_actual_period(config.MACHINE_XLSX, config.SHEET0, config.PERIOD0,
                              config.UPH_THEORETICAL, config.COLUMN_MAP_ACTUAL)
    df1 = load_actual_period(config.MACHINE_XLSX, config.SHEET1, config.PERIOD1,
                              config.UPH_THEORETICAL, config.COLUMN_MAP_ACTUAL)
    hist_df = load_historical_baseline(config.HISTORICAL_XLSX, config.HISTORICAL_SHEET,
                                        config.UPH_THEORETICAL, config.COLUMN_MAP_HISTORICAL)
    return df0, df1, hist_df
