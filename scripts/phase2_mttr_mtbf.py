"""
phase2_mttr_mtbf.py
Purpose: within an Availability loss, decide whether MTTR (repair time,
severity) or MTBF (time between failures, frequency) is the dominant
driver, and whether that shift is real vs. random sample noise.

METHOD NOTE - Availability is NEVER recomputed here
------------------------------------------------------------------
Availability_% is the one that comes directly from the source data
(see io_loaders.py) and is used as-is throughout the whole pipeline -
Phase 1 already reports it accurately. This module does NOT reconstruct
or model Availability from MTTR/MTBF (no 100/(1+MTTR/MTBF) formula
anywhere here). MTTR and MTBF are analysed purely to answer WHICH of
the two is driving the change, using the rho = MTTR/MTBF decomposition:

    ln(rho1/rho0) = (+1)*ln(MTTR1/MTTR0) + (-1)*ln(MTBF1/MTBF0)

Each term compares a duration ONLY to itself across time - MTTR to
MTTR, MTBF to MTBF - never a cross-period hybrid of the two, so this
can never imply a downtime total exceeding a period's real calendar
horizon (the flaw in the original cross-period counterfactual). The
trade-off, by design: rho's decomposition tells you WHO dominates
(thanks to the strict monotonicity of Availability = 1/(1+rho)), not
the exact point-change in Availability - for that, read the real
Availability_% already reported in Phase 1.
"""

import numpy as np

import config
import stats_helpers as sh


def rho_decomposition(s0, s1):
    mttr0, mttr1 = s0.get("MTTR_mean"), s1.get("MTTR_mean")
    mtbf0, mtbf1 = s0.get("MTBF_mean"), s1.get("MTBF_mean")

    # rho = MTTR * MTBF^(-1)  ->  ln(rho1/rho0) = (+1)*ln(MTTR1/MTTR0) + (-1)*ln(MTBF1/MTBF0)
    # Each term compares a duration ONLY to itself across time - never
    # mixed with the other period's other metric - so no calendar-horizon
    # violation is possible, and Availability itself is never touched.
    contribution_mttr = np.log(mttr1 / mttr0) if mttr0 and mttr1 and mttr0 > 0 and mttr1 > 0 else np.nan
    contribution_mtbf = -np.log(mtbf1 / mtbf0) if mtbf0 and mtbf1 and mtbf0 > 0 and mtbf1 > 0 else np.nan

    terms = {"MTTR": contribution_mttr, "MTBF": contribution_mtbf}
    valid_terms = {k: v for k, v in terms.items() if v == v}  # drop NaN
    loss = {k: v for k, v in valid_terms.items() if v > 0}     # rho UP = Availability DOWN = loss
    offset = {k: v for k, v in valid_terms.items() if v <= 0}
    loss_sum, offset_sum = sum(loss.values()), sum(offset.values())

    def share(term):
        group = loss_sum if term > 0 else offset_sum
        return round(term / group * 100, 2) if group else 0.0

    # Root cause = the dominant LOSS factor specifically (never a factor
    # that improved), matching the same convention used in Phase 1.
    root = max(loss, key=loss.get) if loss else None
    contribution_pct = share(loss[root]) if root else np.nan

    return {
        "MTTR_Jul": mttr0, "MTTR_Aug": mttr1, "MTBF_Jul": mtbf0, "MTBF_Aug": mtbf1,
        "ln_rho_Total": (contribution_mttr if contribution_mttr == contribution_mttr else 0)
                        + (contribution_mtbf if contribution_mtbf == contribution_mtbf else 0),
        "Contribution_MTTR": contribution_mttr, "Contribution_MTBF": contribution_mtbf,
        "MTTR_Share_%": share(contribution_mttr) if contribution_mttr == contribution_mttr else None,
        "MTBF_Share_%": share(contribution_mtbf) if contribution_mtbf == contribution_mtbf else None,
        "Root_Cause": root, "Root_Cause_Contribution_%": contribution_pct,
        "Note": "Root cause via rho=MTTR/MTBF log decomposition (never mixes periods, never "
                "recomputes Availability). Real Availability_% (Phase 1) is the only Availability "
                "used in this pipeline; rho's share tells you WHO drives its change, not the "
                "exact point-change itself.",
    }


def run(s0, s1, hist_df):
    decomp = rho_decomposition(s0, s1)
    sig = sh.significance_report(config.PHASE2_METRICS, s0, s1, hist_df)
    v = sh.verdict(decomp["Root_Cause"], decomp["Root_Cause_Contribution_%"], sig)
    return {
        "decomp": decomp, "sig": sig,
        "randomness": sh.randomness_label(sig, config.PHASE2_METRICS),
        "verdict": v,
    }
