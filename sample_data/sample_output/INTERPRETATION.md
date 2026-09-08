# Downtime CA / MCA - Interpretation

*Auto-generated from `sample_downtime.csv` - 1500 events, 506,600 sec (140.7 hr) of total downtime.*

## Data scope
- No Int/interruption flag column was found - all rows were used.
- `DurationLevel` categories (quantile bins): Low, Medium, High, Critical.
- Associations below use a minimum count of **5** events and show the top **5**.

## Top single-cause contributors (CA)

| Variable | Top value | % of total downtime | # values for 80% of downtime |
|---|---|---|---|
| Machine ID | M-101 | 13.4% | 7 of 8 |
| Machine Group | GroupA | 39.7% | 3 of 3 |
| ReasonCode | Changeover | 44.6% | 4 of 10 |
| Product Code | PRD-E | 21.4% | 4 of 5 |
| Lot ID | LOT-0032 | 3.6% | 30 of 40 |
| DurationLevel | Critical | 60.8% | 2 of 4 |

**Single biggest lever:** `DurationLevel` = 'Critical' alone accounts for **60.8%** of all downtime - the highest-leverage single-variable fix available in this dataset.

## Strongest pairwise associations (Correspondence Analysis)

| Pair | Cramer's V | Strength | p-value | Most surprising cell |
|---|---|---|---|---|
| Machine Group x Machine ID | 1.000 | strong | 0 | Machine Group='GroupB' & Machine ID='M-202' (over-represented) |
| DurationLevel x ReasonCode | 0.625 | strong | 0 | DurationLevel='Critical' & ReasonCode='Changeover' (over-represented) |
| DurationLevel x Machine ID | 0.098 | negligible | 0.00303 | DurationLevel='High' & Machine ID='M-101' (over-represented) |

## Strongest associations by lift

| Combination | Lift | Count |
|---|---|---|
| Machine Group='GroupB', Machine ID='M-201' | 4.13x | 173 |
| DurationLevel='Critical', ReasonCode='PM-Scheduled' | 4.02x | 82 |
| Lot ID='LOT-0029', ReasonCode='Operator Break' | 3.14x | 5 |
| Lot ID='LOT-0001', Machine ID='M-102' | 2.69x | 11 |
| Lot ID='LOT-0011', Product Code='PRD-D' | 1.93x | 14 |

**Strongest co-occurrence:** Machine Group='GroupB', Machine ID='M-201' happens **4.13x** more often than random chance would predict - worth investigating as a potential shared root cause.

## Multi-cause highlight: Machine Group + ReasonCode + DurationLevel

- Largest raw combination: Machine Group='GroupC', ReasonCode='Changeover', DurationLevel='Critical' -> 64,327 sec (12.7% of total downtime, 90 events).
- Strongest lift combination: Machine Group='GroupA', ReasonCode='PM-Scheduled', DurationLevel='Critical' -> 4.77x expected rate (37 events).
- Closest categories in the MCA map: ReasonCode: Sensor Fault and DurationLevel: Low (distance 0.054) - these travel together most.
- These 3 variables together explain **24.3%** of the total inertia (variance) on the first 2 MCA dimensions.

## Notes & recommendations

- Prioritize fixes in this order: the single biggest contributor above, then the strongest significant pairwise association, then the strongest lift combination - each is progressively more specific (and rarer) but often points to a more precise, fixable root cause.
- Associations built on fewer than 5 events were dropped from the lift tables to avoid over-reacting to one-off coincidences; lower `--min-count` to see rarer combinations (with less statistical confidence).
- `DurationLevel` thresholds are quantile-based (relative to this dataset) - they will shift if the dataset changes substantially, so re-run this script rather than hand-copying the bin edges.
- See the `<Variable>/` folders (or report.html) for full detail and plots behind every number in this summary.
- A pair with Cramer's V near 1.0 (e.g. Machine ID x Machine Group) can simply reflect a structural/hierarchical relationship already known by construction (each machine belongs to exactly one group) rather than a new finding - treat those as expected, not actionable.
