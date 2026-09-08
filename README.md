# test1
mon premier test

## Downtime CA / MCA analyzers

Both scripts analyze a machine-downtime CSV with columns `Machine ID`,
`Duration (sec)`, `ReasonCode`, `Lot ID`, `Product Code`, `Machine Group`,
and optionally an **`Int`** flag column (0/1: 1 = the row is a counted
interruption, 0 = not counted).

### `scripts/analyze_ca_mca.py` — the full one-script process (recommended)

Uses [`prince`](https://github.com/MaxHalford/prince) 0.14.0 to run real
Correspondence Analysis (CA) and Multiple Correspondence Analysis (MCA):

- If an `Int` column is present, **filters to only rows where `Int == 1`**
  first — every table below is built from that filtered data only (rows
  with `Int == 0` are dropped as not-counted). If no such column is found,
  all rows are used and the summary says so.
- Adds a derived **DurationLevel** feature (`Low`/`Medium`/`High`/`Critical`)
  by cutting `Duration (sec)` into quantile bins (`pandas.qcut`, `q=4` by
  default), and analyzes it alongside the 5 original columns.
- **CA** — for *every* pair of the 6 variables (all 15 combinations):
  contingency table, chi-square test, Cramer's V, eigenvalues/explained
  inertia, row/column coordinates, and the top-N statistically most
  associated cells (largest standardized residuals).
- **MCA** — for every 3-way combination of variables (all 20), for all 6
  variables at once, and specifically for **Machine Group + ReasonCode +
  DurationLevel** (highlighted): eigenvalues, category coordinates/cos2,
  top-N raw combinations by downtime, a lift table, and the top-N closest
  cross-variable category pairs in the MCA map (most "associated").
- **Lift analysis** — for every pair and every MCA combination: lift =
  observed proportion ÷ expected proportion under independence (the
  classic market-basket "association rule" metric). Lift > 1x means the
  combination co-occurs more than chance predicts. Combinations below
  `--min-count` events are excluded as too rare to trust.
- **Plots** — a Pareto chart (bars + cumulative-% line) per variable, a CA
  biplot per pair, and an MCA category map per multi-way combination, all
  saved as PNGs next to their corresponding text report.
- **`INTERPRETATION.md`** — an auto-generated, plain-language takeaways
  file at the top of `--outdir`: top single-cause contributors, strongest
  significant pairwise associations, strongest lift combinations, the
  highlighted MCA triple's headline findings, and a few notes/caveats
  (e.g. flagging trivially-strong associations from hierarchical columns).
- Output is organized **variable by variable**, one subfolder per variable
  under `--outdir`, each holding that variable's own Pareto detail plus its
  CA/Lift/top-N reports (and plots) against every other variable; multi-way
  MCA reports live in `MultiWayMCA/`, and duration-sum pivot CSVs for every
  pair live in `pivots/`. See the script's docstring for the full layout.

Both scripts default to a file named **`cleaned_file.csv` placed next to the
script itself** (`scripts/cleaned_file.csv`), so on a local setup you can
just drop your CSV there and run the script with no arguments:

```bash
pip install -r requirements.txt
cd scripts
python3 analyze_ca_mca.py                       # reads ./cleaned_file.csv, writes ./output/
# or explicitly:
python3 analyze_ca_mca.py path/to/other.csv --outdir output --top 5 --quantiles 4 --min-count 5
```

A synthetic example input/output is checked in under
`sample_data/sample_downtime.csv` and `sample_data/sample_output/` for
reference.

### `scripts/analyze_downtime.py` — lightweight, no dependencies

A simpler, dependency-free (stdlib only) fallback that produces the same
kind of Pareto/top-N-combination text reports (business-style "cause
analysis") without needing pandas/prince installed. Also defaults to
`scripts/cleaned_file.csv`:

```bash
cd scripts
python3 analyze_downtime.py                     # reads ./cleaned_file.csv, writes ./reports/
# or explicitly:
python3 analyze_downtime.py path/to/other.csv --outdir reports --top 5
```
