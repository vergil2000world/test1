# test1
mon premier test

## Downtime CA / MCA analyzers

Both scripts analyze a machine-downtime CSV with columns `Machine ID`,
`Duration (sec)`, `ReasonCode`, `Lot ID`, `Product Code`, `Machine Group`.

### `scripts/analyze_ca_mca.py` — statistical CA/MCA (recommended)

Uses [`prince`](https://github.com/MaxHalford/prince) 0.14.0 to run real
Correspondence Analysis (CA) and Multiple Correspondence Analysis (MCA):

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
  top-N raw combinations by downtime, and the top-N closest cross-variable
  category pairs in the MCA map (most "associated").
- Output is organized **variable by variable**, one subfolder per variable
  under `--outdir`, each holding that variable's own Pareto detail plus its
  CA/top-N reports against every other variable; multi-way MCA reports live
  in `MultiWayMCA/`, and duration-sum pivot CSVs for every pair live in
  `pivots/`. See the script's docstring for the full layout.

```bash
pip install -r requirements.txt
python3 scripts/analyze_ca_mca.py path/to/your.csv --outdir output --top 5 --quantiles 4
```

A synthetic example input/output is checked in under
`sample_data/sample_downtime.csv` and `sample_data/sample_output/` — run the
command above against your own file to get the real reports.

### `scripts/analyze_downtime.py` — lightweight, no dependencies

A simpler, dependency-free (stdlib only) fallback that produces the same
kind of Pareto/top-N-combination text reports (business-style "cause
analysis") without needing pandas/prince installed:

```bash
python3 scripts/analyze_downtime.py path/to/your.csv --outdir reports --top 5
```
