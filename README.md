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
- **Skips variables that can't be used for CA/MCA instead of crashing.** A
  contingency table needs at least 2 rows and 2 columns — a variable with
  only 1 distinct value (or no data at all) after filtering gives a
  degenerate 1-row/1-column table with zero usable components, which is
  exactly what makes `prince`/`scikit-learn` raise an
  `n_components must be between 0 and ...` crash. The script checks every
  variable's distinct-value count up front, drops any that fail from all
  CA/MCA pairing (its univariate Pareto report is still generated — only
  the pairwise/multi-way statistical analysis is skipped), and reports
  *which* variables were skipped and *why* in `00_Summary.txt`, in
  `report.html`'s Overview tab, and on that variable's own tab. If the
  highlighted triple (Machine Group + ReasonCode + DurationLevel) includes
  a skipped variable, a substitute 3-way combination is picked from the
  remaining usable variables and the substitution is noted in the summary.
  If fewer than 2 variables are usable at all, the script exits with a
  clear error instead of producing an empty/broken report.
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
- **`report_data.json`** — every table, stat and image path the dashboard
  uses, written to disk first (same as the `.txt`/`.csv` files) and then
  *read back* to build the HTML — so `report.html` is built strictly from
  files sitting in `--outdir`, never from anything kept only in memory
  during the analysis. Open it in any text/JSON viewer to check the raw
  numbers behind the dashboard.
- **`report.html`** — a single, self-contained, offline HTML dashboard (no
  server, no external network calls — every chart is embedded as base64):
  - **Overview tab** — KPIs, the top-5 pairwise associations across *all
    15 pairs* (by Cramer's V and by lift), the top-5 three-way findings
    across *all 20 MCA combinations* (by downtime and by lift), and a
    Pareto thumbnail grid.
  - **One tab per variable** with its Pareto table+chart, and a "Compare
    with" sub-tab strip to switch between every other variable (CA stats +
    biplot map highlighting both variables' categories, a lift table, a
    top-N combination table).
  - **Multi-Cause tab** — a dropdown over the highlighted triple + all 20
    triples + all 6 variables at once; each shows its category map plus
    *every* combination found (not just the top N) in the downtime, lift,
    and closest-category-pair tables.
  - **Interpretation tab** mirroring `INTERPRETATION.md` with colored
    strength/significance/lift badges.
  - **Every table beyond a handful of rows is paginated** (10 rows/page,
    Prev/Next controls) so large tables — all 40 Lot IDs, all 45+ MCA
    combinations, etc. — stay scannable instead of dumping everything at
    once.
  Just open it in a browser. Run with `--rebuild-html-only` to regenerate
  just `report.html` from an existing `report_data.json` — no CSV, no
  re-run of the analysis.
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
# rebuild only the dashboard from an existing run's report_data.json (no CSV needed):
python3 analyze_ca_mca.py --outdir output --rebuild-html-only
```

A synthetic example input/output is checked in under
`sample_data/sample_downtime.csv` and `sample_data/sample_output/` for
reference.

### `scripts/generate_output_dashboard.py` — standalone viewer, no analysis needed

A **separate** tool from `report.html`, for a different job: turning an
**already-existing** `output/` folder (from this run, an old run, or one
copied from another machine) into a browsable HTML page, without needing
to re-run — or even have installed — `prince`/`scikit-learn`/`matplotlib`/
`pandas`. It only uses the Python standard library.

- Discovers variable folders, `MultiWayMCA/`, and `pivots/` by scanning
  the folder itself (no hardcoded variable list), so it keeps working
  even if the set of analyzed columns changes.
- **Accurate by construction**: every `.txt` report is shown verbatim
  inside a `<pre>` block — the exact bytes `analyze_ca_mca.py` wrote,
  never re-parsed or guessed. `.csv` pivots are parsed with the stdlib
  `csv` module into real, paginated tables (unambiguous). PNGs are
  embedded as base64 (the exact image bytes). `INTERPRETATION.md` is
  rendered as styled HTML via a small tailored markdown converter, with
  a one-click "View raw markdown" toggle to see the untouched original.
- Same tabbed layout as `report.html` (Overview, one tab per variable
  with "Compare with" sub-tabs, Multi-Cause, Pivots, Interpretation,
  Summary), with the same table pagination.

```bash
cd scripts
python3 generate_output_dashboard.py                    # reads ./output, writes ./output/dashboard.html
python3 generate_output_dashboard.py path/to/output --out somewhere/dashboard.html
```

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
