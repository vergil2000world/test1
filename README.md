# test1
mon premier test

## Downtime CA / MCA analyzer

`scripts/analyze_downtime.py` analyzes a machine-downtime CSV (columns:
`Machine ID`, `Duration (sec)`, `ReasonCode`, `Lot ID`, `Product Code`,
`Machine Group`) and generates text reports:

- **CA (Cause Analysis)** — full Pareto detail for each single dimension
  (ReasonCode, Machine ID, Machine Group, Product Code, Lot ID): count,
  total duration, % of total, cumulative %, and an 80/20 callout.
- **MCA (Multiple Cause Analysis)** — top-N most associated pair
  combinations (e.g. Machine ID + ReasonCode, Machine Group + ReasonCode,
  Product Code + ReasonCode, Lot ID + ReasonCode, Machine Group + Machine ID).
- **Top-N per group** — e.g. top 5 ReasonCodes for *each* Machine ID or
  Machine Group.
- **Pivot CSVs** — Machine/Group/Product x ReasonCode matrices for Excel.

Pure Python standard library, no dependencies required.

```bash
python3 scripts/analyze_downtime.py path/to/your.csv --outdir reports --top 5
```

Reports are written to `--outdir` (default `reports/`), including a combined
`CA_MCA_Full_Report.txt`. A synthetic example input and its generated output
are checked in under `sample_data/sample_downtime.csv` and
`sample_data/sample_reports/` — run the command above against your own file
to reproduce them for real data.
