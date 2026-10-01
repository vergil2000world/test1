# Holiday Impact KPI Plan (2-Year Descriptive Analysis)

Here's a concrete set of KPIs and a descriptive analysis plan designed specifically for a 2-year constraint — built to extract maximum signal without pretending to have statistical power you don't have.

## Core KPI 1 — Holiday Impact % (per category, per year)

```
Impact % = (Normal_day_avg − Holiday_day_avg) / Normal_day_avg × 100
```

Calculate this separately for each year, not pooled — this is your most important structural choice.

| Category | 2025 Impact % | 2026 Impact % | Average | Spread |
| --- | --- | --- | --- | --- |
| Hijri holidays | −27.8% | −31.2% | **−29.5%** | 3.4 pts |
| Gregorian holidays | −37.7% | −35.0% | **−36.4%** | 2.7 pts |
| Bridge days | +84.8% | ? | ? | ? |

**The "Spread" column is your real finding** — it tells you whether the two years agree (small spread = probably a real, stable effect) or disagree wildly (large spread = can't trust either number alone).

## Core KPI 2 — Consistency Ratio (your substitute for a significance test)

```
Consistency Ratio = Spread_between_years / Average_impact
```

- Ratio < 0.2 → the two years roughly agree, effect looks stable
- Ratio > 0.5 → the two years disagree almost as much as the effect itself — treat with real caution

This is the single most honest number you can report with n=2.

## Core KPI 3 — Lost Production Days (interpretable business metric)

```
Lost Days = Σ(Normal_avg − Actual_holiday_production) / Normal_avg
```

Reported per year and combined: *"Hijri holidays cost the equivalent of 1.8 production-days in 2025 and 2.1 in 2026."*

## Core KPI 4 — Category Share of Total Annual Loss

```
Share % = Total_loss_from_category / Total_loss_from_all_special_days × 100
```

Tells you whether Hijri, Gregorian, or Bridge days dominate your total disruption — useful for prioritizing what to investigate or mitigate operationally.

## Core KPI 5 — Year-over-Year Trend Direction (not a trend line, just a signal)

```
Direction = sign(Impact_2026 − Impact_2025)
```

With only 2 points you cannot fit a trend, but you can flag: *"the Hijri effect deepened slightly from 2025 to 2026"* — descriptive, not predictive.

## Core KPI 6 — Outlier Flag per Special Day

For each individual holiday occurrence (not aggregated), compute its individual deviation from that year's normal baseline, then flag any single day whose deviation is more than ~2x the category average — this is how you catch your suspicious "bridge day +996 units" case before it contaminates the averages.

## Descriptive analysis sequence (step-by-step)

1. **Build the raw comparison table** — normal-day baseline, holiday-day values, both years side by side (not pre-averaged).
2. **Flag outliers** at the individual-day level before computing any average (KPI 6).
3. **Compute Impact % per year, per category** (KPI 1) — keep years separate at this stage.
4. **Compute the Consistency Ratio** (KPI 2) to judge how much to trust the pooled average.
5. **Only then** pool into a 2-year average, with the spread reported alongside it — never a bare single number.
6. **Compute Lost Days and Category Share** (KPIs 3–4) for business-facing communication.
7. **Visualize**: a grouped bar chart (2025 vs 2026 side-by-side, not merged) per category — this makes the Consistency Ratio visually obvious to a reader/jury without needing the formula.

## Table format recommended for your thesis

| Category | n | 2025 avg | 2026 avg | Pooled avg | Spread | Consistency | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hijri | 2 | −27.8% | −31.2% | −29.5% | 3.4 pts | 0.12 | Stable signal |
| Gregorian | 2 | −37.7% | −35.0% | −36.4% | 2.7 pts | 0.07 | Stable signal |
| Bridge days | 2 | +84.8% | ? | ? | ? | ? | Check for outliers first |

## Language to use in the thesis text

> *"Across the two years of available data, Hijri holidays showed a consistent production reduction (−27.8% in 2025, −31.2% in 2026; consistency ratio = 0.12), suggesting a stable effect despite the limited sample. In contrast, the Bridge Day category showed an unexpected positive deviation that requires outlier verification before any conclusion can be drawn."*
