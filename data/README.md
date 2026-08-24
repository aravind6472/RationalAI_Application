# Data

This analysis merges two public county-level datasets. Place the following
CSVs in this directory before running the pipeline:

- `q1_atlas_outcomes.csv` — intergenerational mobility outcomes
- `q1_social_capital_county.csv` — economic connectedness measures

## Sources

- **Opportunity Atlas** (intergenerational mobility): predicted adult outcomes
  for children by county, race, gender, and parental income percentile, built
  from linked Census Bureau administrative records, federal tax records, and
  the American Community Survey. Children in the cohort were born 1978–1983.
- **Social Capital Atlas** (economic connectedness): county-level social
  capital measures derived from aggregated, privacy-protected Facebook
  friendship data (users aged 25–44, active, ≥100 U.S. friends, valid ZIP).

Both providers add statistical noise to released estimates for privacy, so a
small number of out-of-range values (e.g. mobility percentiles ≥ 1) appear and
are removed during preprocessing.

## Key variables used

| Column | Meaning |
| --- | --- |
| `ec_county` | Economic connectedness — 2× the share of high-SES friends among low-SES individuals (theoretical range 0–2). |
| `kfr_pooled_pooled_p25` | Upward mobility: predicted adult income rank for children from the 25th parental income percentile. |
| `county_name` | "County, ST" — the state suffix is parsed into the grouping variable. |
