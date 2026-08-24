# Economic Connectedness & Intergenerational Mobility: A Bayesian Hierarchical Analysis

A Bayesian analysis of how **economic connectedness (EC)**: the degree to
which low and high-income people are friends within a county, relates to
**intergenerational mobility (IM)**: how much a child's adult income is
determined by their parents' income, across approximately 3,000 U.S. counties nested
within states.

The core methodological point: counties are grouped within states, so
observations are not independent. A **hierarchical (partial-pooling) model**
sits between ignoring state structure entirely (complete pooling) and fitting a
separate regression per state (no pooling), letting small states borrow
strength from the overall trend.

> This is a Python port of a project originally written in R (`rstanarm`). The
> Python implementation uses [bambi](https://bambinos.github.io/bambi/) over
> [PyMC](https://www.pymc.io/), with [ArviZ](https://python.arviz.org/) for
> diagnostics.

## Research questions

1. **Is there a relationship between EC and IM, and in which direction?**
   Fit a Bayesian linear regression (complete pooling) of IM on EC.
2. **How much IM variation is between-state vs within-state, and does modeling
   the state structure change the EC–IM estimate?**
   Fit a varying-intercept hierarchical model and decompose the variance.

## Findings

- The EC–IM relationship is **positive** under both models - higher
  connectedness is associated with higher upward mobility. The pooled model
  estimates a slope 95% credible interval of **(0.237, 0.253)**.
- Roughly **60% of IM variation is between states**, which justifies the
  hierarchical model over complete pooling.
- Accounting for state structure **shrinks the EC slope** to a 95% credible
  interval of **(0.155, 0.172)**: part of what the pooled model attributed to
  EC was really unmodeled state-level variation. The direction of the effect is
  unchanged, but its magnitude is smaller and the posterior predictive check
  fits the data noticeably better.

Results below are produced on the full dataset (3,009 counties across 51
states) with 4 chains × 2,000 draws; both models pass R-hat and ESS
convergence checks.

| Figure | What it shows |
| --- | --- |
| `figures/ec_vs_im.png` | EC vs IM scatter with OLS trend — the positive relationship. |
| `figures/im_by_state.png` | IM by state (ordered by median) — motivates grouping. |
| `figures/ppc_complete_pooling.png` | Pooled posterior predictive check — looser fit. |
| `figures/ppc_hierarchical.png` | Hierarchical posterior predictive check — tighter fit. |
| `figures/trace_*.png` | Trace + density panels for population-level parameters. |

## Repository layout

```
ec-im-bayesian/
├── src/
│   ├── data.py          # load + merge + clean the two datasets
│   ├── models.py        # complete-pooling & hierarchical models, variance decomposition
│   ├── diagnostics.py   # R-hat, ESS, credible intervals, pass/fail convergence check
│   ├── plots.py         # scatter, boxplot, posterior predictive, trace
│   └── run_analysis.py  # end-to-end pipeline (CLI)
├── tests/               # unit tests for the preprocessing logic
├── notebooks/           # narrative walkthrough
├── data/                # place CSVs here (see data/README.md)
└── figures/             # generated output
```

## Getting started

```bash
git clone https://github.com/<your-username>/ec-im-bayesian.git
cd ec-im-bayesian
pip install -r requirements.txt
```

Add the two CSVs to `data/` (see [`data/README.md`](data/README.md)), then run
the full pipeline:

```bash
python -m src.run_analysis --data-dir data --fig-dir figures
```

This cleans the data, fits both models, checks convergence, writes figures to
`figures/`, and prints the answers to both research questions.

## Modeling details

**Complete pooling** — a single Normal-likelihood regression:

```
IM_i ~ Normal(β0 + β1 · EC_i, σ²)
```

**Hierarchical (varying intercept by state)**:

```
IM_ij ~ Normal(β0 + b_j + β1 · EC_ij, σ_y²)
b_j   ~ Normal(0, σ_μ²)
```

where `b_j` is state `j`'s offset from the global intercept. Priors are weakly
informative (`β0 ~ Normal(0.4, 0.5)`, `β1 ~ Normal(0, 0.5)`,
`σ ~ Exponential(1)`); with ~3,000 counties the likelihood dominates the prior.
Sampling uses 4 chains; convergence is checked via R-hat, effective sample
size, trace plots, and posterior predictive checks.

## Testing

```bash
pytest
```

The tests cover the merge-and-clean logic — the FIPS join, column pruning,
state parsing, and removal of missing, negative, out-of-range, and summary
rows — using small synthetic frames so they run without the full dataset or any
MCMC sampling.

## Limitations

- State is the only grouping level; commuting zones might cluster counties more
  meaningfully.
- Both models assume a constant EC slope across states and constant residual
  variance (the raw scatter shows mild heteroskedasticity).
- Inference is at the county level and does not transfer to individuals.
- EC is derived from a constrained Facebook-based sample, so it under-covers
  people who don't use the platform.
