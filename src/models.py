"""Bayesian models for the EC-IM analysis.

Two models, both with a Normal likelihood on intergenerational mobility (IM)
regressed on economic connectedness (EC):

* ``fit_complete_pooling`` - a single regression ignoring state structure.
* ``fit_hierarchical`` - a varying-intercept model with a group term on state,
  partially pooling counties toward the global trend.

We use bambi (formula interface over PyMC), the closest Python analogue to the
original R ``rstanarm`` implementation. Priors are weakly informative: with
~3000 counties the likelihood dominates regardless of the exact prior.
"""

from __future__ import annotations

import logging

import bambi as bmb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_IM_COL = "kfr_pooled_pooled_p25"
_EC_COL = "ec_county"
SEED = 84735

# Weakly informative priors, matching the R report's intent.
_PRIORS = {
    "Intercept": bmb.Prior("Normal", mu=0.4, sigma=0.5),
    _EC_COL: bmb.Prior("Normal", mu=0.0, sigma=0.5),
    "sigma": bmb.Prior("Exponential", lam=1.0),
}


def fit_complete_pooling(df: pd.DataFrame, *, draws: int = 2000,
                         tune: int = 2000, chains: int = 4,
                         cores: int | None = None):
    """Fit IM ~ EC with no grouping structure (complete pooling).

    Returns the fitted ``(model, idata)`` tuple.
    """
    model = bmb.Model(
        f"{_IM_COL} ~ {_EC_COL}", data=df, priors=_PRIORS, family="gaussian",
    )
    idata = model.fit(draws=draws, tune=tune, chains=chains, cores=cores,
                      random_seed=SEED, progressbar=False)
    logger.info("Fit complete-pooling model on %d counties", len(df))
    return model, idata


def fit_hierarchical(df: pd.DataFrame, *, draws: int = 2000,
                     tune: int = 2000, chains: int = 4,
                     cores: int | None = None):
    """Fit IM ~ EC + (1 | state): varying intercept by state.

    Each state gets its own baseline IM offset drawn from a shared Normal,
    partially pooling small states toward the global intercept.
    """
    priors = dict(_PRIORS)
    priors["1|state"] = bmb.Prior(
        "Normal", mu=0.0, sigma=bmb.Prior("Exponential", lam=1.0),
    )
    model = bmb.Model(
        f"{_IM_COL} ~ {_EC_COL} + (1|state)", data=df, priors=priors,
        family="gaussian",
    )
    idata = model.fit(draws=draws, tune=tune, chains=chains, cores=cores,
                      random_seed=SEED, progressbar=False)
    logger.info("Fit hierarchical model on %d counties, %d states",
                len(df), df["state"].nunique())
    return model, idata


def variance_decomposition(idata) -> dict[str, float]:
    """Share of IM variance attributable to between-state vs within-state.

    Uses the posterior-mean of the state-intercept SD and the residual SD.
    Returns proportions in [0, 1] keyed ``between_state`` / ``within_state``.
    """
    posterior = idata.posterior
    sigma_state = float(posterior["1|state_sigma"].mean())
    sigma_resid = float(posterior["sigma"].mean())

    between = sigma_state**2
    within = sigma_resid**2
    total = between + within
    return {"between_state": between / total, "within_state": within / total}
