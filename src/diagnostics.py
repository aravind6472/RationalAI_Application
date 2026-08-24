"""Convergence diagnostics and posterior summaries.

Thin wrappers over ArviZ that return tidy DataFrames, so the analysis script
and notebook can report R-hat, effective sample size, and credible intervals
without repeating boilerplate.
"""

from __future__ import annotations

import arviz as az
import pandas as pd


def summary(idata, var_names: list[str] | None = None,
            hdi_prob: float = 0.95) -> pd.DataFrame:
    """Posterior summary table: mean, SD, HDI, R-hat, and ESS per parameter.

    The credible-interval keyword to ``az.summary`` has changed across ArviZ
    releases (``hdi_prob`` -> ``prob`` -> ``ci_prob``); try each in turn.
    """
    for kw in ("ci_prob", "prob", "hdi_prob"):
        try:
            return az.summary(idata, var_names=var_names, kind="all",
                              **{kw: hdi_prob})
        except TypeError:
            continue
    # Last resort: let ArviZ use its default interval width.
    return az.summary(idata, var_names=var_names, kind="all")


def credible_interval(idata, var_name: str,
                      hdi_prob: float = 0.95) -> tuple[float, float]:
    """Return the (low, high) HDI bounds for a single parameter.

    Handles both the pre-1.0 ArviZ signature (``hdi_prob``) and the 1.0+
    signature (``prob``).
    """
    try:
        hdi = az.hdi(idata, var_names=[var_name], hdi_prob=hdi_prob)
    except TypeError:
        hdi = az.hdi(idata, var_names=[var_name], prob=hdi_prob)
    bounds = hdi[var_name].values
    return float(bounds[0]), float(bounds[1])


def convergence_report(idata, rhat_threshold: float = 1.01,
                       ess_threshold: float = 400.0) -> pd.DataFrame:
    """Per-parameter convergence table with a pass/fail column.

    Returns R-hat and bulk ESS for every parameter plus a boolean ``ok``
    (R-hat below ``rhat_threshold`` and bulk ESS above ``ess_threshold``).
    Useful for hierarchical models, where a handful of correlated group-level
    parameters often need more iterations than the population-level ones.
    """
    stats = az.summary(idata, kind="diagnostics")
    stats = stats[["r_hat", "ess_bulk"]].copy()
    stats["ok"] = (stats["r_hat"] < rhat_threshold) & \
                  (stats["ess_bulk"] > ess_threshold)
    return stats


def converged(idata, rhat_threshold: float = 1.01,
              ess_threshold: float = 400.0) -> bool:
    """Cheap pass/fail convergence check across all parameters.

    True when every parameter's R-hat is below ``rhat_threshold`` and its bulk
    ESS is above ``ess_threshold``. For hierarchical models with many
    correlated group-level parameters, prefer ``convergence_report`` to see
    which parameters (if any) fall short.
    """
    report = convergence_report(idata, rhat_threshold, ess_threshold)
    return bool(report["ok"].all())
