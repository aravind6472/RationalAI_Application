"""Missing-data diagnostics: are the rows dropped in cleaning missing at random?

`build_analysis_frame` drops ~8% of rows (missing/negative values, "average"
summary rows, out-of-range IM). Dropping rows only leaves the analysis unbiased
if the dropped rows don't differ systematically from the kept ones on the
analysis variables. This module rebuilds the merged frame, reconstructs the same
drop mask used in `data.build_analysis_frame`, and reports:

  - per-reason drop counts,
  - kept-vs-dropped comparison on EC and IM (means, standardized mean
    difference, two-sample t-test),
  - the AUC of a logistic model predicting "was this row dropped?" from the
    observed variables. AUC near 0.5 means drop-status is not predictable from
    the observed data (consistent with missing-at-random); well above 0.5 means
    the dropped rows differ systematically.

"Missing at random" is not strictly testable (it concerns unobserved values),
so this checks whether the *observed* characteristics of dropped rows differ
from kept rows -- evidence, not a verdict.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .data import _EC_COL, _IM_COL, _DROP_COLS

logger = logging.getLogger(__name__)


@dataclass
class VarComparison:
    variable: str
    mean_kept: float
    mean_dropped: float
    std_mean_diff: float
    p_value: Optional[float]
    n_dropped_observed: int


@dataclass
class MissingnessReport:
    n_total: int
    n_kept: int
    n_dropped: int
    reason_counts: dict
    comparisons: list = field(default_factory=list)
    dropped_predictable_auc: Optional[float] = None

    @property
    def drop_rate(self) -> float:
        return self.n_dropped / self.n_total if self.n_total else float("nan")


def _rebuild_merged(atlas_df: pd.DataFrame, social_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the merged frame from data.build_analysis_frame, pre-clean."""
    atlas_df = atlas_df.assign(fips=atlas_df["state"] * 1000 + atlas_df["county"])
    merged = social_df.merge(
        atlas_df, how="outer", left_on="county", right_on="fips",
        suffixes=("", ".y"),
    )
    return merged.drop(columns=[c for c in _DROP_COLS if c in merged.columns])


def _drop_masks(merged: pd.DataFrame):
    numeric = merged.select_dtypes("number")
    masks = {
        "missing": merged.isna().any(axis=1),
        "negative": (numeric < 0).any(axis=1),
        "average_row": merged["county_name"].str.contains("average", case=False, na=False),
        "im_out_of_range": merged[_IM_COL] >= 1,
    }
    union = masks["missing"] | masks["negative"] | masks["average_row"] | masks["im_out_of_range"]
    return masks, union


def _compare(merged, kept_mask, var) -> Optional[VarComparison]:
    from scipy import stats

    kept = merged.loc[kept_mask, var].dropna()
    dropped = merged.loc[~kept_mask, var].dropna()
    if len(kept) < 2 or len(dropped) < 2:
        return None
    pooled_sd = np.sqrt(
        ((kept.var(ddof=1) * (len(kept) - 1)) + (dropped.var(ddof=1) * (len(dropped) - 1)))
        / (len(kept) + len(dropped) - 2)
    )
    smd = (dropped.mean() - kept.mean()) / pooled_sd if pooled_sd > 0 else float("nan")
    _, p = stats.ttest_ind(dropped, kept, equal_var=False)
    return VarComparison(
        variable=var,
        mean_kept=float(kept.mean()),
        mean_dropped=float(dropped.mean()),
        std_mean_diff=float(smd),
        p_value=float(p),
        n_dropped_observed=len(dropped),
    )


def _predictability_auc(merged, drop_union) -> Optional[float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    cols = [c for c in (_EC_COL, _IM_COL) if c in merged.columns]
    if not cols:
        return None
    X = merged[cols].fillna(merged[cols].median())
    y = drop_union.astype(int).to_numpy()
    if y.sum() in (0, len(y)):
        return None
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))


def analyze_missingness(atlas_df: pd.DataFrame, social_df: pd.DataFrame) -> MissingnessReport:
    """Compare kept vs dropped rows to assess whether dropping is benign."""
    merged = _rebuild_merged(atlas_df, social_df)
    masks, drop_union = _drop_masks(merged)
    kept_mask = ~drop_union

    comparisons = [c for c in (_compare(merged, kept_mask, _EC_COL),
                               _compare(merged, kept_mask, _IM_COL)) if c is not None]
    auc = _predictability_auc(merged, drop_union)

    report = MissingnessReport(
        n_total=len(merged),
        n_kept=int(kept_mask.sum()),
        n_dropped=int(drop_union.sum()),
        reason_counts={k: int(v.sum()) for k, v in masks.items()},
        comparisons=comparisons,
        dropped_predictable_auc=auc,
    )
    logger.info(
        "Missingness: dropped %d/%d (%.1f%%); reasons=%s; predictability AUC=%s",
        report.n_dropped, report.n_total, 100 * report.drop_rate,
        report.reason_counts,
        f"{auc:.3f}" if auc is not None else "n/a",
    )
    return report
