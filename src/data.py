"""Data loading and preprocessing for the EC-IM analysis.

Merges the Social Capital Atlas (economic connectedness) and Opportunity
Atlas (intergenerational mobility) county-level datasets into a single clean
analysis frame. Mirrors the preprocessing done in the original R report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns kept for the analysis. We use ec_county (baseline county-level EC)
# and kfr_pooled_pooled_p25 (upward mobility for children from low-income
# backgrounds), which are the measures aligned with the research questions.
_EC_COL = "ec_county"
_IM_COL = "kfr_pooled_pooled_p25"
_DROP_COLS = ["state", "county.y", "ec_high_county", "kfr_pooled_pooled_p75"]


def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two raw CSVs from ``data_dir``.

    Raises
    ------
    FileNotFoundError
        If either expected CSV is missing.
    """
    atlas_path = data_dir / "q1_atlas_outcomes.csv"
    social_path = data_dir / "q1_social_capital_county.csv"

    for path in (atlas_path, social_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Expected data file not found: {path}. "
                "See data/README.md for how to obtain the datasets."
            )

    atlas_df = pd.read_csv(atlas_path)
    social_df = pd.read_csv(social_path)
    logger.info("Loaded atlas (%d rows) and social capital (%d rows)",
                len(atlas_df), len(social_df))
    return atlas_df, social_df


def build_analysis_frame(atlas_df: pd.DataFrame,
                         social_df: pd.DataFrame) -> pd.DataFrame:
    """Merge and clean the raw frames into the final analysis dataset.

    Steps
    -----
    1. Build a FIPS key on the atlas frame (state * 1000 + county) to match
       the social capital ``county`` column.
    2. Full-outer-join so no rows are silently dropped before cleaning.
    3. Drop columns not used by the analysis.
    4. Remove rows with any missing or negative values, "average" summary
       rows, and IM values >= 1 (percentile measures cannot exceed 1; values
       above it come from privacy noise added by the data providers).
    5. Derive a ``state`` grouping column from the county name.
    """
    atlas_df = atlas_df.assign(fips=atlas_df["state"] * 1000 + atlas_df["county"])

    merged = social_df.merge(
        atlas_df, how="outer", left_on="county", right_on="fips",
        suffixes=("", ".y"),
    )

    merged = merged.drop(columns=[c for c in _DROP_COLS if c in merged.columns])

    n_start = len(merged)

    numeric = merged.select_dtypes("number")
    has_missing = merged.isna().any(axis=1)
    has_negative = (numeric < 0).any(axis=1)
    is_average = merged["county_name"].str.contains(
        "average", case=False, na=False)
    im_out_of_range = merged[_IM_COL] >= 1

    clean = merged[~(has_missing | has_negative | is_average | im_out_of_range)].copy()

    # Derive state from the trailing token of "County, ST".
    clean["state"] = clean["county_name"].str.rsplit(",", n=1).str[-1].str.strip()

    n_dropped = n_start - len(clean)
    logger.info("Dropped %d of %d rows (%.1f%%) during cleaning",
                n_dropped, n_start, 100 * n_dropped / n_start)

    return clean.reset_index(drop=True)


def load_clean(data_dir: Path) -> pd.DataFrame:
    """Convenience wrapper: load raw CSVs and return the clean analysis frame."""
    atlas_df, social_df = load_raw(data_dir)
    return build_analysis_frame(atlas_df, social_df)
