"""Unit tests for the data preprocessing logic in src.data."""

import pandas as pd
import pytest

from src import data as data_mod


@pytest.fixture
def raw_frames():
    """Small synthetic pair of frames mirroring the real schema."""
    atlas = pd.DataFrame({
        "state": [1, 1, 6],
        "county": [1, 3, 75],
        "county_name": ["Autauga County, AL", "Baldwin County, AL",
                        "San Francisco County, CA"],
        "kfr_pooled_pooled_p25": [0.42, 0.45, 0.55],
        "kfr_pooled_pooled_p75": [0.60, 0.61, 0.70],
    })
    social = pd.DataFrame({
        "county": [1001, 1003, 6075],
        "ec_county": [0.9, 1.1, 1.4],
        "ec_high_county": [1.0, 1.2, 1.5],
    })
    return atlas, social


def test_build_frame_merges_on_fips(raw_frames):
    atlas, social = raw_frames
    out = data_mod.build_analysis_frame(atlas, social)
    assert len(out) == 3
    assert {"ec_county", "kfr_pooled_pooled_p25", "state"} <= set(out.columns)


def test_dropped_columns_removed(raw_frames):
    atlas, social = raw_frames
    out = data_mod.build_analysis_frame(atlas, social)
    for col in ("ec_high_county", "kfr_pooled_pooled_p75"):
        assert col not in out.columns


def test_state_derived_from_county_name(raw_frames):
    atlas, social = raw_frames
    out = data_mod.build_analysis_frame(atlas, social)
    assert set(out["state"]) == {"AL", "CA"}


def test_im_values_at_or_above_one_dropped(raw_frames):
    atlas, social = raw_frames
    atlas.loc[0, "kfr_pooled_pooled_p25"] = 1.3  # privacy-noise artifact
    out = data_mod.build_analysis_frame(atlas, social)
    assert (out["kfr_pooled_pooled_p25"] < 1).all()
    assert len(out) == 2


def test_negative_values_dropped(raw_frames):
    atlas, social = raw_frames
    social.loc[1, "ec_county"] = -0.5
    out = data_mod.build_analysis_frame(atlas, social)
    assert (out["ec_county"] >= 0).all()
    assert len(out) == 2


def test_average_rows_dropped(raw_frames):
    atlas, social = raw_frames
    atlas.loc[2, "county_name"] = "State Average, CA"
    out = data_mod.build_analysis_frame(atlas, social)
    assert not out["county_name"].str.contains("average", case=False).any()


def test_missing_values_dropped(raw_frames):
    atlas, social = raw_frames
    atlas.loc[1, "kfr_pooled_pooled_p25"] = None
    out = data_mod.build_analysis_frame(atlas, social)
    assert not out.isna().any().any()
    assert len(out) == 2


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        data_mod.load_raw(tmp_path)
