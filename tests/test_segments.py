"""Segment heterogeneity with deterministic quantile bins."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from abtest.segments import segment_effects


def make_conn(n=40_000, effect_by_f0_half=None, seed=0):
    """Synthetic data; optionally a strong treatment effect on high-f0 users."""
    rng = np.random.default_rng(seed)
    f0 = rng.normal(0, 1, n)
    f0[rng.random(n) < 0.26] = 1.616365  # giant tie block, like the real data
    treatment = (rng.random(n) < 0.85)
    base = 0.02 + treatment * 0.0
    if effect_by_f0_half is not None:
        up = f0 > 0
        base = base + treatment * np.where(up, effect_by_f0_half, 0.001)
    conversion = (rng.random(n) < base)
    visit = (rng.random(n) < 0.05)
    df = pd.DataFrame(
        {"f0": f0, "treatment": treatment,
         "conversion": conversion, "visit": visit}
    )
    conn = duckdb.connect()
    conn.register("raw_df", df)
    conn.execute("CREATE VIEW raw AS SELECT * FROM raw_df")
    return conn


def test_returns_all_bins():
    conn = make_conn()
    segs = segment_effects(conn, "f0", "conversion", n_bins=4)
    assert len(segs) == 4
    assert list(segs["segment"]) == [0, 1, 2, 3]
    assert {"n_t", "n_c", "rate_t", "rate_c", "diff", "ci_low", "ci_high",
            "p_value", "p_adj", "significant"} <= set(segs.columns)


def test_deterministic_across_calls():
    conn = make_conn()
    a = segment_effects(conn, "f0", "conversion")
    b = segment_effects(conn, "f0", "conversion")
    pd.testing.assert_frame_equal(a, b)


def test_detects_injected_heterogeneity():
    conn = make_conn(n=80_000, effect_by_f0_half=0.02, seed=3)
    segs = segment_effects(conn, "f0", "conversion", n_bins=4)
    # top bin (largest f0) carries the injected effect
    assert segs.iloc[-1]["diff"] > segs.iloc[0]["diff"]
    assert segs.iloc[-1]["diff"] > 0.01


def test_rejects_unknown_outcome_feature_and_rel():
    conn = make_conn(n=5_000)
    with pytest.raises(ValueError):
        segment_effects(conn, "f0", "revenue")
    with pytest.raises(ValueError):
        segment_effects(conn, "f99")
    with pytest.raises(ValueError):
        segment_effects(conn, "f0", "conversion", rel="missing_rel")


def test_holm_marking():
    conn = make_conn(n=40_000)
    segs = segment_effects(conn, "f0", "conversion")
    assert segs["significant"].sum() <= 4
