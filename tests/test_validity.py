import duckdb
import numpy as np
import pandas as pd
import pytest

from abtest.validity import (
    MAX_LEVEL_CARDINALITY,
    assignment_report,
    bin_expression,
    feature_balance,
    quantile_boundaries,
    share_independence,
    standardized_ate,
    standardized_mean_difference,
    treatment_counts,
)


def make_conn(n=30_000, share_by_level=None, drift=0.0, seed=42):
    """Synthetic experiment; treated share 0.85 unless overridden.

    ``f0`` — continuous, 26% of rows share one giant tie value. Assignment
    is independent of f0 by default; ``drift`` tilts the share linearly
    across f0's quantile bins.

    ``f3`` — 10 discrete levels; ``share_by_level`` maps each level to its
    own treated share (takes precedence over ``drift``).
    """
    rng = np.random.default_rng(seed)
    f0 = rng.normal(0, 1, n)
    tie_mask = rng.random(n) < 0.26
    f0[tie_mask] = 1.616365  # the giant tied value

    f3 = rng.integers(0, 10, n)  # low-cardinality -> level-based test

    if share_by_level is not None:
        share = np.array([share_by_level[int(v)] for v in f3])
    else:
        share = np.full(n, 0.85)
        if drift:
            q = pd.qcut(f0, 5, labels=False, duplicates="drop")
            share = 0.85 + drift * (np.asarray(q) - 2) / 2

    treatment = (rng.random(n) < share).astype(int)
    conversion = (rng.random(n) < 0.01).astype(int)
    visit = (rng.random(n) < 0.05).astype(int)

    df = pd.DataFrame(
        {"f0": f0, "f3": f3, "treatment": treatment,
         "conversion": conversion, "visit": visit}
    )
    conn = duckdb.connect()
    conn.register("raw_df", df)
    conn.execute("CREATE VIEW raw AS SELECT * FROM raw_df")
    return conn


def test_smd_zero_when_identical():
    assert standardized_mean_difference(5.0, 2.0, 5.0, 2.0) == pytest.approx(0.0)


def test_smd_sign_and_scale():
    assert standardized_mean_difference(6.0, 2.0, 4.0, 2.0) == pytest.approx(1.0)


def test_treatment_counts_shape():
    conn = make_conn()
    counts = treatment_counts(conn)
    # treatment is BOOLEAN in the real dataset; int-cast for arm checking
    assert sorted(counts["treatment"].astype(int)) == [0, 1]
    assert counts["n"].sum() == 30_000


def test_balance_on_randomized_data():
    conn = make_conn()
    balance = feature_balance(conn)
    assert balance["balanced"].all()


def test_feature_balance_ignores_absent_features():
    # only f0 exists -> f1..f11 silently skipped, f0 still tested
    conn = duckdb.connect()
    conn.execute(
        "CREATE VIEW raw AS SELECT 1.0::DOUBLE f0, 1::INT treatment "
        "UNION ALL SELECT 2.0, 0"
    )
    balance = feature_balance(conn)
    assert list(balance["feature"]) == ["f0"]


# --- treated-share independence -------------------------------------------


def test_uniform_assignment_stable_levels():
    share_by_level = [0.85] * 10
    conn = make_conn(share_by_level=share_by_level, n=60_000)
    r = share_independence(conn, "f3")
    assert r.method == "levels"
    assert r.share_range_pp < 3.0
    assert r.stable


def test_level_dependent_assignment_detected():
    share_by_level = [0.70, 0.70, 0.70, 0.70, 0.70, 0.95, 0.95, 0.95, 0.95, 0.95]
    conn = make_conn(share_by_level=share_by_level, n=60_000)
    r = share_independence(conn, "f3")
    assert r.method == "levels"
    assert r.share_range_pp == pytest.approx(25.0, abs=2.0)
    assert not r.stable
    assert r.p_value < 1e-10


def test_quantile_method_for_continuous_feature():
    conn = make_conn(n=60_000)
    r = share_independence(conn, "f0")
    assert r.method == "quantiles"
    assert r.groups == 5
    assert r.stable  # assignment independent of f0 by construction


def test_quantile_bins_deterministic_despite_ties():
    """A 26% tie block must land whole in one bin — same answer every run."""
    conn = make_conn(n=60_000, seed=7)
    bounds = quantile_boundaries(conn, "f0", 5)
    r1 = share_independence(conn, "f0")
    r2 = share_independence(conn, "f0")
    pd.testing.assert_frame_equal(r1.shares, r2.shares)
    # the giant tied value must not straddle bins: bin expression on the
    # tie value always returns the same group number
    expr = bin_expression("f0", bounds)
    g1 = conn.execute(f"SELECT {expr} FROM raw WHERE f0 = 1.616365 LIMIT 1").fetchone()[0]
    g2 = conn.execute(f"SELECT {expr} FROM raw WHERE f0 = 1.616365 ORDER BY 1 DESC LIMIT 1").fetchone()[0]
    assert g1 == g2


def test_bin_expression_covers_all_values():
    conn = make_conn(n=5_000)
    bounds = quantile_boundaries(conn, "f0", 5)
    expr = bin_expression("f0", bounds)
    n_null = conn.execute(
        f"SELECT count(*) FROM raw WHERE {expr} IS NULL"
    ).fetchone()[0]
    assert n_null == 0


def test_material_drift_flagged():
    share_by_level = [0.80, 0.80, 0.80, 0.90, 0.90, 0.90, 0.80, 0.90, 0.80, 0.90]
    conn = make_conn(share_by_level=share_by_level, n=60_000)
    r = share_independence(conn, "f3")
    assert not r.stable
    assert r.share_range_pp == pytest.approx(10.0, abs=1.5)


def test_drift_on_continuous_feature_detected():
    conn = make_conn(n=80_000, drift=0.10)
    r = share_independence(conn, "f0")
    assert r.method == "quantiles"
    assert r.groups == 5
    assert not r.stable
    # share tilts 0.75 -> 0.95 across the five f0 quantile bins
    assert r.share_range_pp == pytest.approx(20.0, abs=1.5)

def test_unknown_feature_rejected():
    conn = make_conn()
    with pytest.raises(ValueError):
        share_independence(conn, "not_a_feature")


def test_assignment_report_shape():
    conn = make_conn(n=30_000)
    rep = assignment_report(conn)
    # synthetic frame only has f0 and f3
    assert set(rep["feature"]) == {"f0", "f3"}
    assert (rep["method"].isin(["levels", "quantiles"])).all()
    assert rep["share_range_pp"].between(0, 100).all()


def test_cardinality_threshold_is_small():
    assert MAX_LEVEL_CARDINALITY == 1_000


# --- standardized ATE ------------------------------------------------------


def test_standardized_ate_removes_composition_bias():
    """Known structure: true additive effect 0.005, strong share+baseline
    confounding across strata -> crude is badly biased, standardized is not."""
    rng = np.random.default_rng(11)
    n = 200_000
    s = rng.integers(0, 2, n)                      # stratum 0/1, 50/50
    share = np.where(s == 0, 0.60, 0.95)           # assignment depends on s
    t = (rng.random(n) < share).astype(int)
    baseline = np.where(s == 0, 0.02, 0.12)
    p = baseline + t * 0.005                       # true ATE = 0.005
    conv = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({"f3": s, "treatment": t, "conversion": conv})
    conn = duckdb.connect()
    conn.register("raw_df", df)
    conn.execute("CREATE VIEW raw AS SELECT * FROM raw_df")

    res = standardized_ate(conn, "f3", "conversion")
    assert res.method == "levels"
    # control stratum s=1 holds only ~5% of 100K rows -> SE ~0.005, so the
    # tolerance reflects Monte Carlo noise, not estimator bias
    assert res.ate == pytest.approx(0.005, abs=0.0025)   # unbiased
    assert res.crude_ate > 0.04                          # heavily biased
    assert res.dropped_rows == 0
    assert res.covered_rows == n
    assert res.pct_of_crude == pytest.approx(res.ate / res.crude_ate)


def test_standardized_ate_outcome_validation():
    conn = make_conn(n=2_000)
    with pytest.raises(ValueError):
        standardized_ate(conn, "f0", "revenue")
