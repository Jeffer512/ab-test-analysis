"""Cross-check our two-proportion inference against statsmodels."""

import numpy as np
import pytest
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from abtest.inference import (
    holm_correction,
    mde_absolute,
    mde_relative,
    power_at,
    two_proportion_ztest,
)

CASES = [
    (100, 10_000, 80, 10_000),
    (3_671, 11_882_655, 406, 2_096_937),
    (1, 1_000, 0, 1_000),
    (470, 50_000, 455, 50_000),
    (25, 2_000, 30, 2_000),
]


@pytest.mark.parametrize("x_t,n_t,x_c,n_c", CASES)
def test_z_matches_statsmodels(x_t, n_t, x_c, n_c):
    res = two_proportion_ztest(x_t, n_t, x_c, n_c)
    z_ref, p_ref = proportions_ztest([x_t, x_c], [n_t, n_c])
    assert res.z == pytest.approx(z_ref, rel=1e-9)
    assert res.p_value == pytest.approx(p_ref, rel=1e-9)


@pytest.mark.parametrize("x_t,n_t,x_c,n_c", CASES)
def test_ci_matches_statsmodels(x_t, n_t, x_c, n_c):
    res = two_proportion_ztest(x_t, n_t, x_c, n_c)
    lo_ref, hi_ref = confint_proportions_2indep(
        x_t, n_t, x_c, n_c, method="wald", compare="diff"
    )
    assert res.ci_low == pytest.approx(lo_ref, rel=1e-7, abs=1e-12)
    assert res.ci_high == pytest.approx(hi_ref, rel=1e-7, abs=1e-12)


@pytest.mark.parametrize("x_t,n_t,x_c,n_c", CASES)
def test_rates_and_lift(x_t, n_t, x_c, n_c):
    res = two_proportion_ztest(x_t, n_t, x_c, n_c)
    assert res.rate_t == pytest.approx(x_t / n_t)
    assert res.rate_c == pytest.approx(x_c / n_c)
    assert res.diff == pytest.approx(res.rate_t - res.rate_c)
    if res.rate_c:
        assert res.rel_lift == pytest.approx(res.diff / res.rate_c)


def test_rejects_impossible_counts():
    with pytest.raises(ValueError):
        two_proportion_ztest(10, 5, 1, 10)
    with pytest.raises(ValueError):
        two_proportion_ztest(1, 0, 1, 10)


def test_direction_sign():
    better = two_proportion_ztest(200, 10_000, 100, 10_000)
    assert better.z > 0 and better.diff > 0 and better.significant
    worse = two_proportion_ztest(100, 10_000, 200, 10_000)
    assert worse.z < 0 and worse.p_value == pytest.approx(better.p_value, rel=1e-9)


# --- Holm correction -------------------------------------------------------


def test_holm_known_example():
    p = [0.01, 0.04, 0.03, 0.005]
    res = holm_correction(p, alpha=0.05)
    m = 4
    order = np.argsort(p)
    p_s = np.array(p)[order]
    expected_sorted = np.maximum.accumulate((m - np.arange(m)) * p_s)
    expected = np.empty(m)
    expected[order] = np.clip(expected_sorted, 0, 1)
    np.testing.assert_allclose(res.adjusted, expected)


def test_holm_rejections_ordered():
    res = holm_correction([0.001, 0.02, 0.5], alpha=0.05)
    assert res.reject[0]  # smallest p survives even strict thresholds
    assert not res.reject[2]


def test_holm_all_null_never_rejects():
    rng = np.random.default_rng(0)
    res = holm_correction(rng.uniform(0.2, 1.0, 20).tolist(), alpha=0.05)
    assert not res.reject.any()


# --- Power / MDE -----------------------------------------------------------


def test_power_at_zero_is_alpha():
    assert power_at(0.0, 0.05, 10_000, 10_000) == pytest.approx(0.05)


def test_power_monotone_in_effect():
    powers = [power_at(d, 0.05, 10_000, 10_000) for d in [0.001, 0.005, 0.02]]
    assert powers[0] < powers[1] < powers[2]


def test_mde_roundtrip():
    rate_c, n_t, n_c = 0.02, 50_000, 50_000
    mde = mde_absolute(rate_c, n_t, n_c, target_power=0.8)
    assert not np.isnan(mde)
    assert power_at(mde, rate_c, n_t, n_c) == pytest.approx(0.8, abs=1e-4)
    assert mde_relative(rate_c, n_t, n_c) == pytest.approx(mde / rate_c, rel=1e-9)


def test_mde_shrinks_with_sample_size():
    small = mde_absolute(0.02, 5_000, 5_000)
    large = mde_absolute(0.02, 500_000, 500_000)
    assert large < small
