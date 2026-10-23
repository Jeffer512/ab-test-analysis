from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass(frozen=True)
class TwoProportionResult:
    n_t: int
    n_c: int
    x_t: int
    x_c: int
    rate_t: float
    rate_c: float
    diff: float           # absolute risk difference (treatment - control)
    rel_lift: float       # diff / rate_c (NaN when control rate is 0)
    z: float
    p_value: float
    ci_low: float         # 95% CI on the absolute difference (unpooled Wald)
    ci_high: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def ci_rel_low(self) -> float:
        return self.ci_low / self.rate_c if self.rate_c else float("nan")

    @property
    def ci_rel_high(self) -> float:
        return self.ci_high / self.rate_c if self.rate_c else float("nan")

    def summary(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return (
            f"control {self.rate_c:.4%} vs treatment {self.rate_t:.4%} "
            f"-> diff {self.diff:+.4%} (rel {self.rel_lift:+.2%}), "
            f"95% CI [{self.ci_low:+.4%}, {self.ci_high:+.4%}], "
            f"z={self.z:.2f}, p={self.p_value:.3g} [{sig}]"
        )


def two_proportion_ztest(
    x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05
) -> TwoProportionResult:
    """Pooled two-proportion z-test with an unpooled Wald CI on the difference."""
    if n_t <= 0 or n_c <= 0:
        raise ValueError("Both groups need at least one observation")
    if not (0 <= x_t <= n_t and 0 <= x_c <= n_c):
        raise ValueError("Successes must be within [0, n] for each group")

    rate_t, rate_c = x_t / n_t, x_c / n_c
    diff = rate_t - rate_c
    rel = diff / rate_c if rate_c else float("nan")

    # Pooled SE for the hypothesis test (H0: p_t = p_c).
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    if se_pool == 0:
        raise ValueError("Zero variance: both groups are constant")
    z = diff / se_pool
    # sf, not 1-cdf: at z≈28 the subtraction cancels to a false p=0.0
    p_value = 2 * float(stats.norm.sf(abs(z)))

    # Unpooled SE for the confidence interval.
    se_diff = np.sqrt(rate_t * (1 - rate_t) / n_t + rate_c * (1 - rate_c) / n_c)
    crit = stats.norm.ppf(1 - alpha / 2)

    return TwoProportionResult(
        n_t=n_t,
        n_c=n_c,
        x_t=x_t,
        x_c=x_c,
        rate_t=rate_t,
        rate_c=rate_c,
        diff=diff,
        rel_lift=rel,
        z=z,
        p_value=p_value,
        ci_low=diff - crit * se_diff,
        ci_high=diff + crit * se_diff,
        alpha=alpha,
    )


@dataclass(frozen=True)
class HolmResult:
    reject: np.ndarray          # step-down decisions, original order
    adjusted: np.ndarray        # multiplicity-adjusted p-values, original order
    alpha: float


def holm_correction(pvals: list[float] | np.ndarray, alpha: float = 0.05) -> HolmResult:
    """Holm-Bonferroni step-down correction (controls family-wise error rate)."""
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pvals must be a non-empty 1-D sequence")
    m = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    factors = m - np.arange(m)
    adj_sorted = np.clip(np.maximum.accumulate(factors * p_sorted), 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj_sorted
    reject_sorted = adj_sorted <= alpha
    reject = np.empty(m, dtype=bool)
    reject[order] = reject_sorted
    return HolmResult(reject=reject, adjusted=adjusted, alpha=alpha)


def _power(delta: float, rate_c: float, n_t: int, n_c: int, alpha: float) -> float:
    """Power of the two-sided test given an absolute effect ``delta``."""
    p_t = rate_c + delta
    if p_t <= 0 or p_t >= 1:
        return 0.0
    if delta == 0:
        return alpha
    h = proportion_effectsize(p_t, rate_c)
    return float(NormalIndPower().power(
        effect_size=h, nobs1=n_t, alpha=alpha, ratio=n_c / n_t, alternative="two-sided"
    ))


def power_at(delta: float, rate_c: float, n_t: int, n_c: int, alpha: float = 0.05) -> float:
    """Power to detect an absolute effect ``delta`` with the observed sample sizes."""
    return _power(delta, rate_c, n_t, n_c, alpha)


def mde_absolute(
    rate_c: float, n_t: int, n_c: int, alpha: float = 0.05, target_power: float = 0.8
) -> float:
    """Smallest absolute effect detectable with ``target_power`` (Cohen's h based).

    Returns NaN if even a maximal effect cannot reach the target power.
    """
    from scipy.optimize import brentq

    lo, hi = 1e-9, min(1 - rate_c, 0.5) - 1e-9
    if _power(hi, rate_c, n_t, n_c, alpha) < target_power:
        return float("nan")
    return cast(float, brentq(
        lambda d: _power(d, rate_c, n_t, n_c, alpha) - target_power, lo, hi, xtol=1e-12
    ))


def mde_relative(
    rate_c: float, n_t: int, n_c: int, alpha: float = 0.05, target_power: float = 0.8
) -> float:
    """MDE as a fraction of the control rate (e.g. 0.05 = +5% relative)."""
    abs_mde = mde_absolute(rate_c, n_t, n_c, alpha, target_power)
    return abs_mde / rate_c if rate_c and not np.isnan(abs_mde) else float("nan")

