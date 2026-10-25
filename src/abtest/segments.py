import duckdb
import pandas as pd

from abtest.inference import holm_correction, two_proportion_ztest
from abtest.load import RAW_TABLE
from abtest.validity import FEATURES, OUTCOMES, bin_expression, quantile_boundaries


def segment_effects(
    conn: duckdb.DuckDBPyConnection,
    feature: str = "f0",
    outcome: str = "conversion",
    n_bins: int = 4,
    rel: str = RAW_TABLE,
    holm: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Treatment effect per quantile-bin of ``feature``.

    Exact quantile boundaries applied via ``CASE`` — not ``NTILE(ORDER BY f)``,
    which nondeterministically splits through giant tied blocks.

    Returns one row per bin with arm sizes, outcome rates, the absolute
    difference with its 95% CI, and (optionally) Holm-adjusted p-values
    across bins
    """
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature {feature!r}")
    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown outcome {outcome!r}")

    try:
        cols = {r[0] for r in conn.execute(f"DESCRIBE {rel}").fetchall()}
    except Exception as e:
        raise ValueError(f"Relation {rel!r} not found") from e
    if feature not in cols:
        raise ValueError(f"Feature {feature!r} not found in {rel!r}")

    bounds = quantile_boundaries(conn, feature, n_bins, rel)
    expr = bin_expression(feature, bounds)

    counts = conn.execute(
        f"""
        SELECT {expr} as segment,
                treatment, 
                COUNT(*) as n, 
                SUM({outcome}) as successes 
        FROM {rel}
        GROUP BY segment, treatment
        ORDER BY segment, treatment
        """
    ).df()
    rows = []
    for segment, grp in counts.groupby("segment"):
        g_t = grp[grp["treatment"]]
        g_c = grp[~grp["treatment"]]
        test_result = two_proportion_ztest(
            g_t["successes"].iloc[0],
            g_t["n"].iloc[0],
            g_c["successes"].iloc[0],
            g_c["n"].iloc[0],
            alpha=alpha
        )

        rows.append(
            {
                "segment": int(segment),
                "feature": feature,
                "outcome": outcome,
                "n_t": test_result.n_t,
                "n_c": test_result.n_c,
                "rate_t": test_result.rate_t,
                "rate_c": test_result.rate_c,
                "diff": test_result.diff,
                "rel_lift": test_result.rel_lift,
                "p_value": test_result.p_value,
                "ci_low": test_result.ci_low,
                "ci_high": test_result.ci_high,
            }
        )
        
    df = pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)

    if holm and len(df) > 1:
        adj = holm_correction(df["p_value"].tolist(), alpha=alpha)
        df["p_adj"] = adj.adjusted
        df["significant"] = adj.reject
    else:
        df["p_adj"] = df["p_value"]
        df["significant"] = df["p_value"] < 0.05
    return df
