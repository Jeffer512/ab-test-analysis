import math
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from abtest.load import RAW_TABLE

FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ("conversion", "visit")

SMD_THRESHOLD = 0.1
MAX_LEVEL_CARDINALITY = 1_000   # at/below this, test treatment vs exact value
MIN_LEVEL_COUNT = 100           # levels rarer than this collapse into "rare"
MATERIAL_SHARE_DRIFT_PP = 2.0   # practical threshold for share instability


def standardized_mean_difference(mean_t: float, sd_t: float, mean_c: float, sd_c: float) -> float:
    """Cohen-style SMD using the unweighted average of the two SDs."""
    denom = math.sqrt((sd_t**2 + sd_c**2) / 2)
    return (mean_t - mean_c) / denom if denom else float("nan")


def _available_features(conn: duckdb.DuckDBPyConnection, rel: str) -> list[str]:
    """Which of the 12 experiment features actually exist in ``rel``."""
    cols = {r[0] for r in conn.execute(f"DESCRIBE {rel}").fetchall()}
    return [f for f in FEATURES if f in cols]


def treatment_counts(
    conn: duckdb.DuckDBPyConnection, rel: str = RAW_TABLE
) -> pd.DataFrame:
    """Row counts and outcomes per arm."""
    return conn.execute(
        f"""
        SELECT treatment,
               COUNT(*) AS n,
               SUM(CASE WHEN conversion THEN 1 ELSE 0 END) AS conversions,
               SUM(CASE WHEN visit THEN 1 ELSE 0 END)      AS visits
        FROM {rel}
        GROUP BY treatment
        ORDER BY treatment
        """
    ).df()


def feature_balance(
    conn: duckdb.DuckDBPyConnection, rel: str = RAW_TABLE
) -> pd.DataFrame:
    """Per-feature mean/SD by arm and the standardized mean difference.

    |SMD| is compared against ``SMD_THRESHOLD``: with ~14M rows, significance
    flags negligible differences, so a flat cutoff is the practical test.
    """
    features = _available_features(conn, rel)
    if not features:
        raise ValueError(f"No experiment features (f0..f11) found in {rel!r}")
    selects = [f"AVG({f}) AS m_{f}, stddev_samp({f}) AS s_{f}" for f in features]
    df = conn.execute(
        f"""
        SELECT treatment, COUNT(*) AS n,
               {', '.join(selects)}
        FROM {rel}
        GROUP BY treatment
        ORDER BY treatment
        """
    ).df()

    t_row = df[df["treatment"] == 1].iloc[0]
    c_row = df[df["treatment"] == 0].iloc[0]
    rows = []
    for f in features:
        smd = standardized_mean_difference(
            t_row[f"m_{f}"], t_row[f"s_{f}"], c_row[f"m_{f}"], c_row[f"s_{f}"]
        )
        rows.append(
            {
                "feature": f,
                "mean_treatment": t_row[f"m_{f}"],
                "mean_control": c_row[f"m_{f}"],
                "smd": smd,
                "balanced": abs(smd) < SMD_THRESHOLD,
            }
        )
    out = pd.DataFrame(rows)
    out["abs_smd"] = out["smd"].abs()
    return out.sort_values("abs_smd", ascending=False).reset_index(drop=True)


def cardinality(conn: duckdb.DuckDBPyConnection, feature: str, rel: str = RAW_TABLE) -> int:
    return conn.execute(f"SELECT COUNT(DISTINCT {feature}) FROM {rel}").fetchone()[0]


def quantile_boundaries(
    conn: duckdb.DuckDBPyConnection, feature: str, bins: int = 5, rel: str = RAW_TABLE
) -> list[float]:
    """Exact quantile cut-points for deterministic binning.

    The quantile values themselves bound half-open ``CASE`` bins, so each
    tied block lands wholly on one side — midpoint cut-points would split
    ties arbitrarily.
    """
    qs = [i / bins for i in range(1, bins)]
    row = conn.execute(
        f"SELECT quantile_cont({feature}, {qs}) FROM {rel}"
    ).fetchone()[0]
    return [float(x) for x in row]


def bin_expression(feature: str, boundaries: list[float]) -> str:
    """Deterministic SQL CASE binning: (-inf, b0], (b0, b1], ..., (b_last, inf)."""
    parts = ["CASE"]
    for i, b in enumerate(boundaries):
        parts.append(f" WHEN {feature} <= {b!r} THEN {i}")
    parts.append(f" ELSE {len(boundaries)} END")
    return " ".join(parts)


def _chi2_share(table: np.ndarray) -> tuple[float, float, int]:
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return float(chi2), float(p), int(dof)


@dataclass(frozen=True)
class IndependenceResult:
    feature: str
    method: str                  # "levels" | "quantiles"
    groups: int
    chi2: float
    p_value: float
    dof: int
    share_range_pp: float        # (max - min) treated share across groups, in pp
    weighted_share_sd_pp: float  # mass-weighted standard deviation of treated share, in pp
    drift_mass_pct: float        # % of all users in strata deviating > MATERIAL_SHARE_DRIFT_PP from pooled share
    shares: pd.DataFrame         # group, n_total, n_treatment, treatment_share

    @property
    def stable(self) -> bool:
        """Practical stability: conservative range heuristic."""
        return self.share_range_pp <= MATERIAL_SHARE_DRIFT_PP


def share_independence(
    conn: duckdb.DuckDBPyConnection,
    feature: str,
    bins: int = 5,
    rel: str = RAW_TABLE,
) -> IndependenceResult:
    """Test whether the treated share varies across a pre-treatment feature.

    Discrete features (≤ ``MAX_LEVEL_CARDINALITY``) use an exact-value
    chi-square; continuous ones use deterministic quantile bins — never
    ``NTILE(ORDER BY f)``, which nondeterministically splits through giant
    tied blocks. 

    Reports both range (sensitive to tail noise), weighted SD and mass-weighted
    dispersion (representative of actual population drift).
    """
    if feature not in _available_features(conn, rel):
        raise ValueError(f"Feature {feature!r} not found in {rel!r}")

    card = cardinality(conn, feature, rel)
    if card <= MAX_LEVEL_CARDINALITY:
        # Exact-value test; collapse vanishingly rare levels so expected
        # counts stay valid for the chi-square approximation.
        counts = conn.execute(
            f"SELECT {feature} AS v, treatment::INT AS treatment, COUNT(*) AS n "
            f"FROM {rel} GROUP BY 1, 2"
        ).df()
        totals = counts.groupby("v")["n"].sum()
        rare = totals[totals < MIN_LEVEL_COUNT].index
        if len(rare):
            counts["v"] = counts["v"].where(~counts["v"].isin(rare), "__rare__")
        table = (
            counts.pivot_table(index="v", columns="treatment", values="n",
                               aggfunc="sum", fill_value=0)
        )
        table = table[[c for c in sorted(table.columns)]]  # ensure both arms present
        chi2, p, dof = _chi2_share(table.to_numpy())
        shares = (
            counts.groupby(["v", "treatment"])["n"].sum().unstack(fill_value=0)
        )
        shares.columns = [f"n_{c}" for c in shares.columns]
        shares["n_total"] = shares.sum(axis=1)
        shares["n_treatment"] = shares["n_1"] if "n_1" in shares.columns else 0
        out = shares.reset_index().rename(columns={"v": "group"})
        out["treatment_share"] = out["n_treatment"] / out["n_total"]
        method = "levels"
        n_groups = len(out)
    else:
        bounds = quantile_boundaries(conn, feature, bins, rel)
        expr = bin_expression(feature, bounds)
        out = conn.execute(
            f"""
            WITH binned AS (SELECT {expr} AS grp, treatment FROM {rel})
            SELECT grp AS "group",
                   COUNT(*)             AS n_total,
                   SUM(treatment::INT)  AS n_treatment,
                   AVG(treatment::INT)::DOUBLE AS treatment_share
            FROM binned GROUP BY grp ORDER BY grp
            """
        ).df()
        table = np.vstack([
            out["n_treatment"].to_numpy(),
            (out["n_total"] - out["n_treatment"]).to_numpy(),
        ])
        chi2, p, dof = _chi2_share(table)
        method = "quantiles"
        n_groups = len(out)

    share_range_pp = float(
        (out["treatment_share"].max() - out["treatment_share"].min()) * 100
    )

    ntot = out["n_total"].to_numpy(dtype=float)
    ntreat = out["n_treatment"].to_numpy(dtype=float)
    q = out["treatment_share"].to_numpy(dtype=float)

    total_rows = ntot.sum()
    q_pooled = ntreat.sum() / total_rows if total_rows > 0 else 0.0

    # Mass-weighted SD of treated share (in percentage points)
    weighted_var = np.average((q - q_pooled) ** 2, weights=ntot) if total_rows > 0 else 0.0
    weighted_share_sd_pp = float(np.sqrt(weighted_var) * 100)

    # Share of users living in strata that deviate > MATERIAL_SHARE_DRIFT_PP from baseline
    drift_threshold = MATERIAL_SHARE_DRIFT_PP / 100.0
    drifting_mask = np.abs(q - q_pooled) > drift_threshold
    drift_mass_pct = float((ntot[drifting_mask].sum() / total_rows) * 100) if total_rows > 0 else 0.0

    return IndependenceResult(
        feature=feature,
        method=method,
        groups=n_groups,
        chi2=chi2,
        p_value=p,
        dof=dof,
        share_range_pp=share_range_pp,
        weighted_share_sd_pp=weighted_share_sd_pp,
        drift_mass_pct=drift_mass_pct,
        shares=out,
    )


def assignment_report(
    conn: duckdb.DuckDBPyConnection, bins: int = 5, rel: str = RAW_TABLE
) -> pd.DataFrame:
    """One row per feature with raw range, weighted SD, and drifting population mass."""
    rows = []
    for f in _available_features(conn, rel):
        r = share_independence(conn, f, bins=bins, rel=rel)
        rows.append(
            {
                "feature": f,
                "method": r.method,
                "groups": r.groups,
                "chi2": r.chi2,
                "dof": r.dof,
                "p_value": r.p_value,
                "share_range_pp": r.share_range_pp,
                "weighted_share_sd_pp": r.weighted_share_sd_pp,
                "drift_mass_pct": r.drift_mass_pct,
                "negligible_drift": r.stable,
            }
        )
    return pd.DataFrame(rows).sort_values("share_range_pp", ascending=False).reset_index(drop=True)

@dataclass(frozen=True)
class StandardizedATEResult:
    """Standardized average treatment effect for one feature."""

    feature: str
    outcome: str
    method: str                # "levels" | "quantiles"
    ate: float                 # standardized average treatment effect (treatment - control)
    crude_ate: float           # unpooled difference of pooled arm rates
    strata: int
    covered_rows: int
    dropped_rows: int          # strata lacking one arm (cannot contribute)

    @property
    def pct_of_crude(self) -> float:
        return self.ate / self.crude_ate if self.crude_ate else float("nan")


def standardized_ate(
    conn: duckdb.DuckDBPyConnection,
    feature: str,
    outcome: str = "conversion",
    rel: str = RAW_TABLE,
) -> StandardizedATEResult:
    """Risk difference standardized to the pooled stratum mix of ``feature``.

    Direct standardization averages the within-stratum effect over pooled
    stratum shares, removing the composition bias the crude arm-vs-arm
    difference picks up when treated share and outcomes co-vary across
    strata. Strata missing one arm are dropped (``dropped_rows``); exact
    values for low cardinality, deterministic quantile bins otherwise.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"Unknown outcome {outcome!r}")
    card = cardinality(conn, feature, rel)
    if card <= MAX_LEVEL_CARDINALITY:
        strata_expr, method = feature, "levels"
    else:
        bounds = quantile_boundaries(conn, feature, 5, rel)
        strata_expr, method = bin_expression(feature, bounds), "quantiles"

    row = conn.execute(
        f"""
        WITH per AS (
            SELECT {strata_expr} AS s, treatment,
                   COUNT(*) AS n, SUM({outcome}::INT) AS successes
            FROM {rel} GROUP BY 1, 2
        ),
        piv AS (
            SELECT s,
                   MAX(n)        FILTER (WHERE treatment) AS nt,
                   MAX(n)        FILTER (WHERE treatment = 0) AS nc,
                   MAX(successes) FILTER (WHERE treatment) AS xt,
                   MAX(successes) FILTER (WHERE treatment = 0) AS xc,
                   SUM(n) AS ntot
            FROM per GROUP BY s
        )
        SELECT SUM((1.0 * xt / nt - 1.0 * xc / nc) * ntot) / SUM(ntot) AS ate,
               SUM(ntot) AS covered,
               COUNT(*) FILTER (WHERE nt IS NULL OR nc IS NULL) AS dropped
        FROM piv
        WHERE nt IS NOT NULL AND nc IS NOT NULL
        """
    ).fetchone()

    crude = conn.execute(
        f"""
        SELECT AVG({outcome}::INT) FILTER (WHERE treatment)
             - AVG({outcome}::INT) FILTER (WHERE NOT treatment)
        FROM {rel}
        """
    ).fetchone()[0]

    return StandardizedATEResult(
        feature=feature,
        outcome=outcome,
        method=method,
        ate=float(row[0]),
        crude_ate=float(crude),
        strata=card if method == "levels" else 5,
        covered_rows=int(row[1]),
        dropped_rows=int(row[2]),
    )


C2ST_MATERIAL_GAP = 0.05        # practical threshold on auc - null_auc
C2ST_TRAIN_ROWS = 1_000_000     # default size of the balanced training sample
C2ST_NULL_ROWS = 100_000        # rows per label-permuted null fit


@dataclass(frozen=True)
class C2STResult:
    """C2ST: out-of-fold assignment-prediction AUC against a permuted null."""

    auc: float            # pooled out-of-fold AUC on the real assignment
    fold_aucs: tuple[float, ...]
    null_auc_mean: float  # label-permuted refits (smaller sample)
    null_auc_sd: float
    n_rows: int
    features: tuple[str, ...]
    random_state: int
    model: str

    @property
    def auc_gap(self) -> float:
        return self.auc - self.null_auc_mean

    @property
    def material(self) -> bool:
        """Gap large enough to act on — independent of n's p-value pressure."""
        return self.auc_gap >= C2ST_MATERIAL_GAP


def _balanced_sample(
    conn: duckdb.DuckDBPyConnection, rows: int, seed: int, rel: str
) -> pd.DataFrame:
    """Reservoir sample ``rows`` rows, half from each arm.

    Class-balancing decouples the AUC from the base treated share.
    """
    features = _available_features(conn, rel)
    cols = ", ".join(features)
    half = max(rows // 2, 1)
    parts = []
    for arm in (0, 1):
        parts.append(
            f"SELECT {cols}, {arm} AS treatment "
            f"FROM (SELECT {cols} FROM {rel} WHERE treatment = {arm}) "
            f"USING SAMPLE {half} ROWS (reservoir, {seed + arm})"
        )
    return conn.execute(" UNION ALL ".join(parts)).df()


def c2st_validity(
    conn: duckdb.DuckDBPyConnection,
    features: list[str] | None = None,
    n_rows: int = C2ST_TRAIN_ROWS,
    folds: int = 5,
    null_runs: int = 5,
    null_rows: int = C2ST_NULL_ROWS,
    rel: str = RAW_TABLE,
    random_state: int = 42,
) -> C2STResult:
    """Can a classifier predict ``treatment`` from the pre-treatment features?

    C2ST: histogram-gradient-boosted trees are cross-validated on a
    class-balanced sample, and the pooled out-of-fold AUC is compared to a
    label-permuted null.
    """
    available = _available_features(conn, rel)
    fs = list(features) if features is not None else available
    unknown = [f for f in fs if f not in available]
    if unknown:
        raise ValueError(f"Feature(s) {unknown} not found in {rel!r}")
    if not fs:
        raise ValueError(f"No experiment features (f0..f11) found in {rel!r}")
    if folds < 2:
        raise ValueError("Need at least 2 folds for out-of-fold AUC")
    if null_runs < 1:
        raise ValueError("Need at least one null run to establish the null")

    df = _balanced_sample(conn, n_rows, random_state, rel)
    X = df[fs].to_numpy()
    y = df["treatment"].to_numpy()

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    proba = cross_val_predict(
        HistGradientBoostingClassifier(random_state=random_state),
        X, y, cv=cv, method="predict_proba",
    )[:, 1]
    auc = float(roc_auc_score(y, proba))
    fold_aucs = tuple(
        float(roc_auc_score(y[test], proba[test])) for _, test in cv.split(X, y)
    )

    rng = np.random.default_rng(random_state)
    null_auc = np.empty(null_runs, dtype=float)
    for i in range(null_runs):
        null_df = _balanced_sample(conn, null_rows, random_state + 10_000 + i, rel)
        Xn = null_df[fs].to_numpy()
        yn = rng.permutation(null_df["treatment"].to_numpy())
        Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
            Xn, yn, test_size=0.25, shuffle=True, random_state=random_state + i
        )
        null_model = HistGradientBoostingClassifier(random_state=random_state)
        null_model.fit(Xn_tr, yn_tr)
        null_auc[i] = roc_auc_score(
            yn_te, null_model.predict_proba(Xn_te)[:, 1]
        )

    return C2STResult(
        auc=auc,
        fold_aucs=fold_aucs,
        null_auc_mean=float(null_auc.mean()),
        null_auc_sd=float(null_auc.std()),
        n_rows=len(y),
        features=tuple(fs),
        random_state=random_state,
        model="HistGradientBoostingClassifier",
    )