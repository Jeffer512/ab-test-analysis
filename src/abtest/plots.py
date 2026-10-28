"""Figures for the A/B test report. Every function saves a PNG and returns it."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from abtest.inference import TwoProportionResult, power_at

sns.set_theme(style="whitegrid", context="talk")
FIGSIZE_WIDE = (12, 6)
FIGSIZE_TALL = (10, 7)

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY = "#7f7f7f"


def _save(fig: Figure, path: str | Path | None) -> Figure:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_rates_by_arm(counts: pd.DataFrame, path: str | Path | None = None) -> Figure:
    """Conversion and visit rates per arm."""
    df = counts.copy()
    df["treatment"] = df["treatment"].astype(int)
    df["conv_rate"] = df["conversions"] / df["n"]
    df["visit_rate"] = df["visits"] / df["n"]
    labels = {0: "Control", 1: "Treatment"}
    df["arm"] = df["treatment"].map(labels)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    for ax, col in zip(
        axes,
        ["visit_rate", "conv_rate"],
    ):
        rates = df[col].to_numpy()
        bars = ax.bar(df["arm"], rates * 100, color=[GRAY, BLUE], alpha=0.9)
        for b, r in zip(bars, rates):
            ax.annotate(f"{r:.4%}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{col.replace('_rate', '').capitalize()} rate by arm")
        ax.set_ylabel("rate (%)")
    fig.suptitle("Outcome rates by assignment", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def plot_smd(balance: pd.DataFrame, path: str | Path | None = None) -> Figure:
    """Horizontal bar chart of standardized mean differences per feature."""
    df = balance.sort_values("smd")
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    colors = [GREEN if b else RED for b in df["balanced"]]
    ax.barh(df["feature"], df["smd"], color=colors, alpha=0.9)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(0.1, color=GRAY, ls="--", lw=1)
    ax.axvline(-0.1, color=GRAY, ls="--", lw=1)
    ax.set_xlabel("standardized mean difference (treatment - control)")
    ax.set_title("Pre-treatment covariate balance (|SMD| < 0.1 = balanced)")
    fig.tight_layout()
    return _save(fig, path)


def plot_lift(
    results: dict[str, TwoProportionResult],
    path: str | Path | None = None,
) -> Figure:
    """Forest plot of the overall treatment effect per metric (95% CI)."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    names = list(results)
    ys = np.arange(len(names))[::-1]
    los, his = [], []
    for y, name in zip(ys, names):
        r = results[name]
        x = r.rel_lift * 100
        lo, hi = r.ci_rel_low * 100, r.ci_rel_high * 100
        los.append(lo)
        his.append(hi)
        ax.errorbar(
            x, y, xerr=[[x - lo], [hi - x]], fmt="o", color=BLUE, capsize=6, ms=9
        )
        p_txt = f"{r.p_value:.3g}" if r.p_value > 0 else "< 1e-300"
        ax.annotate(
            f"{r.rel_lift:+.2%} ({p_txt})",
            (x, y),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    span = max(his) - min(los)
    ax.set_xlim(min(los) - 0.10 * span, max(his) + 0.10 * span)
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_yticks(ys, names)
    ax.set_xlabel("relative lift vs control (%) — 95% CI")
    ax.set_title("Treatment effect by metric")
    fig.tight_layout()
    return _save(fig, path)


def plot_segment_lift(seg_df: pd.DataFrame, path: str | Path | None = None) -> Figure:
    """Forest plot of treatment effect across feature quantile bins."""
    df = seg_df.copy()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ys = np.arange(len(df))[::-1]
    for (_, row), y in zip(df.iterrows(), ys):
        x = row["rel_lift"] * 100
        lo = row["ci_low"] / row["rate_c"] * 100
        hi = row["ci_high"] / row["rate_c"] * 100
        color = BLUE if row["significant"] else GRAY
        ax.errorbar(
            x, y, xerr=[[x - lo], [hi - x]], fmt="o", color=color, capsize=6, ms=8
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ys, [f"Q{s}" for s in df["segment"]])
    ax.set_xlabel(f"relative lift in conversion (%) — 95% CI by {df['feature'].iloc[0]} quartile")
    ax.set_title(
        "Segment heterogeneity (blue = Holm-significant, gray = not)"
    )
    ax.margins(x=0.2)
    fig.tight_layout()
    return _save(fig, path)


def plot_power_curve(
    rate_c: float,
    n_t: int,
    n_c: int,
    alpha: float = 0.05,
    path: str | Path | None = None,
) -> Figure:
    """Power to detect a relative lift, given the observed sample sizes."""
    rels = np.linspace(0.001, 0.30, 300)
    powers = [power_at(r * rate_c, rate_c, n_t, n_c, alpha) for r in rels]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(rels * 100, powers, color=BLUE, lw=2)
    ax.axhline(0.8, color=RED, ls="--", lw=1, label="80% power")
    ax.set_xlabel("true relative lift we want to detect (%)")
    ax.set_ylabel("power")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"Power curve at observed n (control rate {rate_c:.3%}, "
        f"n={n_t + n_c:,})"
    )
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)
