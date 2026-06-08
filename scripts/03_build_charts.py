#!/usr/bin/env python3
"""Compute retention metrics and build presentation charts (Part 4)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis import (  # noqa: E402
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    CLAIMED_RETURN_RATE,
    ELECTION_COHORT_MONTH,
    EVER_RETURNED_MATURE_END,
    EVER_RETURNED_MATURE_START,
    MATURE_COHORT_START,
    EverReturnedSummary,
    acquisition_by_month,
    acquisition_mix,
    election_cohort_retention,
    load_cohort_by_category,
    load_cohort_retention,
    load_ever_returned,
    mature_average_retention,
    mature_ever_returned_summary,
    median_sequential_retention,
    pct_str,
    rate_at_month,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RAW_OVERALL = _ROOT / "data" / "raw" / "cohort_retention.csv"
RAW_CATEGORY = _ROOT / "data" / "raw" / "cohort_retention_by_category.csv"
RAW_EVER_RETURNED = _ROOT / "data" / "raw" / "ever_returned.csv"
PROCESSED_DIR = _ROOT / "data" / "processed"
FIGURES_DIR = _ROOT / "outputs" / "figures"
OUTPUTS_DIR = _ROOT / "outputs"

FIG_SIZE = (16, 9)
DPI = 100


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelsize": 13,
            "axes.titlesize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "font.family": "sans-serif",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
        }
    )


def _save_fig(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Wrote %s", path)
    return path


def _format_thousands(ax: plt.Axes, axis: str = "y") -> None:
    fmt = plt.FuncFormatter(lambda x, _p: f"{int(x):,}")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def chart_acquisition_by_month(overall: pd.DataFrame, sizes: pd.DataFrame) -> None:
    labels = sizes["cohort_label"].tolist()
    values = sizes["cohort_size"].tolist()
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(x, values, color="#4A6FA5", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Polymarket acquisition is violently event-driven")
    ax.set_xlabel("Cohort month (first trade)")
    ax.set_ylabel("New traders (cohort size)")
    _format_thousands(ax, "y")

    spikes = {
        "2024-10": "Election\n158k",
        "2026-03": "Mar 2026\n252k",
    }
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    for lbl, text in spikes.items():
        if lbl in label_to_idx:
            i = label_to_idx[lbl]
            ax.annotate(
                text,
                xy=(i, values[i]),
                xytext=(i, values[i] * 1.05),
                ha="center",
                fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1),
            )
    _save_fig(fig, "01_acquisition_by_month.png")


def chart_mature_average(mature: pd.DataFrame) -> str:
    """Return best category at month 6 for title tuning."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    months = mature.index.tolist()
    for cat in CATEGORY_ORDER:
        if cat not in mature.columns:
            continue
        y = 100.0 * mature[cat].values
        ax.plot(months, y, marker="o", linewidth=2.5, label=cat.title(), color=CATEGORY_COLORS[cat])

    # Mature mean-of-rates: politics trails sports/crypto by a few pp at M1–M6;
    # election-cohort chart (03) tells a different same-vintage story.
    ax.set_title(
        "Politics retention trails sports & crypto\n"
        "(mature cohorts, equal-weight average)"
    )
    ax.set_xlabel("Months since first trade")
    ax.set_ylabel("Retention rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xticks(range(0, 13))

    pol_m6 = rate_at_month(mature, 6, "politics")
    best_cat = max(
        CATEGORY_ORDER,
        key=lambda c: rate_at_month(mature, 6, c) or 0.0,
    )
    best_m6 = rate_at_month(mature, 6, best_cat)
    if pol_m6 is not None and best_m6 is not None:
        gap = 100.0 * (best_m6 - pol_m6)
        ax.annotate(
            f"Month 6: {best_cat.title()} {pct_str(best_m6)} vs\n"
            f"Politics {pct_str(pol_m6)} ({gap:.1f} pp gap)",
            xy=(6, 100.0 * pol_m6),
            xytext=(8.5, 100.0 * pol_m6 + 12),
            fontsize=11,
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1),
        )

    _save_fig(fig, "02_retention_by_category.png")
    return best_cat


def chart_election_cohort(election: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    months = election.index.tolist()
    for cat in CATEGORY_ORDER:
        if cat not in election.columns:
            continue
        y = 100.0 * election[cat].values
        ax.plot(
            months,
            y,
            marker="o",
            linewidth=2.5,
            label=cat.title(),
            color=CATEGORY_COLORS[cat],
        )
    ax.set_title(
        "Same acquisition wave, different first category: Oct 2024 cohort"
    )
    ax.set_xlabel("Months since first trade")
    ax.set_ylabel("Retention rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    _save_fig(fig, "03_election_cohort_retention.png")


def chart_75pct_myth(
    medians: pd.Series,
    ever: EverReturnedSummary,
) -> None:
    """Compare claimed ~75% to ever-returned (similar metric) vs sequential (stricter)."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    labels = [
        "Ever returned\n(mature pooled)",
        "Sequential\nmonth 1",
        "Sequential\nmonth 3",
        "Sequential\nmonth 6",
    ]
    values = [
        100.0 * ever.pooled_rate,
        100.0 * medians[1],
        100.0 * medians[3],
        100.0 * medians[6],
    ]
    colors = ["#2E7D32", "#4A6FA5", "#4A6FA5", "#4A6FA5"]
    x = list(range(len(labels)))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="none")

    ax.axhline(
        100.0 * CLAIMED_RETURN_RATE,
        color="#D1495B",
        linestyle="--",
        linewidth=2.5,
        label="Polymarket claimed ~75% return",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Retention rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Three different questions: ever returned vs. active in a specific later month"
    )
    ax.legend(loc="upper right")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val:.1f}%",
            ha="center",
            fontsize=11,
        )

    ax.text(
        0.02,
        0.02,
        f"Ever returned = traded in ≥2 calendar months (pooled {EVER_RETURNED_MATURE_START}"
        f"–{EVER_RETURNED_MATURE_END}).\n"
        "Sequential = active in exactly month N after cohort month (stricter; median across cohorts).",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    _save_fig(fig, "04_the_75pct_myth.png")


def chart_acquisition_mix(mix: pd.DataFrame) -> None:
    mix = mix.set_index("first_category").reindex(CATEGORY_ORDER).dropna()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    cats = mix.index.tolist()
    pcts = mix["pct_of_users"].values
    colors = [CATEGORY_COLORS[c] for c in cats]
    bars = ax.barh(cats, pcts, color=colors)
    ax.set_title("Who Polymarket acquires (first-trade category, all time)")
    ax.set_xlabel("Share of acquired users (%)")
    ax.set_xlim(0, 45)
    for bar, (_, row) in zip(bars, mix.iterrows()):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{row['pct_of_users']:.1f}% ({int(row['acquired_users']):,})",
            va="center",
            fontsize=12,
        )
    ax.invert_yaxis()
    _save_fig(fig, "05_acquisition_mix.png")


def build_plotly_dashboard(mature: pd.DataFrame, election: pd.DataFrame) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / "retention_dashboard.html"

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Mature average retention (2024-05 – 2025-05 cohorts)",
            f"Oct 2024 election cohort ({ELECTION_COHORT_MONTH})",
        ),
    )

    def add_lines(table: pd.DataFrame, col: int) -> None:
        for cat in CATEGORY_ORDER:
            if cat not in table.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=table.index,
                    y=100.0 * table[cat],
                    mode="lines+markers",
                    name=cat.title(),
                    line=dict(color=CATEGORY_COLORS[cat], width=3),
                    legendgroup=cat,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

    add_lines(mature.loc[:12], 1)
    add_lines(election, 2)
    fig.update_yaxes(title_text="Retention %", range=[0, 105])
    fig.update_xaxes(title_text="Months since first trade")
    fig.update_layout(
        title="Polymarket retention by first-trade category",
        template="plotly_white",
        height=520,
        width=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.5, xanchor="center"),
    )
    fig.write_html(path, include_plotlyjs="cdn")
    logging.info("Wrote %s", path)
    return path


def print_console_summary(
    mature: pd.DataFrame,
    election: pd.DataFrame,
    medians: pd.Series,
    ever: EverReturnedSummary,
) -> None:
    print("=" * 60)
    print("Mature average retention curve (%), months 0–12")
    print("Cohorts: 2024-05 .. 2025-05 | mean-of-rates per cohort")
    print("=" * 60)
    display = (100.0 * mature).round(1)
    print(display.to_string(float_format=lambda x: f"{x:.1f}"))

    print("\n" + "=" * 60)
    print("Mature curve — politics vs sports vs crypto")
    print("=" * 60)
    for m in (3, 6, 12):
        print(f"  Month {m:2d}:")
        for cat in ("politics", "sports", "crypto"):
            print(f"    {cat:8s} {pct_str(rate_at_month(mature, m, cat))}")

    print("\n" + "=" * 60)
    print(f"Election cohort {ELECTION_COHORT_MONTH} — retention by category")
    print("=" * 60)
    for m in (3, 6, 12):
        print(f"  Month {m:2d}:")
        for cat in CATEGORY_ORDER:
            print(f"    {cat:8s} {pct_str(rate_at_month(election, m, cat))}")

    print("\n" + "=" * 60)
    print("The 75% claim — ever returned vs sequential (different metrics)")
    print("=" * 60)
    print(
        f"  Ever returned, pooled ({ever.cohort_start}..{ever.cohort_end}, "
        f"n={ever.n_cohorts} cohorts): {pct_str(ever.pooled_rate)}"
    )
    print(f"  Ever returned, median per cohort: {pct_str(ever.median_rate)}")
    print(f"  Claimed benchmark: {pct_str(CLAIMED_RETURN_RATE)}")
    print("  Sequential retention (median across cohorts, stricter):")
    for m in (1, 3, 6):
        print(f"    Month {m}: {pct_str(medians[m])}")


def main() -> int:
    _configure_matplotlib()
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (LookupError, OSError):
            pass

    if not RAW_EVER_RETURNED.is_file():
        raise FileNotFoundError(
            f"Missing {RAW_EVER_RETURNED}. Run: python scripts/05_pull_ever_returned.py"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    overall = load_cohort_retention(RAW_OVERALL)
    by_cat = load_cohort_by_category(RAW_CATEGORY)
    ever_df = load_ever_returned(RAW_EVER_RETURNED)
    ever_summary = mature_ever_returned_summary(ever_df)

    mature_ever = ever_df[
        (ever_df["cohort_label"] >= EVER_RETURNED_MATURE_START)
        & (ever_df["cohort_label"] <= EVER_RETURNED_MATURE_END)
    ]
    mature_ever.to_csv(PROCESSED_DIR / "ever_returned_mature_cohorts.csv", index=False)
    pd.DataFrame(
        [
            {
                "pooled_rate": ever_summary.pooled_rate,
                "median_rate": ever_summary.median_rate,
                "cohort_start": ever_summary.cohort_start,
                "cohort_end": ever_summary.cohort_end,
                "total_cohort_size": ever_summary.total_cohort_size,
                "total_ever_returned": ever_summary.total_ever_returned,
                "n_cohorts": ever_summary.n_cohorts,
            }
        ]
    ).to_csv(PROCESSED_DIR / "ever_returned_summary.csv", index=False)

    mature = mature_average_retention(by_cat)
    mature.to_csv(PROCESSED_DIR / "mature_average_retention.csv")

    election = election_cohort_retention(by_cat)
    election.to_csv(PROCESSED_DIR / "election_cohort_retention.csv")

    medians = median_sequential_retention(overall)
    medians.to_frame("median_retention_rate").to_csv(
        PROCESSED_DIR / "median_sequential_retention.csv"
    )

    mix = acquisition_mix(by_cat)
    mix.to_csv(PROCESSED_DIR / "acquisition_mix.csv", index=False)

    monthly = acquisition_by_month(overall, start_label=MATURE_COHORT_START)
    monthly.to_csv(PROCESSED_DIR / "acquisition_by_month.csv", index=False)

    print_console_summary(mature, election, medians, ever_summary)

    chart_acquisition_by_month(overall, monthly)
    chart_mature_average(mature)
    chart_election_cohort(election)
    chart_75pct_myth(medians, ever_summary)
    chart_acquisition_mix(mix)
    build_plotly_dashboard(mature, election)

    print("\n" + "=" * 60)
    print("Outputs written to data/processed/ and outputs/figures/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
