"""Plots RQ1: each category's between-document variance (rho, % of the maximum
possible at that rate) by source, as grouped bars.

Same layout as RQ0. Pick the figure set with a flag (default: one figure per
domain):

    (default)    one figure per domain with two panels, sentence type and
                 sentence structure
    --grouped    one figure with the domains as rows, on a shared y-axis so
                 bar heights compare across domains

Error bars are the 95% resampling intervals from rq1_variance.csv, which are
asymmetric. Reads that CSV only.

Inputs:
    results/rq1/rq1_variance.csv

Outputs (.png, or .pdf with --pdf):
    results/rq1/rq1_variance_{essay,reuter,wp}.png   (default)
    results/rq1/rq1_variance_grouped.png             (--grouped)

Usage (from the project root):
    python src/analysis/rq1/plot_rq1_variance.py                  # per-domain PNGs
    python src/analysis/rq1/plot_rq1_variance.py --grouped        # grouped PNG
    python src/analysis/rq1/plot_rq1_variance.py --grouped --pdf  # grouped PDF
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.shared.plotting import (
    BAR_WIDTH,
    COLOURS,
    DOMAIN_TITLES,
    SCHEME_TITLES,
    SCHEMES,
    SOURCE_LABELS,
    SOURCES,
    doc_count_footnote,
    save_figure,
)

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
VARIANCE_INPUT = PROJECT_ROOT / "results" / "rq1" / "rq1_variance.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "rq1"


def _plot_panel(ax: plt.Axes, rates: pd.DataFrame, scheme: str) -> None:
    """Draws one scheme's grouped bars with 95% error bars.

    Each category gets an x position, and the three sources are drawn
    side by side around it. Rows are selected by category in SCHEMES
    order, so the bars follow a fixed order whatever the CSV's row order,
    and a missing category raises a KeyError rather than drawing fewer
    bars.

    Args:
        ax (plt.Axes): The panel to draw on.
        rates (pd.DataFrame): One domain's rows of rq1_variance.csv.
        scheme (str): "sentence_type" or "sentence_structure".
    """
    sub = rates[rates["scheme"] == scheme]  # OTHER exists in both schemes
    categories = SCHEMES[scheme]
    x = np.arange(len(categories))

    for i, source in enumerate(SOURCES):
        rows = sub[sub["source"] == source].set_index("category").loc[categories]
        ax.bar(
            x + (i - 1) * BAR_WIDTH,
            100 * rows["rho"],
            BAR_WIDTH,
            yerr=[
                100 * (rows["rho"] - rows["rho_ci_low"]),
                100 * (rows["rho_ci_high"] - rows["rho"]),
            ],
            capsize=3,
            color=COLOURS[source],
            label=SOURCE_LABELS[source],
        )

    ax.set_xticks(x, [c.replace("-", "-\n") for c in categories])
    ax.set_title(SCHEME_TITLES[scheme])
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.axhline(0, color="black", linewidth=0.8)


def _plot_domain(domain: str, rates: pd.DataFrame) -> plt.Figure:
    """Builds one domain's two-panel figure.

    Panel widths follow their category counts (5:3), and the panels share
    a y-axis so bar heights compare across them.

    Args:
        domain (str): e.g. "wp".
        rates (pd.DataFrame): All 72 rows of rq1_variance.csv.

    Returns:
        plt.Figure: The figure, not yet saved.
    """
    d = rates[rates["domain"] == domain]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        gridspec_kw={"width_ratios": [5, 3]},
        sharey=True,
    )
    for ax, scheme in zip(axes, SCHEMES):
        _plot_panel(ax, d, scheme)

    axes[0].set_ylabel("Between-document variance\n(% of maximum)")
    axes[0].legend(frameon=False)

    n = d.groupby("source")["n_docs"].first()
    fig.suptitle(
        f"{DOMAIN_TITLES[domain]}: between-document variance (documents: human "
        f"{n['human']}, GPT {n['gpt']}, Claude {n['claude']})"
    )
    fig.tight_layout()
    return fig


def _plot_grouped(rates: pd.DataFrame) -> plt.Figure:
    """Builds the single figure: one row of panels per domain.

    Every panel shares one y-axis, so a bar's height means the same thing in
    every row (the per-domain figures each have their own scale). Scheme
    titles appear on the top row only, and one legend sits below the grid.

    Args:
        rates (pd.DataFrame): All 72 rows of rq1_variance.csv.

    Returns:
        plt.Figure: The figure, not yet saved.
    """
    fig, axes = plt.subplots(
        len(DOMAIN_TITLES),
        2,
        figsize=(11, 10),
        gridspec_kw={"width_ratios": [5, 3]},
        sharey=True,
    )
    for row, domain in enumerate(DOMAIN_TITLES):
        d = rates[rates["domain"] == domain]
        for col, scheme in enumerate(SCHEMES):
            _plot_panel(axes[row, col], d, scheme)
            if row > 0:
                axes[row, col].set_title("")
        axes[row, 0].set_ylabel(f"{DOMAIN_TITLES[domain]}\nvariance (% of max)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=len(SOURCES),
        frameon=False,
    )
    fig.text(
        0.5, 0.005, doc_count_footnote(rates), ha="center", fontsize=8, color="0.3"
    )
    fig.suptitle("RQ1: between-document variance by domain and source")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def main(pdf: bool = False, grouped: bool = False):
    """Reads rq1_variance.csv and saves the requested figures.

    Saves one figure per domain by default, or the single grouped figure
    with grouped=True. Saves PNGs by default; with pdf=True it saves vector
    PDFs instead (not both), so the results folder only ever holds one
    format.

    Args:
        pdf (bool): Save PDFs instead of PNGs.
        grouped (bool): Save the single grouped figure (domains as rows)
            instead of one per domain.
    """
    rates = pd.read_csv(VARIANCE_INPUT)

    figures: dict[str, plt.Figure] = {}
    if grouped:
        figures["grouped"] = _plot_grouped(rates)
    else:
        for domain in DOMAIN_TITLES:
            figures[domain] = _plot_domain(domain, rates)

    for name, fig in figures.items():
        save_figure(fig, OUTPUT_DIR / f"rq1_variance_{name}", pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RQ1 between-document variance.")
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="save vector PDFs instead of PNGs",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="one figure with domains as rows, instead of one per domain",
    )
    args = parser.parse_args()
    main(pdf=args.pdf, grouped=args.grouped)
