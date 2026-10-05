"""Plots RQ0: each category's mean rate by source, in three possible layouts.

Pick the layout with at most one flag (default: one grouped-bar figure per
domain):

    (default)    one figure per domain with two panels, sentence type
                 (5 categories) and sentence structure (3), since each scheme
                 sums to 100% on its own
    --grouped    one grouped-bar figure with the domains as rows, on a shared
                 y-axis so bar heights compare across domains
    --stacked    one figure with a horizontal bar per domain x source, split
                 into each scheme's categories (no error bars; the 95%
                 intervals stay in the CSV)

--grouped and --stacked are mutually exclusive. Grouped-bar error bars are the
95% intervals from rq0_rates.csv. Reads that CSV only, so charts can be
restyled without recomputing anything.

Inputs:
    results/rq0/rq0_rates.csv

Outputs (.png, or .pdf with --pdf):
    results/rq0/rq0_rates_{essay,reuter,wp}.png   (default)
    results/rq0/rq0_rates_grouped.png             (--grouped)
    results/rq0/rq0_rates_stacked.png             (--stacked)

Usage (from the project root):
    python src/analysis/rq0/plot_rq0.py                     # per-domain PNGs
    python src/analysis/rq0/plot_rq0.py --grouped           # grouped-bar PNG
    python src/analysis/rq0/plot_rq0.py --stacked           # stacked-bar PNG
    python src/analysis/rq0/plot_rq0.py --stacked --pdf     # stacked-bar PDF
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
RATES_INPUT = PROJECT_ROOT / "results" / "rq0" / "rq0_rates.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "rq0"

SOURCES = ["human", "gpt", "claude"]
SOURCE_LABELS = {"human": "Human", "gpt": "GPT", "claude": "Claude"}
# Okabe-Ito palette: distinguishable for colour-blind readers; same in every figure
COLOURS = {"human": "#0072B2", "gpt": "#E69F00", "claude": "#009E73"}

DOMAIN_TITLES = {
    "essay": "Student essays",
    "reuter": "Reuters news",
    "wp": "WritingPrompts",
}

# fixed category order per panel -- labels must match rq0_rates.csv exactly
SCHEMES: dict[str, list[str]] = {
    "sentence_type": ["SIMPLE", "COMPLEX", "COMPOUND", "COMPLEX-COMPOUND", "OTHER"],
    "sentence_structure": ["LOOSE", "PERIODIC", "OTHER"],
}
SCHEME_TITLES = {
    "sentence_type": "Sentence type",
    "sentence_structure": "Sentence structure",
}

# stacked-bar segment colours, one per category in SCHEMES order; OTHER is grey
STACK_COLOURS = {
    "sentence_type": ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#BBBBBB"],
    "sentence_structure": ["#009E73", "#CC79A7", "#BBBBBB"],
}
MIN_LABEL_PCT = 6.0  # stacked segments narrower than this get no number

BAR_WIDTH = 0.26


def _plot_panel(ax: plt.Axes, rates: pd.DataFrame, scheme: str) -> None:
    """Draws one scheme's grouped bars with 95% error bars.

    Each category gets an x position, and the three sources are drawn
    side by side around it. Rows are selected by category in SCHEMES
    order, so the bars follow a fixed order whatever the CSV's row order,
    and a missing category raises a KeyError rather than drawing fewer
    bars.

    Args:
        ax (plt.Axes): The panel to draw on.
        rates (pd.DataFrame): One domain's rows of rq0_rates.csv.
        scheme (str): "sentence_type" or "sentence_structure".
    """
    sub = rates[rates["scheme"] == scheme]  # OTHER exists in both schemes
    categories = SCHEMES[scheme]
    x = np.arange(len(categories))

    for i, source in enumerate(SOURCES):
        rows = sub[sub["source"] == source].set_index("category").loc[categories]
        ax.bar(
            x + (i - 1) * BAR_WIDTH,
            100 * rows["mean_rate"],
            BAR_WIDTH,
            yerr=100 * (rows["ci_high"] - rows["mean_rate"]),
            capsize=3,
            color=COLOURS[source],
            label=SOURCE_LABELS[source],
        )

    ax.set_xticks(x, [c.replace("-", "-\n") for c in categories])
    ax.set_title(SCHEME_TITLES[scheme])
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def _doc_footnote(rates: pd.DataFrame) -> str:
    """Builds the per-domain document-count footnote for multi-domain figures.

    Args:
        rates (pd.DataFrame): All 72 rows of rq0_rates.csv.

    Returns:
        str: e.g. "Documents (human / GPT / Claude): Student essays
            1000/998/1000; ...".
    """
    n = rates.groupby(["domain", "source"])["n_docs"].first()
    parts = [
        f"{title} {n[(domain, 'human')]}/{n[(domain, 'gpt')]}/{n[(domain, 'claude')]}"
        for domain, title in DOMAIN_TITLES.items()
    ]
    return "Documents (human / GPT / Claude): " + "; ".join(parts)


def _plot_domain(domain: str, rates: pd.DataFrame) -> plt.Figure:
    """Builds one domain's two-panel figure.

    Panel widths follow their category counts (5:3), and the panels share
    a y-axis so bar heights compare across them.

    Args:
        domain (str): e.g. "wp".
        rates (pd.DataFrame): All 72 rows of rq0_rates.csv.

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

    axes[0].set_ylabel("% of sentences")
    axes[0].legend(frameon=False)

    n = d.groupby("source")["n_docs"].first()
    fig.suptitle(
        f"{DOMAIN_TITLES[domain]} (documents: human {n['human']}, "
        f"GPT {n['gpt']}, Claude {n['claude']})"
    )
    fig.tight_layout()
    return fig


def _plot_grouped(rates: pd.DataFrame) -> plt.Figure:
    """Builds the single grouped-bar figure: one row per domain.

    Every panel shares one y-axis, so a bar's height means the same thing in
    every row. Scheme titles appear on the top row only, and one legend
    sits below the grid.

    Args:
        rates (pd.DataFrame): All 72 rows of rq0_rates.csv.

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
        axes[row, 0].set_ylabel(f"{DOMAIN_TITLES[domain]}\n% of sentences")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=len(SOURCES),
        frameon=False,
    )
    fig.text(0.5, 0.005, _doc_footnote(rates), ha="center", fontsize=8, color="0.3")
    fig.suptitle("RQ0: mean category rate by domain and source")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def _plot_stacked(rates: pd.DataFrame) -> plt.Figure:
    """Builds the stacked horizontal composition figure.

    One bar per domain x source (9 per panel), split into the scheme's
    categories in SCHEMES order. Each bar must sum to 100%, which is
    asserted rather than assumed, since a bar that stops short would
    silently misread as a smaller share of sentences.

    Args:
        rates (pd.DataFrame): All 72 rows of rq0_rates.csv.

    Returns:
        plt.Figure: The figure, not yet saved.

    Raises:
        AssertionError: If any bar's category rates do not sum to 100%.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, scheme in zip(axes, SCHEMES):
        categories = SCHEMES[scheme]
        colours = STACK_COLOURS[scheme]
        y = 0.0
        ticks, tick_labels, domain_mid = [], [], {}

        for domain in DOMAIN_TITLES:
            first_y = y
            for source in SOURCES:
                rows = (
                    rates[
                        (rates["domain"] == domain)
                        & (rates["scheme"] == scheme)
                        & (rates["source"] == source)
                    ]
                    .set_index("category")
                    .loc[categories]
                )
                values = 100 * rows["mean_rate"].to_numpy()
                assert abs(values.sum() - 100) < 0.05, (
                    f"{domain}/{source}/{scheme} sums to {values.sum():.2f}%"
                )

                left = 0.0
                for value, colour in zip(values, colours):
                    ax.barh(y, value, left=left, color=colour, edgecolor="white")
                    if value >= MIN_LABEL_PCT:
                        r, g, b = to_rgb(colour)
                        dark_text = (0.299 * r + 0.587 * g + 0.114 * b) > 0.6
                        ax.text(
                            left + value / 2,
                            y,
                            f"{value:.0f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="black" if dark_text else "white",
                        )
                    left += value

                ticks.append(y)
                tick_labels.append(SOURCE_LABELS[source])
                y += 1.0
            domain_mid[domain] = (first_y + y - 1.0) / 2
            y += 0.6  # gap between domains

        ax.set_yticks(ticks, tick_labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of sentences")
        ax.set_title(SCHEME_TITLES[scheme])
        for domain, mid in domain_mid.items():
            ax.text(
                -0.2,
                mid,
                DOMAIN_TITLES[domain],
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontweight="bold",
                fontsize=9,
            )
        ax.legend(
            handles=[Patch(color=c, label=k) for k, c in zip(categories, colours)],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=len(categories),
            frameon=False,
            fontsize=8,
        )

    fig.suptitle("RQ0: composition of sentences by domain and source")
    fig.text(0.5, 0.0, _doc_footnote(rates), ha="center", fontsize=8, color="0.3")
    fig.tight_layout()
    return fig


def main(pdf: bool = False, grouped: bool = False, stacked: bool = False):
    """Reads rq0_rates.csv and saves the requested figures.

    With neither grouped nor stacked, saves one grouped-bar figure per
    domain. Saves PNGs by default; with pdf=True it saves vector PDFs
    instead (not both), so the results folder only ever holds one format.

    Args:
        pdf (bool): Save PDFs instead of PNGs.
        grouped (bool): Save the single grouped-bar figure (domains as rows).
        stacked (bool): Save the stacked horizontal composition figure.

    Raises:
        ValueError: If both grouped and stacked are set.
    """
    if grouped and stacked:
        raise ValueError("grouped and stacked are mutually exclusive")

    ext = "pdf" if pdf else "png"
    rates = pd.read_csv(RATES_INPUT)

    figures: dict[str, plt.Figure] = {}
    if grouped:
        figures["grouped"] = _plot_grouped(rates)
    elif stacked:
        figures["stacked"] = _plot_stacked(rates)
    else:
        for domain in DOMAIN_TITLES:
            figures[domain] = _plot_domain(domain, rates)

    for name, fig in figures.items():
        fig.savefig(
            OUTPUT_DIR / f"rq0_rates_{name}.{ext}",
            dpi=200,  # ignored for PDF (vector)
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RQ0 category rates.")
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="save vector PDFs instead of PNGs",
    )
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument(
        "--grouped",
        action="store_true",
        help="one grouped-bar figure with domains as rows, instead of one per domain",
    )
    layout.add_argument(
        "--stacked",
        action="store_true",
        help="one stacked horizontal composition figure",
    )
    args = parser.parse_args()
    main(pdf=args.pdf, grouped=args.grouped, stacked=args.stacked)
