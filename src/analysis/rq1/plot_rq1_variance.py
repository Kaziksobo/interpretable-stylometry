"""Plots RQ1 Track 1: per domain, grouped bars of each category's between-document
variance (rho, % of the maximum possible at that rate) by source.

Same layout as RQ0: one figure per domain, two panels (sentence type, sentence
structure). Error bars are the 95% resampling intervals from rq1_variance.csv,
which are asymmetric. Reads that CSV only.

Inputs:
    results/rq1/rq1_variance.csv

Outputs:
    results/rq1/rq1_variance_{essay,reuter,wp}.png / .pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
VARIANCE_INPUT = PROJECT_ROOT / "results" / "rq1" / "rq1_variance.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "rq1"

SOURCES = ["human", "gpt", "claude"]
SOURCE_LABELS = {"human": "Human", "gpt": "GPT", "claude": "Claude"}
# Okabe-Ito palette: distinguishable for colour-blind readers; same in every figure
COLOURS = {"human": "#0072B2", "gpt": "#E69F00", "claude": "#009E73"}

DOMAIN_TITLES = {
    "essay": "Student essays",
    "reuter": "Reuters news",
    "wp": "WritingPrompts",
}

# fixed category order per panel -- labels must match rq1_variance.csv exactly
SCHEMES: dict[str, list[str]] = {
    "sentence_type": ["SIMPLE", "COMPLEX", "COMPOUND", "COMPLEX-COMPOUND", "OTHER"],
    "sentence_structure": ["LOOSE", "PERIODIC", "OTHER"],
}
SCHEME_TITLES = {
    "sentence_type": "Sentence type",
    "sentence_structure": "Sentence structure",
}

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


def main():
    """Reads rq1_variance.csv and saves one PNG and one PDF per domain."""
    rates = pd.read_csv(VARIANCE_INPUT)
    for domain in DOMAIN_TITLES:
        fig = _plot_domain(domain, rates)
        for ext in ("png", "pdf"):
            fig.savefig(
                OUTPUT_DIR / f"rq1_variance_{domain}.{ext}",
                dpi=200,
                bbox_inches="tight",
            )
        plt.close(fig)


if __name__ == "__main__":
    main()
