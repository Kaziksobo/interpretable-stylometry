"""Constants and helpers shared by the plotting scripts (RQ0, RQ1, repertoire).

One definition of how sources, domains and Feng schemes are named and coloured,
so every figure matches, plus the footnote and the save step that every
multi-domain figure repeats. matplotlib is only imported inside save_figure, so
importing the constants costs nothing.
Not a driver script -- imported, no __main__.
"""

from pathlib import Path

SOURCES = ["human", "gpt", "claude"]
SOURCE_LABELS = {"human": "Human", "gpt": "GPT", "claude": "Claude"}
# Okabe-Ito palette: distinguishable for colour-blind readers; same in every figure
COLOURS = {"human": "#0072B2", "gpt": "#E69F00", "claude": "#009E73"}

DOMAIN_TITLES = {
    "essay": "Student essays",
    "reuter": "Reuters news",
    "wp": "WritingPrompts",
}

# fixed category order per panel -- labels must match the results CSVs exactly
SCHEMES: dict[str, list[str]] = {
    "sentence_type": ["SIMPLE", "COMPLEX", "COMPOUND", "COMPLEX-COMPOUND", "OTHER"],
    "sentence_structure": ["LOOSE", "PERIODIC", "OTHER"],
}
SCHEME_TITLES = {
    "sentence_type": "Sentence type",
    "sentence_structure": "Sentence structure",
}

BAR_WIDTH = 0.26  # three grouped bars (one per source) fit around each x position


def doc_count_footnote(rates) -> str:
    """Builds the per-domain document-count footnote for multi-domain figures.

    Args:
        rates (pd.DataFrame): A results table with "domain", "source" and
            "n_docs" columns, e.g. rq0_rates.csv or rq1_variance.csv.

    Returns:
        str: e.g. "Documents (human / GPT / Claude): Student essays
            987/998/1000; ...".
    """
    n = rates.groupby(["domain", "source"])["n_docs"].first()
    parts = [
        f"{title} {n[(domain, 'human')]}/{n[(domain, 'gpt')]}/{n[(domain, 'claude')]}"
        for domain, title in DOMAIN_TITLES.items()
    ]
    return "Documents (human / GPT / Claude): " + "; ".join(parts)


def save_figure(fig, path_stem: Path, pdf: bool = False) -> Path:
    """Saves a figure as a PNG, or as a vector PDF with pdf=True, then closes it.

    Never saves both, so a results folder only ever holds one format per
    figure.

    Args:
        fig (plt.Figure): The figure to save.
        path_stem (Path): Output path without its extension.
        pdf (bool): Save a PDF instead of a PNG. Defaults to False.

    Returns:
        Path: The file that was written.
    """
    import matplotlib.pyplot as plt

    path = path_stem.parent / f"{path_stem.name}.{'pdf' if pdf else 'png'}"
    fig.savefig(path, dpi=200, bbox_inches="tight")  # dpi is ignored for PDF (vector)
    plt.close(fig)
    return path
