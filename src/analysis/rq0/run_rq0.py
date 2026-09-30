"""RQ0 -- do humans, GPT and Claude use each sentence type/structure at different rates?

For each (domain, source, category): each document's rate (count / n_sents),
then the mean across documents, its standard error and a 95% interval. The
document is the unit throughout. See docs/rq0_methodology.md.

Inputs:
    data/processed/rq1/doc_features.feather

Outputs:
    results/rq0/rq0_rates.csv
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.analysis.shared.filters import filter_min_sents

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "doc_features.feather"
RATES_OUTPUT = PROJECT_ROOT / "results" / "rq0" / "rq0_rates.csv"

Z_95 = 1.96

# count column -> (scheme, category label)
CATEGORIES: dict[str, tuple[str, str]] = {
    "sent_simple": ("sentence_type", "SIMPLE"),
    "sent_complex": ("sentence_type", "COMPLEX"),
    "sent_compound": ("sentence_type", "COMPOUND"),
    "sent_complex_compound": ("sentence_type", "COMPLEX-COMPOUND"),
    "sent_other": ("sentence_type", "OTHER"),
    "struct_loose": ("sentence_structure", "LOOSE"),
    "struct_periodic": ("sentence_structure", "PERIODIC"),
    "struct_other": ("sentence_structure", "OTHER"),
}


def _to_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Converts every category count to a per-document rate (Eq. 1).

    Works on a copy and overwrites the count columns, so nothing
    downstream can read a count where it expects a rate.

    Args:
        df (pd.DataFrame): Filtered doc_features, with n_sents and the
            eight count columns in CATEGORIES.

    Returns:
        pd.DataFrame: Same rows and columns, each count column divided
            by n_sents.
    """
    rates_df = df.copy()
    for count_col in CATEGORIES.keys():
        rates_df[count_col] = rates_df[count_col] / rates_df["n_sents"]
    return rates_df


def _summarise(rates: pd.DataFrame) -> pd.DataFrame:
    """Computes the mean rate, SE and 95% interval per (domain, source, category).

    Args:
        rates (pd.DataFrame): _to_rates' output.

    Returns:
        pd.DataFrame: 72 rows -- domain, source, scheme, category, n_docs,
            mean_rate, se, ci_low, ci_high (Eq. 2-3).
    """
    results = []
    for key, group in rates.groupby(["domain", "source"]):
        domain, source = cast(tuple[str, str], key)
        for count_col, (scheme, category) in CATEGORIES.items():
            values = group[count_col]
            mean_rate = values.mean()
            # Use ddof=1 for sample standard deviation
            std_dev = values.std(ddof=1)
            n_docs = len(values)
            se = std_dev / np.sqrt(n_docs)
            ci_low = mean_rate - Z_95 * se
            ci_high = mean_rate + Z_95 * se
            results.append(
                {
                    "domain": domain,
                    "source": source,
                    "scheme": scheme,
                    "category": category,
                    "n_docs": n_docs,
                    "mean_rate": mean_rate,
                    "se": se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(results)


def main():
    """Loads, filters, converts to rates, summarises, checks and writes the CSV."""
    summary = _summarise(_to_rates(filter_min_sents(pd.read_feather(INPUT_PATH))))

    assert len(summary) == 72, f"expected 72 rows, got {len(summary)}"
    assert np.allclose(
        summary.groupby(["domain", "source", "scheme"])["mean_rate"].sum(), 1
    ), (
        "mean rates within a scheme don't sum to 1 -- "
        "mislabelled scheme, or a count read as a rate?"
    )
    assert (summary["ci_low"] >= 0).all(), "a confidence interval goes below 0"

    RATES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RATES_OUTPUT, index=False)


if __name__ == "__main__":
    main()
