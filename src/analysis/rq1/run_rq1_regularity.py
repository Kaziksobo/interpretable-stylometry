"""RQ1 Track 2 -- Regularity: within-document entropy over syntactic categories.

Tests whether a single AI-generated document spreads across recognised
sentence types/structures more or less evenly than a human document does.

See docs/rq1_methodology.md §5 for the full pipeline (Eq. 9-11) and §10.5
for exact output schemas.

Inputs:
    data/processed/features/doc_features.feather

Outputs:
    data/processed/rq1/doc_entropy.feather
    results/rq1/rq1_regularity_significance.csv   -- Track 2, under review
"""

from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from compositions import (
    compute_proportions_algo1,
    compute_proportions_algo2,
)
from entropy import miller_madow, plugin_entropy
from scipy import stats

from src.analysis.shared.filters import filter_min_sents
from src.analysis.shared.significance_utils import apply_fdr, run_test_family

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "doc_features.feather"
ENTROPY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "rq1" / "doc_entropy.feather"
SIGNIFICANCE_OUTPUT = (
    PROJECT_ROOT / "results" / "rq1" / "rq1_regularity_significance.csv"
)

COMPARISONS = [("human", "gpt"), ("human", "claude")]


class CompositionResult(NamedTuple):
    entropy: pd.DataFrame
    significance: pd.DataFrame


COMPOSITIONS: list[tuple[str, Callable]] = [
    ("sentence_type", compute_proportions_algo1),
    ("sentence_structure", compute_proportions_algo2),
]


def _process_composition(df, proportions_fn, composition_name) -> CompositionResult:
    """Runs the full Track 2 pipeline for one composition (§5.1 steps 1-5).

    Reuses compute_proportions_algo1/2 from compositions.py purely for
    its proportions and `denom` output -- `denom` is exactly the
    composition-specific `n` Eq. 11's bias correction needs, so no
    separate computation is required. No zero-replacement is needed:
    entropy handles p=0 natively. Mann-Whitney per domain, human vs.
    each AI source, on the corrected entropies; a ratio above 1 supports
    "greater regularity", since it means lower AI entropy.

    Args:
        df (pd.DataFrame): Filtered doc_features, shared across both
            compositions -- not yet split by composition.
        proportions_fn (Callable): compute_proportions_algo1 or
            compute_proportions_algo2.
        composition_name (str): "sentence_type" or "sentence_structure",
            stamped onto every output row's `composition` column before
            returning.

    Returns:
        CompositionResult: This composition's contribution to
            doc_entropy.feather and rq1_regularity_significance.csv
            (§10.5). No file I/O happens here -- main() concatenates
            each field across both compositions and writes the combined
            result once.
    """
    # Compute proportions and denom, drop any documents with undefined proportions
    proportions_df = proportions_fn(df)
    value_cols = [c for c in proportions_df.columns if c.startswith("p_")]

    # Compute plugin entropy and Miller-Madow corrected entropy
    H_plugin = plugin_entropy(proportions_df[value_cols])
    D = len(value_cols)
    H_mm = miller_madow(H_plugin, D, proportions_df["denom"])

    cells = {}
    effect_sizes = {}
    for domain in proportions_df["domain"].unique():
        domain_hmm = H_mm[proportions_df["domain"] == domain]
        human_hmm = domain_hmm[proportions_df["source"] == "human"].to_numpy()
        for human_label, ai_label in COMPARISONS:
            ai_values = domain_hmm[proportions_df["source"] == ai_label].to_numpy()
            comparison = f"{human_label}_vs_{ai_label}"
            cells[(domain, comparison)] = (human_hmm, ai_values)
            effect_sizes[(domain, comparison)] = {
                "n_human": len(human_hmm),
                "n_ai": len(ai_values),
                "mean_entropy_human": np.mean(human_hmm),
                "mean_entropy_ai": np.mean(ai_values),
                "median_entropy_human": np.median(human_hmm),
                "median_entropy_ai": np.median(ai_values),
                "entropy_ratio": np.mean(human_hmm) / np.mean(ai_values),
            }

    # Run significance tests
    sig_df = run_test_family(cells, stats.mannwhitneyu, ["domain", "comparison"])

    # Add effect sizes to the significance DataFrame
    effect_df = pd.DataFrame.from_dict(effect_sizes, orient="index")
    effect_df = effect_df.rename_axis(["domain", "comparison"]).reset_index()
    sig_df = pd.merge(
        sig_df, effect_df, on=["domain", "comparison"], validate="one_to_one"
    )

    assert len(sig_df) == 6, (
        f"Expected 6 rows for {composition_name}, got {len(sig_df)}"
    )

    entropy_df = proportions_df[["doc_id", "domain", "source", "author"]].copy()
    entropy_df["entropy_plugin"] = H_plugin
    entropy_df["entropy_mm"] = H_mm

    entropy_df["composition"] = composition_name
    sig_df["composition"] = composition_name

    sig_df = sig_df.rename(columns={"statistic": "U_stat"})

    return CompositionResult(entropy=entropy_df, significance=sig_df)


def main():
    """Loads doc_features, runs both compositions, and writes the outputs.

    FDR correction is applied once, here, on the concatenated 12-row
    significance frame across both compositions -- not inside
    _process_composition, which would correct over 6 tests per call
    instead of the 12-test family §8 specifies.
    """
    df = filter_min_sents(pd.read_feather(INPUT_PATH))
    results = [_process_composition(df, fn, name) for name, fn in COMPOSITIONS]

    entropy = pd.concat([r.entropy for r in results], ignore_index=True)
    entropy = entropy[
        [
            "doc_id",
            "domain",
            "source",
            "author",
            "composition",
            "entropy_plugin",
            "entropy_mm",
        ]
    ]
    entropy.to_feather(ENTROPY_OUTPUT)
    significance = pd.concat([r.significance for r in results], ignore_index=True)
    significance = apply_fdr(significance, pvalue_col="p_value")
    significance = significance.rename(columns={"p_value_significant": "significant"})
    significance = significance[
        [
            "composition",
            "domain",
            "comparison",
            "n_human",
            "n_ai",
            "mean_entropy_human",
            "mean_entropy_ai",
            "median_entropy_human",
            "median_entropy_ai",
            "entropy_ratio",
            "U_stat",
            "p_value",
            "p_value_fdr",
            "significant",
        ]
    ]
    significance.to_csv(SIGNIFICANCE_OUTPUT, index=False)


if __name__ == "__main__":
    main()
