"""RQ1 Track 1 -- Variance: multivariate compositional dispersion.

Tests whether AI-generated prose shows reduced between-document variance
in syntactic composition, via CLR transform + distance-to-centroid, rather
than testing each category rate separately (see Track 3 for that).

See docs/rq1_methodology.md §4 for the full pipeline (Eq. 1-10) and §9.1,
§9.3, §9.6 for exact output schemas.

Inputs:
    data/processed/rq1/doc_features.feather

Outputs:
    data/processed/rq1/doc_distances.feather
    results/rq1/rq1_variance_significance.csv   -- RQ1 answer file
    results/rq1/centroids.csv
"""

from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from compositions import (
    clr_transform,
    compute_centroid,
    compute_deltas,
    compute_distances,
    compute_proportions_algo1,
    compute_proportions_algo2,
    filter_min_sents,
    inverse_clr,
    multiplicative_replacement,
)
from scipy import stats
from significance_utils import apply_fdr, run_test_family

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rq1" / "doc_features.feather"
DISTANCES_OUTPUT = PROJECT_ROOT / "data" / "processed" / "rq1" / "doc_distances.feather"
CENTROIDS_OUTPUT = PROJECT_ROOT / "results" / "rq1" / "centroids.csv"
SIGNIFICANCE_OUTPUT = PROJECT_ROOT / "results" / "rq1" / "rq1_variance_significance.csv"

COMPARISONS = [("human", "gpt"), ("human", "claude")]


class CompositionResult(NamedTuple):
    distances: pd.DataFrame
    centroids: pd.DataFrame
    significance: pd.DataFrame


COMPOSITIONS: list[tuple[str, Callable]] = [
    ("sentence_type", compute_proportions_algo1),
    ("sentence_structure", compute_proportions_algo2),
]


def _process_composition(df, proportions_fn, composition_name) -> CompositionResult:
    """Runs the full Track 1 pipeline for one composition (§4.1 steps 2-13).

    Computes proportions, replaces zeros, CLR-transforms, finds each
    (domain, source) group's centroid, and measures every document's
    distance to its own group's centroid. Centroids are also
    back-transformed to the simplex for centroids.csv, kept separate
    from the CLR-space version used internally for the distance
    calculation. Finishes by testing human vs. each AI source's
    distances per domain (Mann-Whitney) and computing the matching
    effect sizes.

    Args:
        df (pd.DataFrame): Filtered doc_features, shared across both
            compositions -- not yet split by composition.
        proportions_fn (Callable): compute_proportions_algo1 or
            compute_proportions_algo2 -- determines the denominator
            logic (Eq. 1a/1b), not just a column selection.
        composition_name (str): "sentence_type" or "sentence_structure",
            stamped onto every output row's `composition` column before
            returning.

    Returns:
        CompositionResult: This composition's contribution to
            doc_distances.feather, centroids.csv, and
            rq1_variance_significance.csv (§9.1, §9.6, §9.3). No file
            I/O happens here -- main() concatenates each field across
            both compositions and writes the combined result once.
    """
    # Compute proportions, drop any documents with undefined proportions
    proportions_df = proportions_fn(df)
    value_cols = [c for c in proportions_df.columns if c.startswith("p_")]

    # Compute zero-replacement constants
    deltas = compute_deltas(proportions_df["denom"])

    # Apply multiplicative replacement to the proportions
    replaced_proportions = proportions_df.copy()
    replaced_proportions[value_cols] = multiplicative_replacement(
        proportions_df[value_cols], deltas
    )

    # Apply CLR transform to the replaced proportions
    clr_df = replaced_proportions.copy()
    clr_df[value_cols] = clr_transform(clr_df[value_cols].to_numpy())

    # Compute centroids in CLR space
    centroids = compute_centroid(
        clr_df=clr_df, group_cols=["domain", "source"], value_cols=value_cols
    )

    # Compute distances to centroids
    distances = compute_distances(
        clr_df=clr_df,
        centroids_df=centroids,
        group_cols=["domain", "source"],
        value_cols=value_cols,
    )

    # Back-transform centroids to the simplex for centroids.csv (§9.6) --
    # `centroids` itself stays CLR-space, only used internally above
    centroids_simplex = centroids.copy()
    centroids_simplex[value_cols] = inverse_clr(centroids[value_cols].to_numpy())
    centroids_long = centroids_simplex.melt(
        id_vars=["domain", "source"],
        value_vars=value_cols,
        var_name="category",
        value_name="geometric_mean_proportion",
    )

    cells = {}
    effect_sizes = {}
    for domain in distances["domain"].unique():
        domain_dist = distances[distances["domain"] == domain]
        human_values = domain_dist[domain_dist["source"] == "human"][
            "distance"
        ].to_numpy()
        for human_label, ai_label in COMPARISONS:
            ai_values = domain_dist[domain_dist["source"] == ai_label][
                "distance"
            ].to_numpy()
            comparison = f"{human_label}_vs_{ai_label}"
            cells[(domain, comparison)] = (human_values, ai_values)
            effect_sizes[(domain, comparison)] = {
                "n_human": len(human_values),
                "n_ai": len(ai_values),
                "mean_dist_human": np.mean(human_values),
                "mean_dist_ai": np.mean(ai_values),
                "median_dist_human": np.median(human_values),
                "median_dist_ai": np.median(ai_values),
                "dist_ratio": np.mean(human_values) / np.mean(ai_values)
                if np.mean(ai_values) > 0
                else np.nan,
                "mean_distsq_human": np.mean(human_values**2),
                "mean_distsq_ai": np.mean(ai_values**2),
                "distsq_ratio": np.mean(human_values**2) / np.mean(ai_values**2)
                if np.mean(ai_values**2) > 0
                else np.nan,
            }

    # Run significance tests
    sig_df = run_test_family(
        cells, stats.mannwhitneyu, key_names=["domain", "comparison"]
    )

    # Add effect sizes to the significance DataFrame
    effect_df = pd.DataFrame.from_dict(effect_sizes, orient="index")
    effect_df = effect_df.rename_axis(["domain", "comparison"]).reset_index()
    sig_df = pd.merge(
        sig_df, effect_df, on=["domain", "comparison"], validate="one_to_one"
    )

    assert len(sig_df) == 6, (
        f"Expected 6 rows for {composition_name}, got {len(sig_df)}"
    )

    distances["composition"] = composition_name
    centroids_long["composition"] = composition_name
    sig_df["composition"] = composition_name

    sig_df = sig_df.rename(columns={"statistic": "U_stat"})

    return CompositionResult(
        distances=distances,
        centroids=centroids_long,
        significance=sig_df,
    )


def main():
    """Loads doc_features, runs both compositions, and writes the outputs.

    FDR correction is applied once, here, on the concatenated 12-row
    significance frame across both compositions -- not inside
    _process_composition, which would correct over 6 tests per call
    instead of the 12-test family §7 specifies.
    """
    df = filter_min_sents(pd.read_feather(INPUT_PATH))
    results = [_process_composition(df, fn, name) for name, fn in COMPOSITIONS]

    distances = pd.concat([r.distances for r in results], ignore_index=True)
    distances = distances[
        [
            "doc_id",
            "domain",
            "source",
            "author",
            "composition",
            "distance",
            "distance_sq",
        ]
    ]
    distances.to_feather(DISTANCES_OUTPUT)

    centroids = pd.concat([r.centroids for r in results], ignore_index=True)
    centroids = centroids[
        ["domain", "source", "composition", "category", "geometric_mean_proportion"]
    ]
    centroids.to_csv(CENTROIDS_OUTPUT, index=False)
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
            "mean_dist_human",
            "mean_dist_ai",
            "median_dist_human",
            "median_dist_ai",
            "dist_ratio",
            "mean_distsq_human",
            "mean_distsq_ai",
            "distsq_ratio",
            "U_stat",
            "p_value",
            "p_value_fdr",
            "significant",
        ]
    ]
    significance.to_csv(SIGNIFICANCE_OUTPUT, index=False)


if __name__ == "__main__":
    main()
