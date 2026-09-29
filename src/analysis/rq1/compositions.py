"""Shared compositional-data utilities for RQ1 Tracks 1 and 2.

Proportion computation, per-document zero-replacement, and the
CLR transform -- everything needed to turn doc_features.feather counts into
a valid composition ready for either the variance (Track 1) or regularity
(Track 2) pipeline.

Imported by run_rq1_variance.py and run_rq1_regularity.py; not a driver
script itself, so there is no __main__ here. See docs/rq1_methodology.md
§4.1 (Eq. 1-5, 9) for the full specification.
"""

import numpy as np
import pandas as pd


def clr_transform(X: np.ndarray) -> np.ndarray:
    """Applies the centred log-ratio transform to each row of X (Eq. 4).

    Maps a composition on the simplex into unconstrained real space by
    logging each proportion and subtracting the row's own mean log -- the
    standard fix for proportions being bounded and only meaningful in
    ratio, not absolute difference.

    Args:
        X (np.ndarray): (n_docs, D) array of positive proportions, post
            zero-replacement (§4.1 step 4). Every entry must be > 0 -- a
            raw zero produces -inf via np.log.

    Returns:
        np.ndarray: (n_docs, D) array of CLR coordinates; each row sums
            to ~0 by construction.
    """
    x_log = np.log(X)
    x_log_mean = np.mean(x_log, axis=1, keepdims=True)
    return x_log - x_log_mean


def inverse_clr(Y: np.ndarray) -> np.ndarray:
    """Inverts clr_transform via softmax, recovering a valid composition.

    Args:
        Y (np.ndarray): (n_docs, D) or (D,) CLR-space vector(s).

    Returns:
        np.ndarray: Valid composition(s) on the simplex, each row (or the
            single vector) summing to 1.
    """
    exp_y = np.exp(Y)
    return exp_y / np.sum(exp_y, axis=-1, keepdims=True)


def compute_proportions_algo1(df: pd.DataFrame) -> pd.DataFrame:
    """Computes algo1 (sentence-type) proportions per document (Eq. 1a).

    The denominator is sent_simple + sent_complex + sent_compound +
    sent_complex_compound -- the four conforming categories; sent_other
    is excluded and routed to Track 3 instead (§3). Documents where that
    sum is 0 or 1 are dropped: 0 leaves the proportion undefined, and 1
    always puts all the mass on a single category (Z=3 for this 4-part
    composition), which drives multiplicative_replacement's scaling
    factor negative (see docs/rq1_methodology.md §9).

    Args:
        df (pd.DataFrame): Filtered doc_features (post filter_min_sents),
            with doc_id, domain, source, author, and the four sent_*
            count columns.

    Returns:
        pd.DataFrame: The id columns plus four new proportion columns
            (p_simple, p_complex, p_compound, p_complex_compound) and a
            `denom` column, one row per document with denom > 1.
    """
    # Copy the input DataFrame to avoid modifying it in place
    df = df.copy()
    # Compute the sum of the four conforming sentence types
    denom = (
        df["sent_simple"]
        + df["sent_complex"]
        + df["sent_compound"]
        + df["sent_complex_compound"]
    )
    # Compute the proportions
    df["p_simple"] = df["sent_simple"] / denom
    df["p_complex"] = df["sent_complex"] / denom
    df["p_compound"] = df["sent_compound"] / denom
    df["p_complex_compound"] = df["sent_complex_compound"] / denom

    # Drop rows with denom <= 1 - denom=0 is an undefined proportion;
    # denom=1 always forces Z=3 for this 4-category composition (all mass
    # on one category), which drives multiplicative_replacement's scaling
    # factor negative (1 - 3*0.5 = -0.5). See docs/rq1_methodology.md §9.
    df = df[denom > 1]

    # Add a new denom column to the DataFrame for reference
    df["denom"] = denom[denom > 0]
    df.reset_index(drop=True, inplace=True)

    return df


def compute_proportions_algo2(df: pd.DataFrame) -> pd.DataFrame:
    """Computes algo2 (structure) proportions per document (Eq. 1b).

    The denominator is n_sents directly: struct_loose + struct_periodic +
    struct_other sum exactly to n_sents for every document (no residual
    category), so no row is dropped here.

    Args:
        df (pd.DataFrame): Filtered doc_features, with doc_id, domain,
            source, author, n_sents, and the three struct_* count
            columns.

    Returns:
        pd.DataFrame: The id columns plus three new proportion columns
            (p_loose, p_periodic, p_other) and a `denom` column (equal to
            n_sents), same row count as the input.
    """
    # Copy the input DataFrame to avoid modifying it in place
    df = df.copy()
    # Compute the proportions
    df["p_loose"] = df["struct_loose"] / df["n_sents"]
    df["p_periodic"] = df["struct_periodic"] / df["n_sents"]
    df["p_other"] = df["struct_other"] / df["n_sents"]

    # Add a new denom column to the DataFrame for reference
    df["denom"] = df["n_sents"]

    return df


def compute_deltas(denom: pd.Series) -> pd.Series:
    """Computes the per-document zero-replacement constant (Eq. 2).

    δ_i = 0.5 / denom_i: a smaller denominator gives a larger δ, since a
    zero category in a short document is weaker evidence of "never
    occurs" than the same zero in a long one.

    Args:
        denom (pd.Series): Each document's own composition denominator --
            the `denom` column from compute_proportions_algo1 or
            compute_proportions_algo2. Every value must be > 0.

    Returns:
        pd.Series: δ_i, same shape and index as denom.

    Raises:
        AssertionError: If any value in denom is not strictly positive.
    """
    # Check that all denominators are positive
    assert (denom > 0).all()

    return 0.5 / denom


def multiplicative_replacement(
    proportions: pd.DataFrame, deltas: pd.Series
) -> pd.DataFrame:
    """Replaces zero proportions with δ, per document (Eq. 3).

    Each zero category is set to δ_i, and the nonzero categories in that
    row are shrunk by (1 - Z_i·δ_i) to make room, so the row still sums
    to 1 (Eq. 9's closure). The scaling factor is asserted positive
    rather than left to fail silently: a value <= 0 means a document with
    too many zero categories for its own denominator slipped past the
    upstream filter in compute_proportions_algo1.

    Args:
        proportions (pd.DataFrame): (n_docs, D) -- just the p_* columns
            for one composition, sharing an index with deltas.
        deltas (pd.Series): δ_i per document (compute_deltas' output),
            same index as proportions.

    Returns:
        pd.DataFrame: Same shape as proportions, with zeros replaced by
            δ_i and nonzero values scaled down to preserve closure.

    Raises:
        AssertionError: If proportions and deltas have different indices,
            or if any document's scaling factor is <= 0.
    """
    # Copy the input DataFrame to avoid modifying it in place
    proportions = proportions.copy()

    # Check that the indices of proportions and deltas match
    assert proportions.index.equals(deltas.index)

    # Create a boolean mask for zero entries
    zero_mask = proportions == 0

    # Compute Z_i: number of zeros per document (row)
    Z_i = zero_mask.sum(axis=1)

    # Compute the scaling factor for non-zero entries: (1 - Z_i * δ_i)
    scaling_factor = 1 - (Z_i * deltas)
    assert (scaling_factor > 0).all(), (
        "scaling_factor <= 0 for at least one document - Z_i*delta_i >= 1. "
        "This means too many zero categories for that document's own "
        "denominator; it should have been excluded upstream (see "
        "compute_proportions_algo1's denom filter), not silently produce "
        "a negative proportion here."
    )

    # Replace zeros with δ_i and scale non-zero entries
    for col in proportions.columns:
        proportions[col] = np.where(
            zero_mask[col], deltas, proportions[col] * scaling_factor
        )

    return proportions


def compute_centroid(
    clr_df: pd.DataFrame, group_cols: list[str], value_cols: list[str]
) -> pd.DataFrame:
    """Computes each group's centroid in CLR space (Eq. 5).

    Args:
        clr_df (pd.DataFrame): One row per document, with group_cols
            (e.g. domain, source) and value_cols (the CLR-space columns
            for one composition) present together.
        group_cols (list[str]): Columns to group by, e.g.
            ["domain", "source"].
        value_cols (list[str]): The CLR column names for this
            composition.

    Returns:
        pd.DataFrame: One row per group; value_cols now hold that group's
            mean CLR vector rather than a per-document value.
    """
    return clr_df.groupby(group_cols)[value_cols].mean().reset_index()


def compute_distances(
    clr_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    """Computes each document's distance to its own group's centroid (Eq. 6, 7).

    Merges centroids_df onto clr_df by group_cols rather than
    recomputing the centroid here, so distance is always measured
    against compute_centroid's actual output -- there is only one
    definition of "this group's centroid" in the pipeline.

    Args:
        clr_df (pd.DataFrame): One row per document, with group_cols and
            value_cols present.
        centroids_df (pd.DataFrame): compute_centroid's output -- one row
            per group, same group_cols and value_cols.
        group_cols (list[str]): Columns to merge on, e.g.
            ["domain", "source"].
        value_cols (list[str]): The CLR column names for this
            composition.

    Returns:
        pd.DataFrame: clr_df's doc_id, domain, source and author columns,
            plus `distance` (Eq. 6, plain -- the headline statistic) and
            `distance_sq` (Eq. 7, squared -- supplementary), one row per
            document.
    """
    # Merge the centroids into the clr_df based on group_cols
    merged_df = clr_df.merge(centroids_df, on=group_cols, suffixes=("", "_centroid"))

    # Compute the squared Euclidean distance to the centroid
    diff = (
        merged_df[value_cols].values
        - merged_df[[f"{col}_centroid" for col in value_cols]].values
    )
    distance_sq = np.sum(diff**2, axis=1)
    distance = np.sqrt(distance_sq)

    # Prepare the result DataFrame
    result_df = merged_df[group_cols + ["doc_id", "author"]].copy()
    result_df["distance"] = distance
    result_df["distance_sq"] = distance_sq

    return result_df
