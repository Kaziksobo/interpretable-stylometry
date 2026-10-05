"""Proportion utilities for RQ1 Track 2 (regularity).

compute_proportions_algo1/2 turn doc_features counts into per-document
proportions plus a `denom` column, which Track 2's entropy and its
Miller-Madow correction need. Track 2 is under review
(docs/rq1_methodology.md §5.2): the algo1 scope and its `denom <= 1`
exclusion were inherited from the retired CLR track.

Imported by run_rq1_regularity.py; not a driver script, no __main__.
"""

import pandas as pd


def compute_proportions_algo1(df: pd.DataFrame) -> pd.DataFrame:
    """Computes algo1 (sentence-type) proportions per document (§5.1 step 1).

    The denominator is sent_simple + sent_complex + sent_compound +
    sent_complex_compound -- the four conforming categories; sent_other
    is excluded. Documents where that sum is 0 or 1 are dropped: 0 leaves
    the proportion undefined, and the 1 is a rule inherited from the
    retired CLR track, where a single conforming sentence drove
    zero-replacement negative. It's kept so Track 2's document pool is
    unchanged until Track 2 is reviewed (docs/rq1_methodology.md §5.2).

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

    # Drop rows with denom <= 1: denom=0 is an undefined proportion; denom=1
    # is a rule inherited from the retired CLR track, kept until Track 2 is
    # reviewed (docs/rq1_methodology.md §5.2).
    df = df[denom > 1]

    # Add a new denom column to the DataFrame for reference
    df["denom"] = denom[denom > 1]
    df.reset_index(drop=True, inplace=True)

    return df


def compute_proportions_algo2(df: pd.DataFrame) -> pd.DataFrame:
    """Computes algo2 (structure) proportions per document (§5.1 step 1).

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
