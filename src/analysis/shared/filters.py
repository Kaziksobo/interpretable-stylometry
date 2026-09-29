"""Document filters shared across RQ0 and RQ1.

Every analysis filters the same way, so the document pool is identical
across research questions. Not a driver script -- imported, no __main__.
"""

import pandas as pd


def filter_min_sents(df: pd.DataFrame, min_sents: int = 5) -> pd.DataFrame:
    """Excludes documents with fewer than min_sents sentences (§3).

    A document with very few sentences gives an unreliable composition --
    e.g. n_sents=1 is 100% one category by construction, and tells us
    nothing about the document's actual tendency.

    Args:
        df (pd.DataFrame): Any DataFrame with an "n_sents" column, e.g.
            doc_features.feather loaded directly.
        min_sents (int): Minimum sentence count required to keep a
            document. Defaults to 5.

    Returns:
        pd.DataFrame: The filtered rows, with a fresh 0-based index.
    """
    filtered_df = df[df["n_sents"] >= min_sents]
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df
