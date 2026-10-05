"""Benjamini-Hochberg FDR correction, shared across the analyses.

apply_fdr(): Benjamini-Hochberg wrapper, adds _fdr and _significant columns.

See docs/rq1_methodology.md §7 for the FDR family structure this feeds
(RQ1's 48 variance tests form one family).
Not a driver script -- imported, no __main__.
"""

import pandas as pd
from statsmodels.stats.multitest import multipletests


def apply_fdr(df: pd.DataFrame, pvalue_col: str, alpha: float = 0.05) -> pd.DataFrame:
    """Applies Benjamini-Hochberg FDR correction across one test family (§7).

    Every row in df is treated as belonging to the same family -- call
    this once per family (RQ1 methodology §7: the 48 variance tests) and
    never on a concatenation of several families, since each family is
    corrected independently.

    Args:
        df (pd.DataFrame): One row per test, with a p-value column.
        pvalue_col (str): Name of the column holding raw p-values.
        alpha (float): FDR threshold. Defaults to 0.05, used throughout
            this project (§7).

    Returns:
        pd.DataFrame: df with two new columns -- f"{pvalue_col}_fdr"
            (the corrected p-values) and f"{pvalue_col}_significant"
            (bool, whether the corrected p-value is below alpha).
    """
    reject, pvals_corrected, _, _ = multipletests(
        df[pvalue_col], alpha=alpha, method="fdr_bh"
    )
    df[f"{pvalue_col}_fdr"] = pvals_corrected
    df[f"{pvalue_col}_significant"] = reject
    return df
