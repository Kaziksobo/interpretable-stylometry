"""Generic test-family runner shared across RQ1 Tracks 1, 2, and 3.

run_test_family(): takes a {(domain, feature/composition, comparison):
(human_values, ai_values)} dict and a scipy test function, returns raw
stat + p-value per cell.

apply_fdr(): Benjamini-Hochberg wrapper, adds _fdr and _significant columns.

See docs/rq1_methodology.md §7 for the FDR family structure this feeds
(four separate families -- don't pool across tracks).
Not a driver script -- imported, no __main__.
"""

from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def run_test_family(
    cells: dict[tuple, tuple[np.ndarray, np.ndarray]],
    test_fn: Callable,
    key_names: list[str],
) -> pd.DataFrame:
    """Runs one significance test across a family of (human, AI) cells.

    Args:
        cells (dict[tuple, tuple[np.ndarray, np.ndarray]]): Maps a key
            tuple identifying one test -- e.g. (domain, composition,
            comparison) for Tracks 1/2, (domain, feature, comparison) for
            Track 3 -- to (human_values, ai_values), 1-D arrays/Series
            with no NaNs.
        test_fn (Callable): A two-sample scipy test, called as
            test_fn(human_values, ai_values), e.g.
            scipy.stats.mannwhitneyu, or
            functools.partial(scipy.stats.levene, center="median") for
            Track 3's dispersion test.
        key_names (list[str]): Column names for the unpacked key tuple,
            in the SAME ORDER the tuples themselves are built in, e.g.
            ["domain", "composition", "comparison"]. Nothing here checks
            that the order matches the caller's -- getting it wrong
            produces a table with correct values under the wrong headers.

    Returns:
        pd.DataFrame: One row per cell -- key_names columns plus
            `statistic` and `p_value`.
    """
    results = []
    for key_tuple, (human_values, ai_values) in cells.items():
        stat, p_value = test_fn(human_values, ai_values)
        results.append((*key_tuple, stat, p_value))

    return pd.DataFrame(results, columns=[*key_names, "statistic", "p_value"])


def apply_fdr(df: pd.DataFrame, pvalue_col: str, alpha: float = 0.05) -> pd.DataFrame:
    """Applies Benjamini-Hochberg FDR correction across one test family (§7).

    Every row in df is treated as belonging to the same family -- call
    this separately per family (Track 1, Track 2, Track 3-dispersion,
    Track 3-location) and never on a concatenation of several families,
    since §7 corrects the four families independently.

    Args:
        df (pd.DataFrame): run_test_family's output, or any DataFrame
            with a p-value column.
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
