"""Shannon entropy with Miller-Madow bias correction, for RQ1 Track 2.

See docs/rq1_methodology.md §5.1 (Eq. 9-11) for the full specification,
including why the plug-in estimator needs correcting here.
Not a driver script -- imported by run_rq1_regularity.py, no __main__.
"""

import numpy as np
import pandas as pd


def plugin_entropy(proportions: pd.DataFrame) -> pd.Series:
    """Computes the plug-in Shannon entropy per document (Eq. 10).

    Ĥ = -Σ p_i ln(p_i). A zero-probability category should contribute
    exactly 0 to the sum (the true limit of p·ln(p) as p→0), but naive
    floating-point arithmetic gives NaN there (0 * -inf), so zero entries
    are substituted with a safe value of 1 before the log, making their
    ln(1)=0 term drop out cleanly without ever computing log(0).

    Args:
        proportions (pd.DataFrame): (n_docs, D) -- just the p_* columns
            for one composition, as raw proportions. Entropy handles p=0
            natively, so no zero-replacement is needed (§5.1 step 1).

    Returns:
        pd.Series: Ĥ per document, same index as proportions.
    """
    # np.log(0) correction
    safe_p = np.where(proportions > 0, proportions, 1.0)
    p = proportions * np.log(safe_p)

    return -p.sum(axis=1)


def miller_madow(H_plugin: pd.Series, D: int, n: pd.Series) -> pd.Series:
    """Applies the Miller-Madow bias correction to plug-in entropy (Eq. 11).

    H_MM = Ĥ + (D-1)/(2n). The plug-in estimator is negatively biased for
    small samples -- rare categories are under-observed in few draws, so
    entropy mechanically undershoots its true value; this correction adds
    that bias back.

    Args:
        H_plugin (pd.Series): plugin_entropy's output.
        D (int): Number of categories in this composition (4 for algo1,
            3 for algo2) -- a fixed constant, not per-document.
        n (pd.Series): Composition-specific document sample size -- the
            `denom` column from compositions.py, same index as H_plugin.

    Returns:
        pd.Series: H_MM per document, same index as H_plugin.
    """
    return H_plugin + (D - 1) / (2 * n)
