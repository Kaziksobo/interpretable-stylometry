"""RQ1 Track 1:
    is the rate of each sentence type/structure less variable between AI documents?

For each (domain, source, category): the real between-document variance of the
rate, V-hat = S^2 - mean noise, with each document's binomial noise estimated
from its own count, and the relative version rho-hat = V-hat / (rbar(1-rbar)).
Uncertainty by resampling documents (whole authors in reuter); human vs AI
compared with a z-test on the difference in rho-hat, FDR-corrected across all
48 tests. A second pass without the six documents judged invalid on inspection
checks whether any verdict depends on them.

See docs/rq1_methodology.md (Track 1).

Inputs:
    data/processed/features/doc_features.feather

Outputs:
    results/rq1/rq1_variance.csv                     -- 72 rows, plotted
    results/rq1/rq1_variance_tests.csv               -- 48 rows
    results/rq1/rq1_variance_tests_sensitivity.csv   -- 48 rows, main vs six excluded
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.shared.filters import filter_min_sents
from src.analysis.shared.significance_utils import apply_fdr

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "doc_features.feather"
VARIANCE_OUTPUT = PROJECT_ROOT / "results" / "rq1" / "rq1_variance.csv"
TESTS_OUTPUT = PROJECT_ROOT / "results" / "rq1" / "rq1_variance_tests.csv"
SENSITIVITY_OUTPUT = (
    PROJECT_ROOT / "results" / "rq1" / "rq1_variance_tests_sensitivity.csv"
)

N_BOOT = 2000
SEED = 20260930  # fixed, so every run gives identical numbers
CLUSTER_DOMAINS = {
    "reuter"
}  # resample whole authors here (documents cluster by author)

COMPARISONS = [("human", "gpt"), ("human", "claude")]
DOMAINS = ["essay", "reuter", "wp"]
SOURCES = ["human", "gpt", "claude"]

# count column -> (scheme, category label); denominator is n_sents for all, as in RQ0
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

# judged non-prose on inspection: (domain, source, doc_id, author) -- sens. pass only
EXCLUDED_DOCS: list[tuple[str, str, str, str | None]] = [
    ("essay", "claude", "851", None),  # list of interview questions
    ("essay", "human", "841", None),  # sectioned company overview
    ("reuter", "human", "9", "JimGilchrist"),  # mostly numbered statistics
    ("wp", "gpt", "671", None),  # Reddit-thread format
    ("wp", "human", "627", None),  # terminal-log format
    ("wp", "human", "903", None),  # Google-search format
]


def variance_components(k: np.ndarray, n: np.ndarray) -> tuple[float, float, float]:
    """Estimates the mean rate, real between-document variance and rho (steps 1-5).

    Subtracts each document's binomial sampling noise, r(1-r)/(n-1), from the
    observed variance of the rates, which leaves an unbiased estimate of the
    variance of the documents' true rates whatever their distribution. rho
    divides that by its ceiling, rbar(1-rbar). Can be slightly negative when
    there is no real variation; left unclipped, since clipping would bias it.

    Args:
        k (np.ndarray): The category's count in each document.
        n (np.ndarray): Sentences in each document (all >= 2).

    Returns:
        tuple[float, float, float]: (rbar, V, rho).
    """
    r = k / n  # per-document rate
    S2 = r.var(ddof=1)  # observed spread (real + noise)
    noise = np.mean(r * (1 - r) / (n - 1))  # average estimated noise
    V = S2 - noise  # real between-document variance
    rbar = r.mean()  # mean rate across documents
    rho = V / (rbar * (1 - rbar))  # share of the maximum possible
    return rbar, V, rho


def _resample_indices(
    group: pd.DataFrame, cluster: bool, rng: np.random.Generator
) -> list[np.ndarray]:
    """Builds N_BOOT sets of resampled row positions for one (domain, source) group.

    Document-level: N positions drawn with replacement. Author-level: the
    group's authors drawn with replacement, taking all of each chosen author's
    documents, so documents sharing an author stay together and the resamples
    reflect that a new sample would bring in different authors.

    Positions are row numbers, so apply them to numpy arrays taken from this
    same `group` frame -- never to its index labels with `.loc`.

    Args:
        group (pd.DataFrame): One group's documents.
        cluster (bool): Resample by author rather than by document.
        rng (np.random.Generator): The group's own generator, created in _estimate.

    Returns:
        list[np.ndarray]: N_BOOT arrays of row positions into `group`.
            Author-level arrays vary in length, since authors wrote
            different numbers of documents.

    Raises:
        AssertionError: If cluster is True and a document has no author.
    """
    if not cluster:
        n_docs = len(group)
        return list(rng.integers(0, n_docs, size=(N_BOOT, n_docs)))

    codes, authors = pd.factorize(group["author"])
    assert (codes >= 0).all(), "a document has no author -- can't resample by author"
    members = [np.flatnonzero(codes == j) for j in range(len(authors))]
    n_authors = len(authors)
    return [
        np.concatenate([members[j] for j in rng.integers(0, n_authors, n_authors)])
        for _ in range(N_BOOT)
    ]


def _estimate(df: pd.DataFrame) -> pd.DataFrame:
    """Point estimates and resampled uncertainty for all 72 cells.

    Each group gets its own random generator, seeded from SEED and the
    group's position in DOMAINS and SOURCES, so a group's resamples depend
    only on its own documents -- removing documents from one group leaves
    every other group's resamples unchanged. One set of resample indices
    per group is reused for all eight categories.

    Args:
        df (pd.DataFrame): Filtered doc_features.

    Returns:
        pd.DataFrame: 72 rows -- domain, source, scheme, category, n_docs,
            n_clusters, mean_rate, V, rho, rho_se, rho_ci_low, rho_ci_high.
    """
    results = []
    for group_key, group in df.groupby(["domain", "source"]):
        domain, source = cast(tuple[str, str], group_key)
        rng = np.random.default_rng(
            [SEED, DOMAINS.index(domain), SOURCES.index(source)]
        )
        group = group.reset_index(drop=True)
        cluster = domain in CLUSTER_DOMAINS
        indices = _resample_indices(group, cluster, rng)

        n = group["n_sents"].to_numpy()

        for count_col, (scheme, category) in CATEGORIES.items():
            k = group[count_col].to_numpy()
            rbar, V, rho = variance_components(k, n)

            boot = np.array([variance_components(k[pos], n[pos])[2] for pos in indices])
            if not np.isfinite(boot).all():
                raise ValueError(
                    "Non-finite bootstrapped rho-hats for "
                    f"{domain}/{source}/{scheme}/{category} - "
                    "check that all documents have n_sents >= 2 "
                    "and the category is present."
                )
            results.append(
                {
                    "domain": domain,
                    "source": source,
                    "scheme": scheme,
                    "category": category,
                    "n_docs": len(group),
                    "n_clusters": group["author"].nunique() if cluster else len(group),
                    "mean_rate": rbar,
                    "V": V,
                    "rho": rho,
                    "rho_se": boot.std(ddof=1),
                    "rho_ci_low": np.percentile(boot, 2.5),
                    "rho_ci_high": np.percentile(boot, 97.5),
                }
            )

    return pd.DataFrame(results)


def _test(variance: pd.DataFrame) -> pd.DataFrame:
    """z-tests rho_human = rho_AI for every domain, comparison and category (step 9).

    z = (rho_h - rho_a) / sqrt(SE_h^2 + SE_a^2), with the SEs from each
    cell's resampled rho-hats; two-sided p, then BH-FDR across all 48 at once.

    Args:
        variance (pd.DataFrame): _estimate's variance table.

    Returns:
        pd.DataFrame: 48 rows -- domain, comparison, scheme, category,
            rho_human, rho_ai, rho_diff, rho_ratio, z, p_value, p_value_fdr,
            significant.
    """
    v = variance.set_index(["domain", "source", "scheme", "category"]).sort_index()
    rows = []
    for domain in v.index.get_level_values("domain").unique():
        for human_label, ai_label in COMPARISONS:
            for scheme, category in CATEGORIES.values():
                h = v.loc[(domain, human_label, scheme, category)]
                a = v.loc[(domain, ai_label, scheme, category)]

                delta = h["rho"] - a["rho"]
                se = np.sqrt(h["rho_se"] ** 2 + a["rho_se"] ** 2)
                z = delta / se
                p = 2 * stats.norm.sf(abs(z))
                ratio = h["rho"] / a["rho"] if a["rho"] > 0 else np.nan
                rows.append(
                    {
                        "domain": domain,
                        "comparison": f"{human_label}_vs_{ai_label}",
                        "scheme": scheme,
                        "category": category,
                        "rho_human": h["rho"],
                        "rho_ai": a["rho"],
                        "rho_diff": delta,
                        "rho_ratio": ratio,
                        "se_diff": se,
                        "z": z,
                        "p_value": p,
                    }
                )

    tests = pd.DataFrame(rows)
    tests = apply_fdr(tests, pvalue_col="p_value")
    tests = tests.rename(columns={"p_value_significant": "significant"})
    assert len(tests) == 48, f"expected 48 tests, got {len(tests)}"
    return tests


def _exclude_assessed(df: pd.DataFrame) -> pd.DataFrame:
    """Drops the documents in EXCLUDED_DOCS, for the sensitivity pass only.

    Each entry must match exactly one document, checked one by one, so a
    mistyped or ambiguous key fails loudly and names the document rather
    than letting the sensitivity pass silently remove the wrong number.
    essay and wp have no author, so an entry with author None matches a
    missing author via isna() -- a missing value never compares equal to
    anything, including None.

    Args:
        df (pd.DataFrame): Filtered doc_features (post filter_min_sents).

    Returns:
        pd.DataFrame: df without the six documents, with a fresh 0-based
            index so row positions line up for resampling.

    Raises:
        AssertionError: If any entry doesn't match exactly one row.
    """
    drop = np.zeros(len(df), dtype=bool)
    for domain, source, doc_id, author in EXCLUDED_DOCS:
        match = (
            (df["domain"] == domain)
            & (df["source"] == source)
            & (df["doc_id"] == doc_id)
        )
        match &= df["author"].isna() if author is None else (df["author"] == author)
        assert match.sum() == 1, (
            f"{domain}/{source}/{doc_id}/{author} "
            f"matches {match.sum()} rows, expected 1"
        )
        drop |= match.to_numpy()
    return df[~drop].reset_index(drop=True)


def _verdict(tests: pd.DataFrame) -> pd.Series:
    """Labels each test by its outcome, for the CSVs and the sensitivity comparison.

    A verdict needs both significance (after FDR) and a direction, so a
    result that stays significant but flips sign counts as a change.

    Args:
        tests (pd.DataFrame): _test's output.

    Returns:
        pd.Series: "human more variable", "AI more variable" or
            "no difference", aligned to tests' index.
    """
    return pd.Series(
        np.select(
            [
                tests["significant"] & (tests["rho_diff"] > 0),
                tests["significant"] & (tests["rho_diff"] < 0),
            ],
            ["human more variable", "AI more variable"],
            default="no difference",
        ),
        index=tests.index,
    )


def main():
    """Main pass, then the sensitivity pass; writes all three CSVs."""
    df = filter_min_sents(pd.read_feather(INPUT_PATH))

    variance = _estimate(df)
    assert len(variance) == 72, f"expected 72 rows, got {len(variance)}"
    tests = _test(variance)
    tests["verdict"] = _verdict(tests)

    VARIANCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    variance.to_csv(VARIANCE_OUTPUT, index=False)
    tests.to_csv(TESTS_OUTPUT, index=False)

    variance_excl = _estimate(_exclude_assessed(df))
    tests_excl = _test(variance_excl)
    tests_excl["verdict"] = _verdict(tests_excl)

    key = ["domain", "comparison", "scheme", "category"]
    keep = ["rho_diff", "p_value_fdr", "verdict"]
    sens = tests[key + keep].merge(
        tests_excl[key + keep],
        on=key,
        suffixes=("_main", "_excl"),
        validate="one_to_one",
    )
    sens["verdict_changed"] = sens["verdict_main"] != sens["verdict_excl"]
    sens.to_csv(SENSITIVITY_OUTPUT, index=False)

    print("main pass:", tests["verdict"].value_counts().to_dict())
    changed = sens[sens["verdict_changed"]]
    print(
        f"sensitivity: {len(changed)} of {len(sens)} "
        "verdicts changed without the six documents"
    )
    for row in changed.itertuples():
        print(
            f"  {row.domain} {row.comparison} {row.scheme} {row.category}: "
            f"{row.verdict_main} -> {row.verdict_excl}"
        )


if __name__ == "__main__":
    main()
