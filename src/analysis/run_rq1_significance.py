"""RQ1 significance testing: variance and location differences in Feng features.

Tests whether AI-generated prose shows reduced variance (dispersion) and
shifted usage rates (location) in Feng et al. (2012) syntactic features,
compared to human prose.

Two tests are run for every (domain x feature x comparison) combination:

  Brown-Forsythe  -- tests dispersion (spread of per-document rates).
                     This is the primary RQ1 test. Uses Levene's test with
                     center='median' for robustness to skewed proportions.
                     scipy.stats.levene(a, b, center='median')

  Mann-Whitney U  -- tests location (median per-document rate).
                     Secondary test: distinguishes "pure regularisation"
                     (dispersion differs, location similar) from "rate shift
                     plus regularisation" (both differ).
                     scipy.stats.mannwhitneyu(a, b)

All p-values are FDR-corrected (Benjamini-Hochberg) across the full set
of tests to control the false discovery rate at 5%.

Inputs:
    data/processed/doc_features.feather  (built by build_doc_features.py)
    doc_features stores raw sentence COUNTS per document. This script
    converts them to per-document RATES (count / n_sents) before testing,
    because rates are the correct unit for comparing documents of different
    lengths.

Outputs:
    results/rq1_significance.csv  -- full results table, one row per test
    results/rq1_significance.txt  -- human-readable summary of findings
"""

from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "doc_features.feather"
OUTPUT_CSV = PROJECT_ROOT / "results" / "rq1_significance.csv"
OUTPUT_TXT = PROJECT_ROOT / "results" / "rq1_significance.txt"

DOMAINS = ["essay", "reuter", "wp"]
COMPARISONS = [("human", "gpt"), ("human", "claude")]

# Maps raw count column names (from doc_features.feather) to plain-English
# labels used in the summary report. These are the features being tested.
# sent_* = Feng Algorithm 1 (sentence type)
# struct_* = Feng Algorithm 2 (sentence structure)
FEATURES: dict[str, str] = {
    "sent_simple": "Sentence type: SIMPLE",
    "sent_complex": "Sentence type: COMPLEX",
    "sent_compound": "Sentence type: COMPOUND",
    "sent_complex_compound": "Sentence type: COMPLEX-COMPOUND",
    "sent_other": "Sentence type: OTHER",
    "struct_loose": "Sentence structure: LOOSE",
    "struct_periodic": "Sentence structure: PERIODIC",
    "struct_other": "Sentence structure: OTHER",
}


def load_and_normalise(path: Path) -> pd.DataFrame:
    """Load doc_features.feather and convert count columns to per-document rates.

    doc_features stores raw sentence counts, e.g. sent_complex = 7 means
    7 COMPLEX sentences in that document. For the statistical tests we need
    rates (proportions), because documents have different lengths, a document
    with 7 COMPLEX sentences out of 10 total is very different from one with
    7 out of 100. Dividing by n_sents normalises for document length.

    The rate columns replace the count columns in the returned DataFrame.
    n_sents is kept as a column so downstream code can inspect it if needed.
    """
    df = pd.read_feather(path)

    if missing := set(FEATURES.keys()) - set(df.columns):
        raise ValueError(
            f"doc_features.feather is missing expected columns: {missing}\n"
            "Has build_doc_features.py been run?"
        )

    if "n_sents" not in df.columns:
        raise ValueError("doc_features.feather is missing 'n_sents' column.")

    # Convert counts to rates in-place
    for col in FEATURES:
        df[col] = df[col] / df["n_sents"]

    return df


def run_tests(doc_features: pd.DataFrame) -> pd.DataFrame:
    """Run Brown-Forsythe and Mann-Whitney for every domain x feature x comparison.

    For each combination we extract two arrays:
        a = per-document rates for the human group
        b = per-document rates for the AI group (gpt or claude)

    Brown-Forsythe asks: does the SPREAD of these arrays differ?
        - Primary RQ1 test
        - var_ratio = human variance / AI variance
        - var_ratio > 1 means human is more variable (expected direction)

    Mann-Whitney asks: does the MEDIAN of these arrays differ?
        - Secondary test
        - Tells us whether the average usage rate differs, not just the spread
        - A result where dispersion is significant but location is not is the
          cleanest possible RQ1 finding: same average use, narrower spread

    Returns a DataFrame of raw (uncorrected) results, one row per test.
    """
    rows = []

    for domain in DOMAINS:
        domain_df = doc_features[doc_features["domain"] == domain]

        for ref_source, cmp_source in COMPARISONS:
            ref_df = domain_df[domain_df["source"] == ref_source]
            cmp_df = domain_df[domain_df["source"] == cmp_source]

            if len(ref_df) == 0 or len(cmp_df) == 0:
                print(
                    f"  WARNING: no data for {domain}/{ref_source} or "
                    f"{domain}/{cmp_source} - skipping."
                )
                continue

            for feat_col, feat_label in FEATURES.items():
                a = ref_df[feat_col].dropna()  # human rates
                b = cmp_df[feat_col].dropna()  # AI rates

                if len(a) < 3 or len(b) < 3:
                    print(
                        f"  WARNING: too few observations for "
                        f"{domain}/{feat_col}/{cmp_source} - skipping."
                    )
                    continue

                # Brown-Forsythe (Levene with center='median')
                # center='median' rather than 'mean' is what makes this
                # Brown-Forsythe specifically -- robust to skewed distributions,
                # which per-document proportions almost always are (especially
                # for rare categories like COMPLEX-COMPOUND where most docs
                # sit near 0 with a long right tail)
                _, p_disp = stats.levene(a, b, center="median")

                # Mann-Whitney U (two-sided)
                # alternative='two-sided' because we don't restrict the
                # direction of location differences in advance
                _, p_loc = stats.mannwhitneyu(a, b, alternative="two-sided")

                # Variance ratio as effect size for dispersion
                # Infinity if AI variance is zero (every document identical)
                var_ratio = a.var() / b.var() if b.var() > 0 else float("inf")

                rows.append(
                    {
                        "domain": domain,
                        "comparison": f"{ref_source}_vs_{cmp_source}",
                        "feature": feat_col,
                        "feature_label": feat_label,
                        "n_ref": len(a),
                        "n_cmp": len(b),
                        "ref_mean": round(a.mean(), 4),
                        "cmp_mean": round(b.mean(), 4),
                        "ref_std": round(a.std(), 4),
                        "cmp_std": round(b.std(), 4),
                        "var_ratio": round(var_ratio, 3),
                        "p_dispersion": p_disp,
                        "p_location": p_loc,
                    }
                )

    return pd.DataFrame(rows)


def apply_fdr_correction(results: pd.DataFrame) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction across all tests.

    We're running many tests simultaneously (8 features x 3 domains x
    2 comparisons = 48 tests). At p<0.05, roughly 2-3 of these would
    appear significant by chance alone even if nothing real is happening.

    Benjamini-Hochberg controls the False Discovery Rate: the expected
    proportion of significant results that are false positives. At q<0.05,
    no more than 5% of our significant findings should be noise.

    Correction is applied separately to dispersion and location p-values,
    across all 48 tests in each set, not per-domain, not per-feature.
    Correcting across the full set is the conservative and correct choice.
    """
    _, p_disp_fdr, _, _ = multipletests(
        results["p_dispersion"], method="fdr_bh", alpha=0.05
    )
    _, p_loc_fdr, _, _ = multipletests(
        results["p_location"], method="fdr_bh", alpha=0.05
    )

    results = results.copy()
    results["p_dispersion_fdr"] = p_disp_fdr
    results["p_location_fdr"] = p_loc_fdr
    results["disp_significant"] = p_disp_fdr < 0.05
    results["loc_significant"] = p_loc_fdr < 0.05
    return results


def write_summary(results: pd.DataFrame, path: Path) -> None:
    """Write a plain-English summary of findings to a text file."""
    sig_disp = results[results["disp_significant"]].sort_values("p_dispersion_fdr")
    not_sig = results[~results["disp_significant"]].sort_values("p_dispersion_fdr")

    lines = [
        "=== RQ1 SIGNIFICANCE TESTING: FENG ET AL. FEATURES ===\n",
        f"Total tests run: {len(results)}",
        (
            "Significant on DISPERSION (Brown-Forsythe, FDR q<0.05): "
            f"{len(sig_disp)}/{len(results)}"
        ),
        (
            "Significant on LOCATION (Mann-Whitney, FDR q<0.05): "
            f"{results['loc_significant'].sum()}/{len(results)}\n"
        ),
        "=" * 70,
        "DISPERSION RESULTS (primary RQ1 test)",
        "var_ratio = human variance / AI variance",
        "  > 1  human more variable than AI  (supports RQ1)",
        "  < 1  AI more variable than human  (contradicts RQ1)",
        "=" * 70,
    ]
    if len(sig_disp) == 0:
        lines.extend(
            (
                "\nNo features survive FDR correction on dispersion.",
                "Feng features do not show significant regularisation "
                "at the document level.\n",
            )
        )
    else:
        for _, row in sig_disp.iterrows():
            lines.extend(
                (
                    (
                        f"\n[{row['domain'].upper()}] "
                        f"{row['feature_label']} | {row['comparison']}"
                    ),
                    (
                        f"  human : mean={row['ref_mean']:.3f}  "
                        f"std={row['ref_std']:.3f}  (n={row['n_ref']})"
                    ),
                    (
                        f"  AI    : mean={row['cmp_mean']:.3f}  "
                        f"std={row['cmp_std']:.3f}  (n={row['n_cmp']})"
                    ),
                    (
                        f"  var_ratio={row['var_ratio']:.2f}x  "
                        f"p_disp={row['p_dispersion']:.2e}  "
                        f"p_disp_fdr={row['p_dispersion_fdr']:.2e}  "
                        f"p_loc_fdr={row['p_location_fdr']:.2e}"
                    ),
                )
            )
            # Flag the cleanest possible RQ1 result
            if row["disp_significant"] and not row["loc_significant"]:
                lines.append(
                    "  *** PURE REGULARISATION: spread differs, "
                    "average rate does not ***"
                )

    lines.extend(
        (
            "\n" + "=" * 70,
            "NON-SIGNIFICANT ON DISPERSION (after FDR correction)",
            "=" * 70,
        )
    )
    lines.extend(
        (
            f"  [{row['domain'].upper()}] {row['feature_label']} | "
            f"{row['comparison']}  var_ratio={row['var_ratio']:.2f}x  "
            f"p_disp_fdr={row['p_dispersion_fdr']:.3f}"
        )
        for _, row in not_sig.iterrows()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"Loading and normalising {INPUT_PATH.name}...")
    doc_features = load_and_normalise(INPUT_PATH)
    print(f"  {len(doc_features):,} documents")
    print(f"  Domains: {sorted(doc_features['domain'].unique())}")
    print(f"  Sources: {sorted(doc_features['source'].unique())}\n")

    print("Running tests...")
    results = run_tests(doc_features)
    print(f"  {len(results)} tests completed\n")

    print("Applying FDR correction (Benjamini-Hochberg)...")
    results = apply_fdr_correction(results)

    sig_disp = results["disp_significant"].sum()
    sig_loc = results["loc_significant"].sum()
    print(f"  Significant on dispersion: {sig_disp}/{len(results)}")
    print(f"  Significant on location:   {sig_loc}/{len(results)}\n")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"Full results saved to {OUTPUT_CSV.name}")

    write_summary(results, OUTPUT_TXT)
    print(f"Summary saved to {OUTPUT_TXT.name}")

    # Print headline dispersion findings directly to terminal
    sig = results[results["disp_significant"]].sort_values("p_dispersion_fdr")
    if len(sig) > 0:
        print("\n--- HEADLINE FINDINGS (dispersion, FDR-corrected) ---")
        for _, row in sig.iterrows():
            pure = row["disp_significant"] and not row["loc_significant"]
            tag = " [PURE REGULARISATION]" if pure else ""
            print(
                f"  [{row['domain'].upper()}] {row['feature_label']} "
                f"| {row['comparison']} "
                f"| var_ratio={row['var_ratio']:.2f}x "
                f"| p_fdr={row['p_dispersion_fdr']:.2e}"
                f"{tag}"
            )
    else:
        print("\nNo features survive FDR correction on dispersion.")


if __name__ == "__main__":
    main()
