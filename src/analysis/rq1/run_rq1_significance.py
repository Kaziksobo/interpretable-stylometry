"""RQ1 Track 3 -- Per-Category Diagnostics: which categories drive Tracks 1 and 2.

Explanatory layer, not a third RQ1 claim. For each of the 8 Feng categories
(both `OTHER`s included), tests whether the per-document rate differs in
spread (Brown-Forsythe) and location (Mann-Whitney) between human and AI
prose, and reports variance ratio and CV as effect sizes.

See docs/rq1_methodology.md §6 for the full pipeline (Eq. 14-16) and §9.5
for the exact output schema.

Inputs:
    data/processed/rq1/doc_features.feather

Outputs:
    results/rq1/rq1_diagnostics_significance.csv   -- supporting, not the answer
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.shared.filters import filter_min_sents
from src.analysis.shared.significance_utils import apply_fdr, run_test_family

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "doc_features.feather"
DIAGNOSTICS_OUTPUT = (
    PROJECT_ROOT / "results" / "rq1" / "rq1_diagnostics_significance.csv"
)

COMPARISONS = [("human", "gpt"), ("human", "claude")]

# raw count column -> label used in the `feature` column (§9.5)
# sent_* = Feng Algorithm 1 (sentence type), struct_* = Algorithm 2 (structure)
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


def _process_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full Track 3 battery across all 8 categories (§6.1 steps 2-8).

    Converts every count column to a per-document rate against n_sents --
    deliberately not compute_proportions_algo1/2, since Track 3's
    denominator is n_sents for every category, including algo1's (§6
    denominator note), unlike Track 1/2's conforming-sentence-only
    denominator. Builds one cell per (domain, comparison, feature),
    running both a Brown-Forsythe (dispersion) and a Mann-Whitney
    (location) test on each -- two separate run_test_family calls, since
    a single cell needs two different tests here, unlike Tracks 1/2.
    Effect sizes (var_ratio, CV, cv_ratio) are collected in the same
    pass, along with a dispersion_direction_unstable flag wherever the
    two ratios disagree on which group is more variable (§6.1 step 8).
    FDR is applied here, once per test type, since this function sees
    all 48 cells at once.

    Args:
        df (pd.DataFrame): Filtered doc_features (post filter_min_sents).

    Returns:
        pd.DataFrame: One row per (domain, comparison, feature) cell --
            48 total -- with columns matching §9.5's schema.
    """
    # Compute rates for all 8 categories, on a copy of df
    rate_df = df.copy()
    for col in FEATURES.keys():
        rate_df[col] = rate_df[col] / rate_df["n_sents"]

    # Build cells
    cells = {}
    effect_sizes = {}
    for domain in rate_df["domain"].unique():
        domain_df = rate_df[rate_df["domain"] == domain]

        for human_label, ai_label in COMPARISONS:
            human_df = domain_df[domain_df["source"] == human_label]
            ai_df = domain_df[domain_df["source"] == ai_label]

            if len(human_df) == 0 or len(ai_df) == 0:
                print(
                    f"  WARNING: no data for {domain}/{human_label} or "
                    f"{domain}/{ai_label} - skipping."
                )
                continue

            for feat_col, feat_label in FEATURES.items():
                human_rates = human_df[feat_col].dropna().to_numpy()
                ai_rates = ai_df[feat_col].dropna().to_numpy()

                if len(human_rates) < 3 or len(ai_rates) < 3:
                    print(
                        f"  WARNING: not enough data for {domain}/{human_label} "
                        f"or {domain}/{ai_label} on {feat_label} - skipping."
                    )
                    continue

                comparison = f"{human_label}_vs_{ai_label}"
                key = (domain, comparison, feat_label)
                cells[key] = (human_rates, ai_rates)

                # Effect sizes for this cell
                mean_human = human_rates.mean()
                mean_ai = ai_rates.mean()
                std_human = human_rates.std(ddof=1)
                std_ai = ai_rates.std(ddof=1)

                # Eq. 15 - inf if AI variance is zero (every document identical)
                var_ratio = std_human**2 / std_ai**2 if std_ai > 0 else np.inf

                # Eq. 16 -- CV divides by the mean, so guard zero means
                cv_human = std_human / mean_human if mean_human > 0 else np.nan
                cv_ai = std_ai / mean_ai if mean_ai > 0 else np.nan
                cv_ratio = cv_human / cv_ai if cv_ai > 0 else np.nan

                # §6.1 step 8 -- the direction can't be trusted if either ratio
                # is non-finite, or if the two ratios point opposite ways
                ratios_finite = np.isfinite(var_ratio) and np.isfinite(cv_ratio)
                dispersion_direction_unstable = (not ratios_finite) or (
                    (var_ratio > 1) != (cv_ratio > 1)
                )

                effect_sizes[key] = {
                    "n_human": len(human_rates),
                    "n_ai": len(ai_rates),
                    "mean_human": mean_human,
                    "std_human": std_human,
                    "mean_ai": mean_ai,
                    "std_ai": std_ai,
                    "var_ratio": var_ratio,
                    "cv_human": cv_human,
                    "cv_ai": cv_ai,
                    "cv_ratio": cv_ratio,
                    "dispersion_direction_unstable": dispersion_direction_unstable,
                }

    assert len(cells) == 48, f"Expected 48 cells, got {len(cells)}"

    # Run the two test families
    disp_df = run_test_family(
        cells,
        partial(stats.levene, center="median"),
        key_names=["domain", "comparison", "feature"],
    )

    loc_df = run_test_family(
        cells, stats.mannwhitneyu, key_names=["domain", "comparison", "feature"]
    )

    # Rename to p_disp/p_loc before merging, to avoid suffixes
    disp_df = disp_df.rename(
        columns={"statistic": "disp_statistic", "p_value": "p_disp"}
    )
    loc_df = loc_df.rename(columns={"statistic": "loc_statistic", "p_value": "p_loc"})

    # Merge the two test results on the key columns
    merged_df = pd.merge(
        disp_df, loc_df, on=["domain", "comparison", "feature"], validate="one_to_one"
    )

    # Merge the effect sizes on the key columns
    effect_df = pd.DataFrame.from_dict(effect_sizes, orient="index")
    effect_df = effect_df.rename_axis(["domain", "comparison", "feature"]).reset_index()
    merged_df = pd.merge(
        merged_df,
        effect_df,
        on=["domain", "comparison", "feature"],
        validate="one_to_one",
    )

    assert len(merged_df) == 48, f"Expected 48 rows after merging, got {len(merged_df)}"

    # Apply FDR correction to both p-value columns
    merged_df = apply_fdr(merged_df, pvalue_col="p_disp")
    merged_df = apply_fdr(merged_df, pvalue_col="p_loc")

    # Rename the FDR-corrected columns to avoid confusion
    merged_df = merged_df.rename(
        columns={
            "p_disp_significant": "disp_significant",
            "p_loc_significant": "loc_significant",
        }
    )

    # Step 7: pure_regularisation = disp_significant & ~loc_significant
    merged_df["pure_regularisation"] = (
        merged_df["disp_significant"] & ~merged_df["loc_significant"]
    )

    return merged_df[
        [
            "feature",
            "domain",
            "comparison",
            "n_human",
            "n_ai",
            "mean_human",
            "std_human",
            "mean_ai",
            "std_ai",
            "var_ratio",
            "cv_human",
            "cv_ai",
            "cv_ratio",
            "dispersion_direction_unstable",
            "disp_statistic",
            "p_disp",
            "p_disp_fdr",
            "disp_significant",
            "loc_statistic",
            "p_loc",
            "p_loc_fdr",
            "loc_significant",
            "pure_regularisation",
        ]
    ]


def main():
    """Loads doc_features, filters once, and writes the diagnostics output."""
    df = filter_min_sents(pd.read_feather(INPUT_PATH))
    diagnostics = _process_features(df)
    diagnostics.to_csv(DIAGNOSTICS_OUTPUT, index=False)


if __name__ == "__main__":
    main()
