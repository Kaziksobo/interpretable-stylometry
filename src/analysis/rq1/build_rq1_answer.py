"""Synthesises the three RQ1 significance tables into one plain-English answer.

Reads rq1_variance_significance.csv, rq1_regularity_significance.csv, and
rq1_diagnostics_significance.csv only -- no doc_features.feather dependency,
no compositional maths, pure pandas + string formatting.

See docs/rq1_methodology.md §9.7 for the generation logic (the
significant x ratio-direction verdict table, and the diagnostic-row
selection rule).

Inputs:
    results/rq1/rq1_variance_significance.csv
    results/rq1/rq1_regularity_significance.csv
    results/rq1/rq1_diagnostics_significance.csv

Outputs:
    results/rq1/rq1_answer.txt
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
VARIANCE_INPUT = PROJECT_ROOT / "results" / "rq1" / "rq1_variance_significance.csv"
REGULARITY_INPUT = PROJECT_ROOT / "results" / "rq1" / "rq1_regularity_significance.csv"
DIAGNOSTICS_INPUT = (
    PROJECT_ROOT / "results" / "rq1" / "rq1_diagnostics_significance.csv"
)
ANSWER_OUTPUT = PROJECT_ROOT / "results" / "rq1" / "rq1_answer.txt"

COMPARISONS = [("human", "gpt"), ("human", "claude")]
# explicit order: alphabetical would put sentence_structure first
COMPOSITIONS = ["sentence_type", "sentence_structure"]

SUPPORTS = "SUPPORTS RQ1"
OPPOSITE = "OPPOSITE DIRECTION"
NO_EFFECT = "NO DETECTABLE EFFECT"

UNITS = """\
UNITS AND HOW TO READ THIS
  Ratios are always human / AI, so a ratio above 1 means the human value is larger.
  Variance (Track 1): distance from each document to the centre of its own group,
      in Aitchison (log-ratio) units. Ratio > 1 = human documents are more spread
      out than AI ones.
  Regularity (Track 2): Shannon entropy in nats (Miller-Madow corrected), per
      document. Ratio > 1 = human documents spread more evenly across sentence
      types than AI ones, i.e. AI is more regular.
  Per-category rates (Track 3): share of a document's sentences, shown as percentages.
  A verdict needs BOTH a significant FDR-corrected p-value (alpha 0.05) and a
  direction. With ~1,000 documents per group a tiny ratio can still be
  significant, so read the ratio, not just the verdict."""


def _verdict(significant: bool, ratio: float) -> str:
    """Applies the §9 reading rule to one test result.

    A result only supports or contradicts RQ1 if it cleared FDR
    correction -- the ratio's direction is meaningless on its own if the
    difference wasn't statistically significant.

    Args:
        significant (bool): Whether the FDR-corrected p-value cleared
            alpha = 0.05.
        ratio (float): The human/AI ratio for this test (distance,
            entropy, etc.). Ignored if significant is False.

    Returns:
        str: SUPPORTS ("significant, ratio > 1"), OPPOSITE ("significant,
            ratio < 1"), or NO_EFFECT ("not significant").

    Raises:
        ValueError: If significant is True and ratio is exactly 1.0, an
            outcome this pipeline's tests shouldn't be able to produce.
    """
    if not significant:
        return NO_EFFECT
    if ratio > 1:
        return SUPPORTS
    if ratio < 1:
        return OPPOSITE
    raise ValueError(f"Significant ratio of exactly 1.0 is unexpected: {ratio}")


def _one_row(df: pd.DataFrame, **conditions) -> pd.Series:
    """Returns the single row of df matching every keyword condition.

    Args:
        df (pd.DataFrame): The table to filter.
        **conditions: Column-value pairs to filter on, e.g. domain="wp",
            comparison="human_vs_gpt", composition="sentence_type".

    Returns:
        pd.Series: The one matching row.

    Raises:
        ValueError: If zero rows or more than one row match -- a silent
            first-match here would misreport an entire cell.
    """
    filtered = df
    for col, val in conditions.items():
        filtered = filtered[filtered[col] == val]
    if len(filtered) != 1:
        raise ValueError(
            f"Expected exactly one row for {conditions}, got {len(filtered)}"
        )
    return filtered.iloc[0]


def _strongest_row(
    cell: pd.DataFrame, p_fdr_col: str, p_raw_col: str, sig_col: str, stable_only: bool
) -> pd.Series | None:
    """Picks the best-supported diagnostics row for one cell (§9.7).

    Keeps only significant rows (and, if stable_only, only rows where
    dispersion_direction_unstable is False), then sorts by the
    FDR-corrected p-value. Raw p-value breaks ties, since FDR values do
    tie in this data.

    Args:
        cell (pd.DataFrame): One (domain, comparison)'s 8 diagnostics
            rows.
        p_fdr_col (str): Column holding the FDR-corrected p-value, e.g.
            "p_loc_fdr".
        p_raw_col (str): Column holding the raw p-value, used as the
            tie-break.
        sig_col (str): Boolean column marking significance, e.g.
            "loc_significant".
        stable_only (bool): If True, also excludes rows flagged
            dispersion_direction_unstable.

    Returns:
        pd.Series | None: The best-supported row, or None if nothing in
            the cell qualifies.
    """
    # Filter to the rows that are significant and (if requested) stable
    filtered = cell[cell[sig_col]]
    if stable_only:
        filtered = filtered[~filtered["dispersion_direction_unstable"]]
    if len(filtered) == 0:
        return None

    # Sort by FDR-corrected p-value, then raw p-value
    sorted_rows = filtered.sort_values(by=[p_fdr_col, p_raw_col], ascending=True)
    return sorted_rows.iloc[0]


def _format_cell(domain, comparison, variance, regularity, diagnostics) -> str:
    """Builds one (domain, comparison) block of the report.

    Prints a verdict line for each composition in both the variance and
    regularity tables, then the strongest supporting diagnostics rows
    for location and dispersion. A pick that doesn't qualify (nothing
    significant in the cell) is printed as "none significant" rather
    than silently dropped, so a reader can tell a genuine absence of
    evidence from a missing line.

    Args:
        domain (str): e.g. "wp".
        comparison (str): e.g. "human_vs_gpt".
        variance (pd.DataFrame): rq1_variance_significance.csv, loaded.
        regularity (pd.DataFrame): rq1_regularity_significance.csv,
            loaded.
        diagnostics (pd.DataFrame): rq1_diagnostics_significance.csv,
            loaded.

    Returns:
        str: The block's text, title line first, no trailing newline.

    Raises:
        AssertionError: If diagnostics doesn't have exactly 8 rows for
            this (domain, comparison).
    """
    title = f"{domain.upper()}: {comparison.replace('_', ' ')}"
    lines = [title]

    tables = [
        ("Variance", variance, "dist_ratio", "mean_dist_human", "mean_dist_ai"),
        (
            "Regularity",
            regularity,
            "entropy_ratio",
            "mean_entropy_human",
            "mean_entropy_ai",
        ),
    ]

    for label, table, ratio_col, human_col, ai_col in tables:
        for comp in COMPOSITIONS:
            row = _one_row(
                table, domain=domain, comparison=comparison, composition=comp
            )
            verdict = _verdict(row["significant"], row[ratio_col])
            ratio = row[ratio_col]
            h = row[human_col]
            a = row[ai_col]
            p = row["p_value_fdr"]
            line = (
                f"  {label:<11}{comp:<20}{verdict:<22}ratio {ratio:.3f} "
                f"(human {h:.3f} vs AI {a:.3f})  p_fdr {p:.1e}"
            )
            lines.append(line)

    # Filter diagnostics to the cell (domain, comparison)
    cell = diagnostics[
        (diagnostics["domain"] == domain) & (diagnostics["comparison"] == comparison)
    ]
    assert len(cell) == 8, (
        f"Expected 8 rows for diagnostics {domain} {comparison}, got {len(cell)}"
    )

    location_pick = _strongest_row(
        cell, "p_loc_fdr", "p_loc", "loc_significant", stable_only=False
    )
    if location_pick is None:
        lines.append("  Strongest location difference:   none significant")
    else:
        feature = location_pick["feature"]
        h = location_pick["mean_human"]
        a = location_pick["mean_ai"]
        p = location_pick["p_loc_fdr"]
        lines.append(
            f"  Strongest location difference:   {feature}"
            f"  human {h:.1%} vs AI {a:.1%} of sentences"
            f"  p_fdr {p:.1e}"
        )

    dispersion_pick = _strongest_row(
        cell, "p_disp_fdr", "p_disp", "disp_significant", stable_only=True
    )
    if dispersion_pick is None:
        lines.append("  Strongest dispersion difference: none significant")
    else:
        feature = dispersion_pick["feature"]
        v = dispersion_pick["var_ratio"]
        c = dispersion_pick["cv_ratio"]
        p = dispersion_pick["p_disp_fdr"]
        who = "human more variable" if v > 1 else "AI more variable"
        lines.append(
            f"  Strongest dispersion difference: {feature}"
            f"  variance ratio {v:.3f}, CV ratio {c:.3f} ({who})"
            f"  p_fdr {p:.1e}"
        )

    return "\n".join(lines)


def _format_header(variance, regularity, diagnostics) -> str:
    """Builds the report's summary block, including Paul's dispersion question.

    Reuses _verdict for the Track 1/2 counts, so the header can't
    disagree with the per-cell blocks below it. Also reports what share
    of Track 3's significant dispersion tests find the human group more
    variable, by raw variance ratio and by CV ratio separately -- the
    two disagree substantially in this data (76% vs. 36%), which is
    itself evidence for why the CV correction mattered.

    Args:
        variance (pd.DataFrame): rq1_variance_significance.csv, loaded.
        regularity (pd.DataFrame): rq1_regularity_significance.csv,
            loaded.
        diagnostics (pd.DataFrame): rq1_diagnostics_significance.csv,
            loaded.

    Returns:
        str: The header's text, ending with the units legend.
    """
    title = (
        "RQ1 ANSWER: do AI and human prose differ in syntactic variance and regularity?"
    )
    subtitle = "SUMMARY (FDR-corrected, alpha 0.05)"
    lines = [title, "", subtitle]

    tracks = [
        ("Track 1 variance:    ", variance, "dist_ratio"),
        ("Track 2 regularity:  ", regularity, "entropy_ratio"),
    ]

    for label, table, ratio_col in tracks:
        verdicts = [
            _verdict(row["significant"], row[ratio_col]) for _, row in table.iterrows()
        ]
        n_significant = verdicts.count(SUPPORTS) + verdicts.count(OPPOSITE)
        lines.append(
            f"  {label:<22}{n_significant} of {len(table)} significant  "
            f"({verdicts.count(SUPPORTS)} support RQ1, "
            f"{verdicts.count(OPPOSITE)} opposite, "
            f"{verdicts.count(NO_EFFECT)} not significant)"
        )

    # Track 3 diagnostics
    n_disp_sig = sum(diagnostics["disp_significant"])
    n_loc_sig = sum(diagnostics["loc_significant"])
    n_unstable = sum(diagnostics["dispersion_direction_unstable"])
    total_rows = len(diagnostics)

    label = "Track 3 per-category:"
    lines.append(
        f"  {label:<22}{n_disp_sig} of {total_rows} significant on dispersion, "
        f"{n_loc_sig} of {total_rows} on location;"
    )

    blank = ""
    lines.append(
        f"  {blank:<22}{n_unstable} of {total_rows} rows flagged "
        "dispersion_direction_unstable"
    )

    sig = diagnostics[diagnostics["disp_significant"]]
    stable = sig[~sig["dispersion_direction_unstable"]]
    by_var = sig["var_ratio"] > 1
    by_cv = sig["cv_ratio"] > 1
    by_var_stable = stable["var_ratio"] > 1

    lines.extend(
        (
            "",
            (
                f"  Of the {len(sig)} significant dispersion tests, "
                "the human group is the more variable one in:"
            ),
            f"    {by_var.sum()} ({by_var.mean():.0%}) by variance ratio",
            f"    {by_cv.sum()} ({by_cv.mean():.0%}) by CV ratio",
            (
                f"    {by_var_stable.sum()} of {len(stable)} "
                f"({by_var_stable.mean():.0%}) among the stable rows only"
            ),
            "",
            UNITS,
        )
    )
    return "\n".join(lines)


def main():
    """Loads the three tables, checks their shape, and writes the report."""
    variance = pd.read_csv(VARIANCE_INPUT)
    regularity = pd.read_csv(REGULARITY_INPUT)
    diagnostics = pd.read_csv(DIAGNOSTICS_INPUT)
    assert (len(variance), len(regularity), len(diagnostics)) == (12, 12, 48), (
        "unexpected row counts -- did a track's output change shape?"
    )

    parts = [_format_header(variance, regularity, diagnostics)]
    for domain in sorted(variance["domain"].unique()):
        for human_label, ai_label in COMPARISONS:
            comparison = f"{human_label}_vs_{ai_label}"
            parts.append(
                _format_cell(domain, comparison, variance, regularity, diagnostics)
            )
    ANSWER_OUTPUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
