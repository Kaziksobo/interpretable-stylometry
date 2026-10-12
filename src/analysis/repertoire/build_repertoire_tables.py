"""Builds the repertoire results tables as one markdown file.

Reads the results CSVs written by run_repertoire.py --all (the main run and the
two sensitivity runs) and writes three tables to one markdown file, each with a
heading, an explanation of what it is, the table and its notes. Every number
comes from the CSVs, so re-running run_repertoire.py --all and then this script
updates the whole file; only the explanations are written here.

Inputs:
    results/repertoire/repertoire_results.csv
    results/repertoire/repertoire_results_depth2_occurrences.csv
    results/repertoire/repertoire_results_depth3_tokens.csv

Outputs:
    results/repertoire/repertoire_tables.md

Usage (from the project root):
    python src/analysis/repertoire/run_repertoire.py --all
    python src/analysis/repertoire/build_repertoire_tables.py
"""

from pathlib import Path

import pandas as pd

from src.analysis.shared.plotting import DOMAIN_TITLES

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
RESULTS_DIR = PROJECT_ROOT / "results" / "repertoire"
OUTPUT_PATH = RESULTS_DIR / "repertoire_tables.md"

SOURCES = ["human", "gpt", "claude"]
MEASURES = {
    "types": "Distinct blocks used (`types`)",
    "types_min5": "Blocks used 5+ times (`types_min5`)",
    "effective_types": "Effective number of blocks (`effective_types`)",
    "cover_50": "Share taken by the 50 commonest (`cover_50`)",
}
MINUS = "\u2212"
DAGGER = "\u2020"

# the three runs: file suffix, pattern depth, what the budget measured
MAIN_RUN = ("", 2, "tokens")
OCCURRENCES_RUN = ("_depth2_occurrences", 2, "occurrences")
DEPTH3_RUN = ("_depth3_tokens", 3, "tokens")


def load_results(run: tuple[str, int, str]) -> pd.DataFrame:
    """Loads one run's results CSV and checks it is the run that was asked for.

    Args:
        run (tuple[str, int, str]): File suffix, pattern depth and budget type.

    Returns:
        pd.DataFrame: The results, indexed by (domain, metric, source).

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the CSV was produced with a different depth or budget.
    """
    suffix, depth, budget_by = run
    path = RESULTS_DIR / f"repertoire_results{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found: run run_repertoire.py --all first."
        )
    results = pd.read_csv(path)
    if not (
        (results["depth"] == depth).all() and (results["budget_by"] == budget_by).all()
    ):
        raise ValueError(f"{path.name} is not a depth {depth}, {budget_by}-budget run.")
    return results.set_index(["domain", "metric", "source"])


def signed(x: float, decimals: int) -> str:
    """Formats a number with an explicit sign, using a proper minus sign."""
    return f"{MINUS if x < 0 else '+'}{abs(x):.{decimals}f}"


def cell_text(results: pd.DataFrame, domain: str, metric: str, source: str) -> str:
    """Formats one table cell.

    A human cell is "value ± margin". An AI cell is "value ± margin", then a
    dagger if the difference from human is not significant, then the change
    from human in brackets: relative (%) for counts, percentage points for the
    share, which is already a percentage.

    Args:
        results (pd.DataFrame): One run's results, from load_results.
        domain (str): "essay", "reuter" or "wp".
        metric (str): A key of MEASURES.
        source (str): "human", "gpt" or "claude".

    Returns:
        str: The cell text.
    """
    row = results.loc[(domain, metric, source)]
    if metric == "cover_50":
        text = f"{100 * row['value']:.1f}% \u00b1 {100 * row['margin']:.1f}"
    else:
        text = f"{row['value']:,.0f} \u00b1 {row['margin']:,.0f}"
    if source == "human":
        return text
    if row["significant"] != 1:
        text += DAGGER
    if metric == "cover_50":
        change = f"{signed(100 * row['difference'], 1)} pts"
    else:
        change = f"{signed(100 * row['relative_difference'], 0)}%"
    return f"{text} ({change})"


def table_markdown(results: pd.DataFrame) -> str:
    """Builds one markdown table: three domains x four measures.

    Args:
        results (pd.DataFrame): One run's results, from load_results.

    Returns:
        str: The table.
    """
    lines = ["| Domain | Measure | Human | GPT | Claude |", "|---|---|---|---|---|"]
    for domain, title in DOMAIN_TITLES.items():
        for i, (metric, label) in enumerate(MEASURES.items()):
            cells = [cell_text(results, domain, metric, s) for s in SOURCES]
            lines.append(
                f"| {title if i == 0 else ''} | {label} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines)


def budget_text(results: pd.DataFrame, unit: str) -> str:
    """Describes each domain's budget, e.g. "413,819 tokens in Student essays, ...".

    Args:
        results (pd.DataFrame): One run's results, from load_results.
        unit (str): What the budget counts, e.g. "tokens".

    Returns:
        str: The budgets of all domains in one phrase.
    """
    budgets = results.reset_index().groupby("domain")["budget"].first()
    parts = [f"{budgets[d]:,.0f} {unit} in {t}" for d, t in DOMAIN_TITLES.items()]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def n_draws(results: pd.DataFrame) -> int:
    """Returns the number of random draws per source in a run."""
    return int(results["n_draws"].iloc[0])


def effective_types_note(results: pd.DataFrame) -> str:
    """Writes the definition of the effective number of blocks.

    A one-sentence definition first, then a longer paragraph, using the main
    run's own numbers in the example.

    Args:
        results (pd.DataFrame): The main run's results.

    Returns:
        str: The definition, as two markdown paragraphs.
    """
    types = results.loc[("essay", "types", "human"), "value"]
    effective = results.loc[("essay", "effective_types", "human"), "value"]
    return (
        "**Effective number of blocks (`effective_types`):** the number of equally "
        "common building blocks that would give the same variety as the real, uneven "
        "usage.\n\n"
        "A raw count of distinct blocks treats a block used once the same as a block "
        "used thousands of times. The effective number weights each block by how much "
        "it is actually used, so a block used once adds very little and a block used "
        "thousands of times adds a lot. For example, usage counts of 50, 30, 10, 5 and "
        "5 are five blocks, but two of them account for 80% of the use, so the "
        f"effective number is 3.45. In these data human essays use {types:,.0f} "
        f"different blocks, yet their effective number is {effective:,.0f}: their "
        f"variety is comparable to {effective:,.0f} equally used blocks, because a few "
        "common blocks dominate. It is a measure of the toolbox in practice and not "
        "its full reach, it is less affected by one-off blocks than the raw count, and "
        "it falls when usage concentrates on fewer blocks. Technically it is "
        "`exp(H)`, where `H = -sum(p ln p)` is the Shannon entropy of each block's "
        "share *p* of all use (a Hill number of order 1)."
    )


def other_notes(results: pd.DataFrame, extra: tuple[str, ...] = ()) -> str:
    """Writes the notes shared by every table.

    Args:
        results (pd.DataFrame): The run's results (for the number of tests).
        extra (tuple[str, ...]): Further notes specific to one table.

    Returns:
        str: A bulleted list of notes.
    """
    n_tests = int(results["difference"].notna().sum())
    items = [
        "**\u00b1** is a margin of error: 1.96 \u00d7 the standard error, about a "
        "95% range. The dagger sits next to it because it says the difference is "
        "within that noise.",
        f"**Percentages** in brackets are each AI value relative to human, so "
        f"({MINUS}35%) means 35% below human. `cover_50` is already a percentage, so "
        "its difference is in percentage points (pts).",
        "**Direction:** a lower value on the first three measures means a narrower "
        "repertoire, and a higher `cover_50` means more of the text rests on the "
        "commonest blocks, which is also narrower.",
        "**Significance** uses Benjamini\u2013Hochberg correction across the "
        f"{n_tests} comparisons in the table.",
        *extra,
    ]
    return "Other notes:\n\n" + "\n".join(f"- {item}" for item in items)


def section(heading: str, explainer: str, results: pd.DataFrame, notes: list) -> str:
    """Assembles one section: heading, explainer, table and notes."""
    return "\n\n".join([f"## {heading}", explainer, table_markdown(results), *notes])


def main() -> None:
    """Builds repertoire_tables.md from the three runs' results CSVs."""
    main_run = load_results(MAIN_RUN)
    occurrences = load_results(OCCURRENCES_RUN)
    depth3 = load_results(DEPTH3_RUN)
    dagger_note = f"{DAGGER} Non-significant results (corrected p \u2265 0.05)."
    see_table_1 = "The measures are defined under Table 1."

    sections = [
        section(
            "Table 1. Main result: each source at the same amount of text",
            "This is the headline comparison. Human, GPT and Claude are each compared "
            f"at the same amount of text ({budget_text(main_run, 'tokens')}), so that "
            "no source is credited with more variety just because more of its text "
            "was read. The building blocks are the smallest kind: a phrase and its "
            "immediate parts, such as `(NP (DT) (NN))`, a noun phrase made of a "
            f"determiner and a noun. Documents were drawn at random, whole, "
            f"{n_draws(main_run):,} times per source, and each value is the average "
            "over the draws. Reuters draws are by whole author.",
            main_run,
            [dagger_note, effective_types_note(main_run), other_notes(main_run)],
        ),
        section(
            "Table 2. Check: the same number of building blocks, not the same amount "
            "of text",
            "Table 1 gives every source the same amount of text. This check instead "
            "gives every source the same number of building blocks "
            f"({budget_text(occurrences, 'block occurrences')}), and everything else "
            "is unchanged. It is needed because a source could look narrower simply "
            "by building fewer blocks per word, which equal text would not reveal.",
            occurrences,
            [dagger_note, other_notes(occurrences, extra=(see_table_1,))],
        ),
        section(
            "Table 3. Check: larger building blocks (depth 3)",
            "Tables 1 and 2 use the smallest building blocks. This check uses larger "
            "ones: a phrase, its parts, and the parts of those parts, such as "
            "`(VP (VBD) (NP (DT) (NN)))`. It is needed because a source could be "
            "narrow in its simplest constructions but not in its more elaborate ones, "
            "or the reverse. Larger blocks are far more varied, so almost every one "
            "is rare: counts are much larger, few blocks reach five uses, and the "
            "share taken by the 50 commonest is much smaller. The sizes are "
            "therefore not comparable with Table 1, only the direction of the "
            "differences. It uses the same amount of text as Table 1 "
            f"({budget_text(depth3, 'tokens')}).",
            depth3,
            [
                dagger_note,
                other_notes(
                    depth3,
                    extra=(
                        see_table_1,
                        "Compare directions, not sizes, with Table 1.",
                    ),
                ),
            ],
        ),
    ]

    header = (
        "# Repertoire diversity and concentration: results\n\n"
        "Generated by `build_repertoire_tables.py` from the CSVs in this folder; do "
        "not edit by hand.\n\n"
        "Human is compared with GPT and with Claude on how varied the set of "
        "syntactic building blocks is and how concentrated its use is. Table 1 is "
        "the main result; Tables 2 and 3 are checks on it."
    )
    OUTPUT_PATH.write_text("\n\n".join([header, *sections]) + "\n", encoding="utf-8")
    print(f"Saved {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
