"""Rebuild constituency_features.feather with author field.

constituency_parses.feather (patched by fix_author_reuters.py) has the
correct author column. constituency_features.feather has feng_algo_1 and
feng_algo_2 but was built before the Reuters fix so lacks author.

This script merges the two on (doc_id, domain, source, sent_idx) to produce
a clean per-sentence feature file with all required columns.

parse_str is intentionally excluded from the output as it is large and no
longer needed once feng features have been extracted.

Output schema (one row per sentence):
    doc_id      : str   - document identifier
    domain      : str   - "essay" | "reuter" | "wp"
    source      : str   - "human" | "gpt" | "claude"
    author      : str   - Reuters author name; None for essay/wp
    sent_idx    : int   - 0-based sentence index within document
    sent_text   : str   - raw sentence text
    feng_algo_1 : str   - sentence type: SIMPLE | COMPLEX | COMPOUND | COMPLEX-COMPOUND
    feng_algo_2 : str   - sentence structure: LOOSE | PERIODIC | OTHER
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
PARSES_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_parses.feather"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_features.feather"

JOIN_COLS = ["doc_id", "domain", "source", "sent_idx", "sent_text"]
OUTPUT_COLS = [
    "doc_id",
    "domain",
    "source",
    "author",
    "sent_idx",
    "sent_text",
    "feng_algo_1",
    "feng_algo_2",
]


def main() -> None:
    print(f"Loading {PARSES_PATH.name}...")
    parses = pd.read_feather(PARSES_PATH, columns=JOIN_COLS + ["author"])
    print(f"  {len(parses):,} rows")

    print(f"Loading {FEATURES_PATH.name}...")
    features = pd.read_feather(
        FEATURES_PATH, columns=JOIN_COLS + ["feng_algo_1", "feng_algo_2"]
    )
    print(f"  {len(features):,} rows")

    print("Merging...")
    merged = features.merge(parses, on=JOIN_COLS, how="left")

    n_null_author = merged["author"].isna().sum()
    n_reuter = (merged["domain"] == "reuter").sum()
    n_non_reuter = len(merged) - n_reuter
    print(f"  Null author rows: {n_null_author:,}")
    print(f"    Expected ~{n_non_reuter:,} (essay/wp have no author by design)")
    print(
        f"    Unexpected reuter nulls: "
        f"{(merged['author'].isna() & (merged['domain'] == 'reuter')).sum():,} "
        f"(boilerplate collisions, acceptable if small)"
    )

    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=JOIN_COLS)
    if before_dedup != len(merged):
        print(
            f"  Dropped {before_dedup - len(merged):,} duplicate rows "
            f"(identical boilerplate text shared across Reuters authors)"
        )

    loss = len(features) - len(merged)
    loss_pct = 100 * loss / len(features)
    print(
        (
            f"  {loss:,} rows lost ({loss_pct:.2f}%) - "
            "unresolvable boilerplate collisions, will have null author."
        )
    )
    if loss_pct > 1.0:
        raise ValueError(
            f"Lost {loss_pct:.1f}% of rows, too many to be explained by boilerplate. "
            "Investigate join keys."
        )

    merged = merged[OUTPUT_COLS]

    print("\nSentence counts per domain/source:")
    print(merged.groupby(["domain", "source"]).size().to_string())

    print(f"\nSaving {len(merged):,} rows to {FEATURES_PATH.name}...")
    merged.to_feather(FEATURES_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
