"""Build per-document feature table from constituency_features.feather.

Aggregates sentence-level Feng classifications into document-level counts.
Each output row represents one document. Category columns contain raw sentence
counts (not proportions) - divide by n_sents in downstream analysis to get
rates when needed.

Requires constituency_features.feather to have an 'author' column.
Run build_constituency_features.py first if it does not.

Output schema (one row per document):
    doc_id               : str - document identifier
    domain               : str - "essay" | "reuter" | "wp"
    source               : str - "human" | "gpt" | "claude"
    author               : str - Reuters author name; None for essay/wp
    n_sents              : int - total sentences in document
    sent_simple          : int - SIMPLE sentences
    sent_complex         : int - COMPLEX sentences
    sent_compound        : int - COMPOUND sentences
    sent_complex_compound: int - COMPLEX-COMPOUND sentences
    struct_loose         : int - LOOSE sentences
    struct_periodic      : int - PERIODIC sentences
    struct_other         : int - OTHER (structure)
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_features.feather"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "doc_features.feather"

GROUP_COLS = ["doc_id", "domain", "source", "author"]


def main() -> None:
    print(f"Loading {INPUT_PATH.name}...")
    df = pd.read_feather(INPUT_PATH)
    print(f"  {len(df):,} sentences")

    if "author" not in df.columns:
        raise ValueError(
            "'author' column missing from constituency_features.feather.\n"
            "Run build_constituency_features.py first."
        )

    # Replace None author with placeholder so groupby doesn't drop essay/wp rows
    df["author"] = df["author"].fillna("__none__")

    # Use get_dummies + groupby sum - avoids type inference issues from
    # value_counts().unstack() and produces integer counts directly.
    #
    # get_dummies produces one binary column per category label, e.g.:
    #   feng_algo_1_SIMPLE, feng_algo_1_COMPLEX, ...
    # groupby sum then counts how many sentences in each doc hit each category.
    algo1_dummies = pd.get_dummies(df["feng_algo_1"], prefix="algo1")
    algo2_dummies = pd.get_dummies(df["feng_algo_2"], prefix="algo2")

    combined = pd.concat([df[GROUP_COLS], algo1_dummies, algo2_dummies], axis=1)
    doc_df = combined.groupby(GROUP_COLS).sum().reset_index()

    # n_sents = sum of all algo1 categories (mutually exclusive and exhaustive)
    algo1_cols = [c for c in doc_df.columns if c.startswith("algo1_")]
    doc_df.insert(4, "n_sents", doc_df[algo1_cols].sum(axis=1).astype(int))

    # Rename to final output column names
    rename_map: dict[str, str] = {}
    for col in doc_df.columns:
        if col.startswith("algo1_"):
            cat = col[len("algo1_") :].lower().replace("-", "_")
            rename_map[col] = f"sent_{cat}"
        elif col.startswith("algo2_"):
            cat = col[len("algo2_") :].lower()
            rename_map[col] = f"struct_{cat}"
    doc_df = doc_df.rename(columns=rename_map)

    # Enforce expected columns exist and are in the right order
    expected_cols = [
        "doc_id",
        "domain",
        "source",
        "author",
        "n_sents",
        "sent_simple",
        "sent_complex",
        "sent_compound",
        "sent_complex_compound",
        "sent_other",
        "struct_loose",
        "struct_periodic",
        "struct_other",
    ]
    if missing_cols := [c for c in expected_cols if c not in doc_df.columns]:
        raise ValueError(
            f"Expected output columns missing: {missing_cols}\n"
            f"Columns present: {list(doc_df.columns)}\n"
            "Check that feng_algo_1 and feng_algo_2 contain the "
            "expected category labels."
        )
    doc_df = doc_df[expected_cols]

    # Sanity checks
    if (doc_df["n_sents"] == 0).any():
        raise ValueError(
            "Some documents have 0 sentences - check constituency_features.feather."
        )

    row_sums = doc_df[
        [
            "sent_simple",
            "sent_complex",
            "sent_compound",
            "sent_complex_compound",
            "sent_other",
        ]
    ].sum(axis=1)
    if not (row_sums == doc_df["n_sents"]).all():
        raise ValueError(
            "Sentence type counts do not sum to n_sents for all documents. "
            "Unexpected category values in feng_algo_1."
        )

    # Restore None author for essay/wp (cleaner than storing "__none__" on disk)
    doc_df["author"] = doc_df["author"].replace("__none__", None)

    print("\nDocument counts per domain/source:")
    print(doc_df.groupby(["domain", "source"]).size().to_string())

    print("\nMedian sentences per document:")
    print(doc_df.groupby(["domain", "source"])["n_sents"].median().to_string())

    print(f"\nSaving {len(doc_df):,} documents to {OUTPUT_PATH.name}...")
    doc_df.to_feather(OUTPUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
