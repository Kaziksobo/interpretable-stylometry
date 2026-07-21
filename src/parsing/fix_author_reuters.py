"""Recovers `author` for Reuters rows in constituency_parses.feather and
dependency_parses.feather, without re-running benepar.

corpus.feather already has `author` (see load_corpus() in
ghostbuster_exploratory_analysis.ipynb) -- constituency_parse.py just never
copied it into its output records. Reuters doc_id alone can't disambiguate
documents (1-20 repeats across 50 author subfolders), but sentence text at a
given sent_idx can, once sentence boundaries are re-derived with the SAME
model used originally (en_core_web_trf) -- minus benepar, which is what made
the original run take ~3 hours and isn't needed for this fix.
"""

from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.feather"
CONSTITUENCY_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_parses.feather"
DEPENDENCY_PATH = PROJECT_ROOT / "data" / "processed" / "dependency_parses.feather"


def build_author_lookup(corpus_df: pd.DataFrame) -> pd.DataFrame:
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")  # no benepar

    reuter = corpus_df[corpus_df["domain"] == "reuter"].copy()
    reuter["id"] = reuter["id"].astype(str)

    rows = []
    for _, row in tqdm(
        reuter.iterrows(), total=len(reuter), desc="Re-segmenting reuter"
    ):
        try:
            doc = nlp(row["text"])
        except ValueError:
            continue
        rows.extend(
            {
                "source": row["source"],
                "doc_id": row["id"],
                "sent_idx": sent_idx,
                "sent_text": sent.text.strip(),
                "author": row["author"],
            }
            for sent_idx, sent in enumerate(doc.sents)
        )
    lookup_df = pd.DataFrame(rows)

    # Mirror remove_empty_sentences from constituency_parse.py —
    # empty sentences were dropped from the parse output so they
    # won't exist in constituency_parses.feather to join against anyway
    lookup_df = lookup_df[lookup_df["sent_text"].str.strip() != ""]

    # For any keys still ambiguous after removing empties (genuine
    # boilerplate collisions between authors), drop both sides rather
    # than risk misassigning author
    key_cols = ["source", "doc_id", "sent_idx", "sent_text"]
    dupes_mask = lookup_df.duplicated(subset=key_cols, keep=False)
    if n_ambiguous := dupes_mask.sum():
        print(
            f"Note: {n_ambiguous} rows remain ambiguous after removing "
            f"empty sentences (identical boilerplate across authors). "
            f"Excluding from lookup — affected parse rows will have null author."
        )
        lookup_df = lookup_df[~dupes_mask]

    print(f"Lookup built: {len(lookup_df)} unambiguous rows.")
    return lookup_df


def recover_authors(parses_path: Path, author_lookup: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_feather(parses_path)
    df["doc_id"] = df["doc_id"].astype(str)

    other = df[df["domain"] != "reuter"].copy()
    other["author"] = None

    reuter = df[df["domain"] == "reuter"].copy()
    before = len(reuter)

    # Left join — rows that couldn't be matched (ambiguous or missing)
    # get null author rather than being dropped
    merged = reuter.merge(
        author_lookup,
        on=["source", "doc_id", "sent_idx", "sent_text"],
        how="left",
        # validate removed — right side is already deduplicated above
    )

    n_unmatched = merged["author"].isna().sum()
    pct = 100 * n_unmatched / before
    print(
        f"{parses_path.name}: {n_unmatched}/{before} reuter rows "
        f"unmatched ({pct:.1f}%) — these will have null author."
    )

    return pd.concat([other, merged], ignore_index=True)


def main():
    corpus_df = pd.read_feather(CORPUS_PATH)
    author_lookup = build_author_lookup(corpus_df)

    dupes = author_lookup.duplicated(
        subset=["source", "doc_id", "sent_idx", "sent_text"], keep=False
    )
    if dupes.any():
        print(
            f"WARNING: {dupes.sum()} candidate sentences are genuinely ambiguous "
            f"(identical text, same doc_id/sent_idx, different authors):"
        )
        print(author_lookup[dupes].sort_values(["source", "doc_id", "sent_idx"]))

    for path in [CONSTITUENCY_PATH, DEPENDENCY_PATH]:
        if not path.exists():
            print(f"Skipping {path.name} (not found)")
            continue
        fixed = recover_authors(path, author_lookup)
        backup = path.with_suffix(".bak.feather")
        path.rename(backup)
        fixed.to_feather(path)
        print(
            (
                f"Patched {path.name} ({len(fixed)} rows); "
                f"original backed up to {backup.name}"
            )
        )


if __name__ == "__main__":
    main()
