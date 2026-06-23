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

PROJECT_ROOT = Path(__file__).parent.parent.parent
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.feather"
CONSTITUENCY_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_parses.feather"
DEPENDENCY_PATH = PROJECT_ROOT / "data" / "processed" / "dependency_parses.feather"


def build_author_lookup(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Re-derives (source, doc_id, sent_idx, sent_text) -> author for reuter only."""
    spacy.prefer_gpu()
    nlp = spacy.load(
        "en_core_web_trf"
    )  # same model as constituency_parse.py; benepar NOT added

    reuter = corpus_df[corpus_df["domain"] == "reuter"].copy()
    reuter["id"] = reuter["id"].astype(str)

    rows = []
    for _, row in reuter.iterrows():
        try:
            doc = nlp(row["text"])
        except ValueError:
            continue  # failed in the original run too -- nothing to recover for this doc
        for sent_idx, sent in enumerate(doc.sents):
            rows.append(
                {
                    "source": row["source"],
                    "doc_id": row["id"],
                    "sent_idx": sent_idx,
                    "sent_text": sent.text.strip(),
                    "author": row["author"],
                }
            )
    return pd.DataFrame(rows)


def recover_authors(parses_path: Path, author_lookup: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_feather(parses_path)
    df["doc_id"] = df["doc_id"].astype(str)

    other = df[df["domain"] != "reuter"].copy()
    other["author"] = None

    reuter = df[df["domain"] == "reuter"].copy()
    before = len(reuter)

    merged = reuter.merge(
        author_lookup,
        on=["source", "doc_id", "sent_idx", "sent_text"],
        how="left",
        validate="m:1",  # raises if any (source,doc_id,sent_idx,sent_text) is ambiguous across authors
    )

    n_unmatched = merged["author"].isna().sum()
    if n_unmatched:
        print(
            f"WARNING: {n_unmatched}/{before} reuter rows in {parses_path.name} "
            f"didn't match an author. Inspect before trusting this file."
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
            f"Patched {path.name} ({len(fixed)} rows); original backed up to {backup.name}"
        )


if __name__ == "__main__":
    main()
