"""Document and sentence filters shared across RQ0, RQ1 and motif mining.

Every analysis filters the same way, so the document pool is identical
across research questions. Not a driver script -- imported, no __main__.
"""

import re

import pandas as pd

# stand-in for a missing author (essay, wp), so those rows can still be
# grouped and merged on the document key
NO_AUTHOR = "__none__"

# doc_id alone is not unique in the Reuters domain, so documents are keyed by
# all four columns
DOC_KEY = ["domain", "source", "doc_id", "author"]

# a sentence that is only a label ending in a colon: a capital first letter, up
# to 40 characters of letters, digits, spaces, commas, apostrophes, ampersands
# and hyphens, and no sentence punctuation ("Introduction:", "SAM:")
HEADING_LABEL = re.compile(r"^[A-Z][A-Za-z0-9 ,'&-]{0,40}:$")


def filter_min_sents(df: pd.DataFrame, min_sents: int = 5) -> pd.DataFrame:
    """Excludes documents with fewer than min_sents sentences (§3).

    A document with very few sentences gives an unreliable composition --
    e.g. n_sents=1 is 100% one category by construction, and tells us
    nothing about the document's actual tendency.

    Args:
        df (pd.DataFrame): Any DataFrame with an "n_sents" column, e.g.
            doc_features.feather loaded directly.
        min_sents (int): Minimum sentence count required to keep a
            document. Defaults to 5.

    Returns:
        pd.DataFrame: The filtered rows, with a fresh 0-based index.
    """
    filtered_df = df[df["n_sents"] >= min_sents]
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


def drop_junk_sentences(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drops sentences that are not analysable prose.

    Three rules: sentences with no letters (sentence-splitter artefacts such
    as a lone quote mark or "...", which every classifier labels OTHER),
    Reuters sentences with no author (orphaned pseudo-documents left over
    from the author fix), and bare heading lines (a section label or speaker
    label such as "Introduction:" or "SAM:", which the splitter turns into a
    sentence of its own). GPT essays in particular contain section labels;
    left in, they inflate that source's type-OTHER rate.

    Args:
        df (pd.DataFrame): Sentence-level DataFrame with "sent_text",
            "domain" and "author" columns.
        verbose (bool): Print how many sentences each rule drops.
            Defaults to True.

    Returns:
        pd.DataFrame: The kept rows, with a fresh 0-based index.
    """
    has_letters = df["sent_text"].str.contains(r"[^\W\d_]", regex=True, na=False)
    reuter_orphan = (df["domain"] == "reuter") & df["author"].isna()
    heading = df["sent_text"].str.strip().str.match(HEADING_LABEL, na=False)

    if verbose:
        print(f"  dropping {(~has_letters).sum():,} sentences with no letters")
        print(f"  dropping {reuter_orphan.sum():,} reuter sentences with no author")
        print(f"  dropping {heading.sum():,} bare heading lines")

    return df[has_letters & ~reuter_orphan & ~heading].reset_index(drop=True)


def restrict_to_docs(sentences: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    """Keeps only the sentences that belong to one of the given documents.

    Matches on DOC_KEY, treating a missing author as a single value so that
    essay and wp rows still match. This puts a sentence-level analysis on
    exactly the document pool a document-level one uses, e.g. after
    filter_min_sents.

    Args:
        sentences (pd.DataFrame): Sentence-level DataFrame with the DOC_KEY
            columns.
        docs (pd.DataFrame): Document-level DataFrame with the DOC_KEY
            columns, e.g. doc_features.feather after filter_min_sents.

    Returns:
        pd.DataFrame: The matching sentences, with a fresh 0-based index.

    Raises:
        ValueError: If a document in docs has no sentences in sentences,
            which means the two tables do not describe the same corpus (or
            the key columns disagree in type or spelling).
    """
    doc_keys = docs[DOC_KEY].fillna({"author": NO_AUTHOR}).drop_duplicates()
    sent_keys = sentences[DOC_KEY].fillna({"author": NO_AUTHOR})

    # doc_keys is unique, so a left merge keeps one row per sentence, in order
    matched = sent_keys.merge(doc_keys, on=DOC_KEY, how="left", indicator=True)
    assert len(matched) == len(sentences)
    kept = sentences[(matched["_merge"] == "both").to_numpy()].reset_index(drop=True)

    kept_keys = kept[DOC_KEY].fillna({"author": NO_AUTHOR}).drop_duplicates()
    if len(kept_keys) != len(doc_keys):
        raise ValueError(
            f"{len(doc_keys) - len(kept_keys):,} of {len(doc_keys):,} documents "
            "have no sentences in the sentence table; check that both tables "
            "were built from the same corpus."
        )
    return kept
