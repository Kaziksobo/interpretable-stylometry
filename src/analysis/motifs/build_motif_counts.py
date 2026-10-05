"""Builds the per-document motif count table.

Counts every induced-subtree pattern (depth 2-4, at least two countable leaves;
see mining.count_patterns) in every sentence and aggregates per document, on
the same document pool as RQ0 and RQ1 (shared/filters.py). Counts are
occurrences: a pattern that occurs at two nodes of one sentence counts twice.

Three tables linked by integer ids, in long (sparse) format because a document
uses only a small fraction of the pattern vocabulary:

    motif_docs.feather       one row per document
        doc_idx  : int32 - row id used by the other tables
        domain, source, doc_id, author : the document key (author is None for
                   essay and wp)
        n_sents  : int32 - sentences in the document
        n_tokens : int32 - tokens (parse-tree leaves, so punctuation counts) in
                   the document; the exposure that rates are measured against
    motif_patterns.feather   one row per distinct pattern
        pattern_id : int32
        pattern    : str   - canonical bracketed form, e.g. "(S (NP) (VP))"
    motif_counts.feather     one row per (document, pattern) with count > 0
        doc_idx, pattern_id : int32
        count      : int32 - occurrences of the pattern in the document

Ids are assigned in a fixed order (documents sorted by key, patterns by first
appearance in that order), so re-running gives identical files.

Inputs:
    data/processed/parses/constituency_parses.feather
    data/processed/features/doc_features.feather

Outputs:
    data/processed/motifs/{motif_docs,motif_patterns,motif_counts}.feather

Usage (from the project root):
    python src/analysis/motifs/build_motif_counts.py
"""

import multiprocessing as mp
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from mining import count_patterns
from nltk import Tree
from tqdm import tqdm

from src.analysis.shared.filters import (
    DOC_KEY,
    NO_AUTHOR,
    drop_junk_sentences,
    filter_min_sents,
    restrict_to_docs,
)

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
PARSES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "parses" / "constituency_parses.feather"
)
DOC_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "features" / "doc_features.feather"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "motifs"

MAX_DEPTH = 4
MIN_DEPTH = 2
MIN_TERMINALS = 2
DOCS_PER_CHUNK = 50


def _count_chunk(
    chunk: list[tuple[int, list[str]]],
) -> tuple[list[tuple[int, Counter, int]], int]:
    """Worker: counts the patterns and tokens in each document of a chunk.

    Args:
        chunk (list[tuple[int, list[str]]]): (doc_idx, parse strings) for
            each document in the chunk.

    Returns:
        tuple[list[tuple[int, Counter, int]], int]: (doc_idx, pattern counts,
            token count) per document, and the number of parse strings that
            failed to parse.
    """
    results = []
    n_failed = 0
    for doc_idx, parse_strs in chunk:
        doc_counts: Counter = Counter()
        doc_tokens = 0
        for parse_str in parse_strs:
            try:
                tree = Tree.fromstring(parse_str)
            except ValueError:
                n_failed += 1
                continue
            doc_counts.update(
                count_patterns(tree, MAX_DEPTH, MIN_DEPTH, MIN_TERMINALS)
            )
            doc_tokens += len(tree.leaves())
        results.append((doc_idx, doc_counts, doc_tokens))
    return results, n_failed


def _group_by_document(sentences: pd.DataFrame) -> tuple[pd.DataFrame, list[list[str]]]:
    """Groups the sentences' parse strings by document.

    Documents are sorted by key, so doc_idx is the same on every run. The
    author is kept as NO_AUTHOR here (not None) so essay and wp rows survive
    the groupby; main() restores None before saving.

    Args:
        sentences (pd.DataFrame): Cleaned sentence table with the DOC_KEY
            columns and "parse_str".

    Returns:
        tuple[pd.DataFrame, list[list[str]]]: The document table (doc_idx,
            DOC_KEY columns, n_sents) and, in the same order, each
            document's parse strings.
    """
    keyed = sentences.assign(author=sentences["author"].fillna(NO_AUTHOR))
    grouped = keyed.groupby(DOC_KEY, sort=True)["parse_str"].agg(list)

    docs = grouped.index.to_frame(index=False)
    docs.insert(0, "doc_idx", np.arange(len(docs), dtype=np.int32))
    docs["n_sents"] = np.array([len(p) for p in grouped], dtype=np.int32)
    return docs, grouped.tolist()


def _check_against_doc_features(docs: pd.DataFrame, doc_features: pd.DataFrame):
    """Checks the document table against doc_features.

    The two tables are built by different scripts, so this confirms they
    describe the same pool: same documents, same sentence counts.

    Args:
        docs (pd.DataFrame): Output of _group_by_document.
        doc_features (pd.DataFrame): Filtered doc_features.

    Raises:
        ValueError: If a document is missing from either table or a
            sentence count differs.
    """
    feats = doc_features[DOC_KEY + ["n_sents"]].fillna({"author": NO_AUTHOR})
    merged = docs.merge(
        feats, on=DOC_KEY, how="outer", suffixes=("", "_feat"), indicator=True
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError(
            f"{(merged['_merge'] != 'both').sum():,} documents appear in only "
            "one of the sentence table and doc_features."
        )
    if not (merged["n_sents"] == merged["n_sents_feat"]).all():
        raise ValueError("Sentence counts differ between the parses and doc_features.")


def _count_all(
    parses_by_doc: list[list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, int]:
    """Counts patterns and tokens in every document, across all cores.

    Args:
        parses_by_doc (list[list[str]]): Each document's parse strings,
            indexed by doc_idx.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.ndarray, int]: The (doc_idx,
            pattern_id, count) table, the (pattern_id, pattern) table, each
            document's token count (indexed by doc_idx), and the number of
            parse strings that failed to parse.
    """
    n_docs = len(parses_by_doc)
    chunks = [
        [(i, parses_by_doc[i]) for i in range(start, min(start + DOCS_PER_CHUNK, n_docs))]
        for start in range(0, n_docs, DOCS_PER_CHUNK)
    ]

    vocab: dict[str, int] = {}
    doc_col, pattern_col, count_col = array("i"), array("i"), array("i")
    n_tokens = np.zeros(n_docs, dtype=np.int32)
    n_failed = 0

    n_cores = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=n_cores) as pool:
        # imap (not imap_unordered): results arrive in chunk order, so the
        # pattern ids are the same on every run
        for results, failed in tqdm(
            pool.imap(_count_chunk, chunks),
            total=len(chunks),
            desc="Counting patterns",
            ncols=80,
        ):
            n_failed += failed
            for doc_idx, doc_counts, doc_tokens in results:
                n_tokens[doc_idx] = doc_tokens
                for pattern, count in doc_counts.items():
                    pattern_id = vocab.setdefault(pattern, len(vocab))
                    doc_col.append(doc_idx)
                    pattern_col.append(pattern_id)
                    count_col.append(count)

    counts_df = pd.DataFrame(
        {
            "doc_idx": np.array(doc_col, dtype=np.int32),
            "pattern_id": np.array(pattern_col, dtype=np.int32),
            "count": np.array(count_col, dtype=np.int32),
        }
    ).sort_values(["doc_idx", "pattern_id"], ignore_index=True)

    patterns_df = pd.DataFrame(
        {"pattern_id": np.arange(len(vocab), dtype=np.int32), "pattern": list(vocab)}
    )
    return counts_df, patterns_df, n_tokens, n_failed


def main() -> None:
    """Builds and saves the three motif tables.

    Raises:
        ValueError: If any parse string fails to parse, since a dropped
            sentence would still be in the document's n_sents and bias its
            rates.
    """
    print("Loading sentences...")
    sentences = pd.read_feather(PARSES_PATH)
    print(f"  {len(sentences):,} sentences")
    sentences = drop_junk_sentences(sentences)
    doc_features = filter_min_sents(pd.read_feather(DOC_FEATURES_PATH))
    sentences = restrict_to_docs(sentences, doc_features)
    print(f"  {len(sentences):,} sentences remain in {len(doc_features):,} documents")

    docs, parses_by_doc = _group_by_document(sentences)
    _check_against_doc_features(docs, doc_features)

    counts_df, patterns_df, n_tokens, n_failed = _count_all(parses_by_doc)
    if n_failed:
        raise ValueError(
            f"{n_failed:,} parse strings failed to parse; their sentences are "
            "still counted in n_sents, so rates would be biased."
        )

    docs["n_tokens"] = n_tokens
    assert docs["n_tokens"].gt(0).all()

    # same sanity checks the explorer and stats rely on
    assert counts_df["doc_idx"].between(0, len(docs) - 1).all()
    assert counts_df["count"].gt(0).all()
    assert not counts_df.duplicated(["doc_idx", "pattern_id"]).any()

    docs["author"] = docs["author"].replace(NO_AUTHOR, None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "motif_docs": docs,
        "motif_patterns": patterns_df,
        "motif_counts": counts_df,
    }
    print("\nSaving...")
    for name, table in tables.items():
        path = OUTPUT_DIR / f"{name}.feather"
        table.to_feather(path)
        print(f"  {path.name}: {len(table):,} rows, {path.stat().st_size / 1e6:.1f} MB")

    print(
        f"\n{len(docs):,} documents, {len(patterns_df):,} distinct patterns, "
        f"{int(counts_df['count'].sum()):,} pattern occurrences."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
