"""Samples example sentences for each pattern, with the matched span.

For every (domain, pattern) in motif_stats.csv and every source that uses it,
picks up to N_EXAMPLES example occurrences and records the sentence and the
span of tokens the pattern's root constituent covers, so the explorer can mark
the match in the sentence without any text matching.

Sampling is deterministic and does not depend on processing order: each
(sentence, pattern) pair gets a pseudo-random priority from a seeded hash, and
a source's examples are the occurrences with the lowest priorities. Rank 0 is
the lowest, so the first n examples are a uniform random sample of size n and
showing more just reveals the next ones. A sentence offers at most one
occurrence of a pattern (the leftmost, innermost), so examples are distinct
sentences. Examples are drawn from every sentence of the source, not only the
first ones met.

Sentences are stored once, as space-separated parse tokens (so punctuation is
a token, and brackets appear as -LRB- and -RRB-), and examples refer to them by
sent_id. Spans index into those tokens. Each example also records where the
pattern's own leaf slots fall inside its span, so a pattern that covers a whole
sentence can still be shown as labelled segments.

Inputs:
    data/processed/parses/constituency_parses.feather
    data/processed/motifs/{motif_docs,motif_patterns}.feather
    results/motifs/motif_stats.csv

Outputs:
    data/processed/motifs/motif_sentences.feather   one row per pool sentence
        sent_id : int32 - position in the pool, sorted by (doc_idx, sent_idx)
        doc_idx : int32 - the document (motif_docs)
        tokens  : str   - the sentence's tokens, space-separated
    data/processed/motifs/motif_examples.feather    one row per example
        domain, source : categorical
        pattern_id     : int32
        rank           : int8  - 0 is the first example to show
        sent_id        : int32
        start, end     : int16 - tokens[start:end] are the pattern's tokens
        bounds         : str   - comma-separated token indices where one slot
                         of the pattern ends and the next begins; with the
                         pattern's leaf labels in order, the slots are
                         [start, b0), [b0, b1), ..., [b_last, end)

Usage (from the project root):
    python src/analysis/motifs/build_motif_examples.py
"""

import hashlib
import heapq
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from build_motif_counts import MAX_DEPTH, MIN_DEPTH, MIN_TERMINALS
from mining import find_patterns_with_spans, remove_empty_nodes
from nltk import Tree
from tqdm import tqdm

from src.analysis.shared.filters import (
    DOC_KEY,
    NO_AUTHOR,
    drop_junk_sentences,
    restrict_to_docs,
)

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
PARSES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "parses" / "constituency_parses.feather"
)
MOTIF_DIR = PROJECT_ROOT / "data" / "processed" / "motifs"
STATS_PATH = PROJECT_ROOT / "results" / "motifs" / "motif_stats.csv"

N_EXAMPLES = 20
SEED = 0
SENTENCES_PER_CHUNK = 2000

_kept: dict[str, set[str]] = {}  # domain -> patterns that have a stats row


def _init_worker(kept: dict[str, set[str]]) -> None:
    """Gives each worker process the per-domain sets of patterns to sample."""
    global _kept
    _kept = kept


def _priority(sent_id: int, pattern: str) -> int:
    """Returns a pseudo-random, order-independent priority for an occurrence."""
    digest = hashlib.blake2b(
        f"{SEED}|{sent_id}|{pattern}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _process_chunk(
    chunk: list[tuple[int, str, str, str]],
) -> tuple[dict, list[tuple[int, str, int]], int]:
    """Worker: finds each sentence's sampled pattern occurrences and tokens.

    Args:
        chunk (list[tuple[int, str, str, str]]): (sent_id, domain, source,
            parse string) per sentence.

    Returns:
        tuple: The N_EXAMPLES lowest-priority (priority, sent_id, start, end,
            bounds) per (domain, source, pattern) within this chunk; (sent_id,
            tokens, n_tokens) per sentence; and the number of parse strings
            that failed to parse.

    Raises:
        ValueError: If a token contains whitespace, which would break the
            space-separated token format.
    """
    examples = defaultdict(list)
    sentences = []
    n_failed = 0

    for sent_id, domain, source, parse_str in chunk:
        try:
            tree = Tree.fromstring(parse_str)
        except ValueError:
            n_failed += 1
            continue
        tree = remove_empty_nodes(tree)  # same cleaning as build_motif_counts
        if tree is None:  # no words at all
            n_failed += 1
            continue

        tokens = tree.leaves()
        joined = " ".join(tokens)
        if len(joined.split(" ")) != len(tokens):
            raise ValueError(f"sentence {sent_id} has a token with whitespace")
        sentences.append((sent_id, joined, len(tokens)))

        kept = _kept[domain]
        first: dict[str, tuple[int, int, tuple[int, ...]]] = {}
        for pattern, start, end, bounds in find_patterns_with_spans(
            tree, MAX_DEPTH, MIN_DEPTH, MIN_TERMINALS
        ):
            if pattern in kept and (
                pattern not in first or (start, end) < first[pattern][:2]
            ):
                first[pattern] = (start, end, bounds)

        for pattern, (start, end, bounds) in first.items():
            examples[(domain, source, pattern)].append(
                (_priority(sent_id, pattern), sent_id, start, end, bounds)
            )

    trimmed = {
        key: heapq.nsmallest(N_EXAMPLES, rows) for key, rows in examples.items()
    }
    return trimmed, sentences, n_failed


def _load_pool_sentences(docs: pd.DataFrame) -> pd.DataFrame:
    """Loads the pool's sentences, sorted by document and position.

    Args:
        docs (pd.DataFrame): motif_docs.

    Returns:
        pd.DataFrame: Columns sent_id, doc_idx, domain, source, parse_str,
            one row per sentence of the pool, in (doc_idx, sent_idx) order.
    """
    sentences = pd.read_feather(PARSES_PATH)
    sentences = restrict_to_docs(drop_junk_sentences(sentences, verbose=False), docs)

    keyed = sentences[DOC_KEY].fillna({"author": NO_AUTHOR})
    doc_keys = docs[["doc_idx"] + DOC_KEY].fillna({"author": NO_AUTHOR})
    merged = keyed.merge(doc_keys, on=DOC_KEY, how="left")
    assert len(merged) == len(sentences) and merged["doc_idx"].notna().all()

    sentences = sentences.assign(doc_idx=merged["doc_idx"].to_numpy(dtype=np.int32))
    sentences = sentences.sort_values(["doc_idx", "sent_idx"], ignore_index=True)
    sentences["sent_id"] = np.arange(len(sentences), dtype=np.int32)
    return sentences[["sent_id", "doc_idx", "domain", "source", "parse_str"]]


def main() -> None:
    """Samples the examples and saves the sentence and example tables.

    Raises:
        ValueError: If any parse string fails to parse, or the sampled
            examples do not cover every (domain, source, pattern) that the
            stats table says occurs.
    """
    docs = pd.read_feather(MOTIF_DIR / "motif_docs.feather")
    patterns = pd.read_feather(MOTIF_DIR / "motif_patterns.feather")
    stats = pd.read_csv(
        STATS_PATH,
        usecols=["domain", "pattern_id", "count_human", "count_gpt", "count_claude"],
    )

    pattern_strs = patterns["pattern"].to_numpy()
    kept = {
        domain: set(pattern_strs[group["pattern_id"].to_numpy()])
        for domain, group in stats.groupby("domain")
    }
    pattern_ids = {pattern: i for i, pattern in enumerate(pattern_strs)}

    print("Loading sentences...")
    sentences = _load_pool_sentences(docs)
    print(f"  {len(sentences):,} sentences")

    rows = list(
        zip(
            sentences["sent_id"],
            sentences["domain"],
            sentences["source"],
            sentences["parse_str"],
        )
    )
    chunks = [
        rows[i : i + SENTENCES_PER_CHUNK]
        for i in range(0, len(rows), SENTENCES_PER_CHUNK)
    ]

    sampled: dict[tuple[str, str, str], list] = defaultdict(list)
    token_strs = [""] * len(sentences)
    n_tokens = np.zeros(len(sentences), dtype=np.int32)
    n_failed = 0

    n_cores = max(1, mp.cpu_count() - 1)
    with mp.Pool(n_cores, initializer=_init_worker, initargs=(kept,)) as pool:
        for chunk_examples, chunk_sentences, failed in tqdm(
            pool.imap(_process_chunk, chunks),
            total=len(chunks),
            desc="Sampling examples",
            ncols=80,
        ):
            n_failed += failed
            for sent_id, text, n in chunk_sentences:
                token_strs[sent_id] = text
                n_tokens[sent_id] = n
            for key, entries in chunk_examples.items():
                merged = sampled[key]
                merged.extend(entries)
                if len(merged) > 2 * N_EXAMPLES:
                    sampled[key] = heapq.nsmallest(N_EXAMPLES, merged)

    if n_failed:
        raise ValueError(f"{n_failed:,} parse strings failed to parse.")

    # the sentence table must describe exactly the tokens the counts used
    doc_tokens = np.bincount(
        sentences["doc_idx"].to_numpy(), weights=n_tokens, minlength=len(docs)
    )
    if not (doc_tokens == docs["n_tokens"].to_numpy()).all():
        raise ValueError("Token counts differ from motif_docs.n_tokens.")

    records = []
    for (domain, source, pattern), entries in sorted(sampled.items()):
        pattern_id = pattern_ids[pattern]
        for rank, (_, sent_id, start, end, bounds) in enumerate(
            heapq.nsmallest(N_EXAMPLES, entries)
        ):
            records.append(
                (
                    domain,
                    source,
                    pattern_id,
                    rank,
                    sent_id,
                    start,
                    end,
                    ",".join(map(str, bounds)),
                )
            )
    examples = pd.DataFrame(
        records,
        columns=[
            "domain",
            "source",
            "pattern_id",
            "rank",
            "sent_id",
            "start",
            "end",
            "bounds",
        ],
    ).astype(
        {
            "domain": "category",
            "source": "category",
            "pattern_id": np.int32,
            "rank": np.int8,
            "sent_id": np.int32,
            "start": np.int16,
            "end": np.int16,
        }
    )

    # every pattern the stats table says a source uses must have an example
    n_expected = sum(int((stats[f"count_{s}"] > 0).sum()) for s in ("human", "gpt", "claude"))
    n_keys = len(sampled)
    if n_keys != n_expected:
        raise ValueError(
            f"Examples cover {n_keys:,} (domain, source, pattern) groups but "
            f"the stats table has {n_expected:,}."
        )

    sentence_table = pd.DataFrame(
        {
            "sent_id": sentences["sent_id"].to_numpy(),
            "doc_idx": sentences["doc_idx"].to_numpy(),
            "tokens": token_strs,
        }
    )

    print("\nSaving...")
    for name, table in {
        "motif_sentences": sentence_table,
        "motif_examples": examples,
    }.items():
        path = MOTIF_DIR / f"{name}.feather"
        table.to_feather(path)
        print(f"  {path.name}: {len(table):,} rows, {path.stat().st_size / 1e6:.1f} MB")

    per_group = examples.groupby(["domain", "source", "pattern_id"], observed=True).size()
    print(
        f"\n{n_keys:,} (domain, source, pattern) groups; "
        f"{(per_group == N_EXAMPLES).mean():.0%} have the full {N_EXAMPLES} examples."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
