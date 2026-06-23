"""Dependency parsing for the Ghostbuster corpus (RQ1).

Parses each document in corpus.feather with spaCy's standard
dependency parser and writes one row per sentence to
dependency_parses.feather, with each tree serialised as a compact
CoNLL-U-style string.

Unlike constituency_parse.py, this script batches documents through
`nlp.pipe()`. That's safe here because the failure mode which forces
per-document looping in the constituency script — benepar raising
mid-batch on an over-length sentence — doesn't apply: spaCy's native
dependency parser has no equivalent hard per-sentence token ceiling,
so there's no risk of `nlp.pipe()` silently dropping a batch.

Output schema (one row per sentence):
    doc_id    : int  - source document id (from corpus.feather "id")
    domain    : str  - "essay" | "reuter" | "wp"
    source    : str  - "human" | "GPT" | "Claude"
    sent_idx  : int  - 0-based sentence index within the document
    sent_text : str  - raw sentence text, stripped
    parse_str : str  - dependency tree, one token per line, in the
                        compact format produced by sent_to_conllu()
"""

from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.feather"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dependency_parses.feather"

BATCH_SIZE = 32

_nlp = None  # lazily-initialised, module-level singleton (see get_nlp)


def get_nlp() -> spacy.language.Language:
    """Builds (or returns the cached) spaCy pipeline for dependency parsing.

    Cached in the module-level `_nlp` global so the model load and
    GPU setup only happen once.

    Returns:
        A spaCy Language object (en_core_web_trf), running on GPU if
        one is available.
    """
    global _nlp
    if _nlp is None:
        spacy.prefer_gpu()
        _nlp = spacy.load("en_core_web_trf")
    return _nlp


def load_data(data_path: Path) -> pd.DataFrame:
    """Loads the corpus from a feather file.

    Args:
        data_path: Path to a .feather file containing at least
            "id", "domain", "source", and "text" columns.

    Returns:
        The loaded DataFrame.
    """
    return pd.read_feather(data_path)


def create_texts_and_contexts(
    df: pd.DataFrame,
) -> list[tuple[str, dict[str, str | int]]]:
    """Builds the (text, metadata) pairs expected by nlp.pipe(as_tuples=True).

    spaCy's `as_tuples=True` mode pairs each parsed Doc with whatever
    context object was passed alongside its text, which is how the
    per-document id/domain/source metadata survives the pipe() call
    and gets reattached to the right Doc on the way out (see main()).

    Args:
        df: DataFrame with "id", "domain", "source", and "text" columns.

    Returns:
        A list of (text, metadata) tuples, where metadata has the
        structure:
            {
                "id": int,
                "domain": str,
                "source": str,
                "author": str | None
            }
    """
    texts_and_contexts = []
    for _, row in df.iterrows():
        text = row["text"]
        metadata = {
            "id": row["id"],
            "domain": row["domain"],
            "source": row["source"],
            "author": row["author"],
        }
        texts_and_contexts.append((text, metadata))

    return texts_and_contexts


def sent_to_conllu(sent) -> str:
    """Converts a spaCy sentence span into a compact CoNLL-U style string.

    One line per token, tab-separated:
        token_index    text    pos    head_index    dep_relation

    token_index and head_index are both 1-based and counted relative
    to the start of the sentence (i.e. token 1 is the sentence's first
    token). head_index of 0 marks the root token. This is the standard
    plain-text convention for serialising dependency trees, analogous
    to bracket notation for constituency trees — though note this is
    a simplified version, not a full CoNLL-U file (no lemma,
    morphology, or sentence-boundary metadata columns).

    Whitespace tokens are excluded from the output lines, but the
    index numbering still counts them (enumerate() runs over every
    token; only the append is skipped for whitespace). This keeps
    token_index and head_index on one consistent numbering scheme, so
    head_index always points at the correct line — the trade-off is
    that token_index in the output won't always be contiguous (e.g.
    1, 2, 4 if token 3 was whitespace).

    This assumes every token's head lies within the same sentence.
    spaCy's sentence segmentation should guarantee this in practice,
    but it is checked explicitly rather than taken on faith: if a
    token's head falls outside the sentence span, this raises
    ValueError naming the offending token, and main() catches that
    and counts the sentence as skipped rather than emitting a row
    with a corrupted (negative or out-of-range) head_index.

    Args:
        sent: A spaCy sentence Span (one element of doc.sents).

    Returns:
        The newline-joined CoNLL-U-style string for this sentence.

    Raises:
        ValueError: If any token's head lies outside this sentence's
            span (a cross-sentence dependency).
    """
    lines = []
    sent_start, sent_end = sent.start, sent.end
    for i, token in enumerate(sent, start=1):
        if token.is_space:
            continue
        if token.head == token:
            head_idx = 0
        elif sent_start <= token.head.i < sent_end:
            head_idx = (token.head.i - sent_start) + 1
        else:
            raise ValueError(
                f"Token {token.i} ('{token.text}') has a head outside its "
                f"own sentence (head token index {token.head.i}, sentence "
                f"span [{sent_start}, {sent_end}))."
            )
        lines.append(f"{i}\t{token.text}\t{token.pos_}\t{head_idx}\t{token.dep_}")
    return "\n".join(lines)


def main() -> None:
    """Runs dependency parsing over the full corpus and writes the output.

    Documents are streamed through `nlp.pipe(as_tuples=True)` in
    batches of BATCH_SIZE, which is considerably faster than the
    per-document loop required in constituency_parse.py (see module
    docstring for why that script can't use this approach).

    As in constituency_parse.py, two failure modes are tracked
    separately: a document can fail outright (skipped_docs), in which
    case sentence segmentation never completes for it; and an
    individual sentence can fail during CoNLL-U conversion
    (skipped_sents), which now specifically means sent_to_conllu()
    found a token whose head falls outside its own sentence (a
    cross-sentence dependency) and raised ValueError rather than
    emit a row with a corrupted head_index. Only that sentence is
    dropped; the rest of the document's sentences are kept.
    """
    df = load_data(INPUT_PATH)
    nlp = get_nlp()

    records = []
    skipped_docs = 0
    skipped_sents = 0

    texts_and_contexts = create_texts_and_contexts(df)

    pipe = nlp.pipe(texts_and_contexts, as_tuples=True, batch_size=BATCH_SIZE)

    for doc, ctx in tqdm(pipe, total=len(df), desc="Parsing", unit="doc"):
        try:
            sents = list(doc.sents)
        except ValueError:
            # Whole document failed sentence segmentation.
            skipped_docs += 1
            continue

        for sent_idx, sent in enumerate(sents):
            try:
                parse_str = sent_to_conllu(sent)
            except ValueError:
                skipped_sents += 1
                continue
            records.append(
                {
                    "doc_id": ctx["id"],
                    "domain": ctx["domain"],
                    "source": ctx["source"],
                    "author": ctx["author"],
                    "sent_idx": sent_idx,
                    "sent_text": sent.text.strip(),
                    "parse_str": parse_str,
                }
            )

    print(f"Skipped {skipped_docs} documents that failed to parse entirely.")
    print(
        f"Skipped {skipped_sents} individual sentences with a "
        "cross-sentence dependency (a token whose head fell outside "
        "its own sentence)."
    )

    parsed_df = pd.DataFrame(records)
    parsed_df.to_feather(OUTPUT_PATH)

    print(f"Saved {len(parsed_df)} parsed sentences to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
