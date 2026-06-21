"""Constituency parsing for the Ghostbuster corpus (RQ1).

Parses each document in corpus.feather into per-sentence constituency
trees using spaCy + benepar, and writes the results to
constituency_parses.feather for downstream feature extraction (Feng
et al. 2012, Algorithm 1: sentence type; Algorithm 2: sentence
structure).

Each document is parsed individually with `nlp(text)` rather than
batched via `nlp.pipe()`. This is deliberate: benepar raises a
ValueError when a sentence exceeds its ~510-token limit, and when
that happens inside an `nlp.pipe()` batch, spaCy does not recover
cleanly — it silently drops the remainder of that batch rather than
just the offending sentence. Looping over single documents with
explicit per-sentence error handling avoids this, at the cost of
losing pipe-level batching speed.

Output schema (one row per sentence):
    doc_id    : int  - source document id (from corpus.feather "id")
    domain    : str  - "essay" | "reuter" | "wp"
    source    : str  - "human" | "GPT" | "Claude"
    sent_idx  : int  - 0-based sentence index within the document
    sent_text : str  - raw sentence text, stripped
    parse_str : str  - bracketed constituency parse string,
                        e.g. "(S (NP ...) (VP ...))"
"""

from pathlib import Path

import benepar
import pandas as pd
import spacy
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.feather"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_parses.feather"

_nlp = None  # lazily-initialised, module-level singleton (see get_nlp)


def get_nlp() -> spacy.language.Language:
    """Builds (or returns the cached) spaCy pipeline with benepar attached.

    Cached in the module-level `_nlp` global so the model load and
    GPU setup only happen once, even if this is called repeatedly.

    Returns:
        A spaCy Language object with "benepar" added as the final
        pipeline component, running on GPU if one is available.
    """
    global _nlp
    if _nlp is None:
        spacy.prefer_gpu()
        _nlp = spacy.load("en_core_web_trf")
        _nlp.add_pipe("benepar", config={"model": "benepar_en3"})
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


def main() -> None:
    """Runs constituency parsing over the full corpus and writes the output.

    Iterates documents one at a time (see module docstring for why
    `nlp.pipe()` is avoided here). Two independent failure modes are
    tracked separately:

      - A document can fail outright when `nlp(text)` itself raises
        ValueError (e.g. the document exceeds the transformer's max
        sequence length). The entire document is skipped.
      - A document can pass sentence segmentation fine, but an
        individual sentence can still be too long for benepar's
        constituency parser, which enforces a hard ~510-token limit
        per sentence. Only that sentence is dropped; the rest of the
        document's sentences are kept.
    """
    df = load_data(INPUT_PATH)
    nlp = get_nlp()

    records = []
    skipped_docs = 0
    skipped_sents = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing", unit="doc"):
        try:
            doc = nlp(row["text"])
        except ValueError:
            # Whole document failed (e.g. too long for the transformer).
            skipped_docs += 1
            continue

        for sent_idx, sent in enumerate(doc.sents):
            try:
                # Accessing .parse_string is what actually triggers
                # benepar's parse for this sentence; this is where the
                # per-sentence token-limit ValueError surfaces.
                parse_str = sent._.parse_string
            except ValueError:
                skipped_sents += 1
                continue
            records.append(
                {
                    "doc_id": row["id"],
                    "domain": row["domain"],
                    "source": row["source"],
                    "sent_idx": sent_idx,
                    "sent_text": sent.text.strip(),
                    "parse_str": parse_str,
                }
            )

    print(f"Skipped {skipped_docs} documents that failed to parse entirely.")
    print(f"Skipped {skipped_sents} individual sentences exceeding the token limit.")

    parsed_df = pd.DataFrame(records)
    parsed_df.to_feather(OUTPUT_PATH)

    print(f"Saved {len(parsed_df)} parsed sentences to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
