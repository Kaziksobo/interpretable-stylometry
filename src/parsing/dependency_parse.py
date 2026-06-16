"""
Dependency parsing pipeline for the Ghostbuster corpus.

Runs spaCy's en_core_web_trf dependency parser over all documents and
saves the parsed output as a DataFrame with one row per sentence,
containing the sentence text and a set of dependency parse features
directly relevant to RQ1: whether AI-generated prose exhibits reduced
syntactic variance compared to human writing.

Features extracted per sentence:
    - n_tokens:         number of tokens (excluding whitespace)
    - max_depth:        longest path from any token to the root
    - mean_depth:       mean dependency depth across all tokens
    - n_clauses:        number of clausal dependents (csubj, ccomp, advcl, relcl, acl)
    - n_subordinate:    number of subordinate clauses (advcl, relcl, acl)
    - n_coord:          number of coordinated elements (conj)
    - is_loose:         heuristic flag — sentence ends with a subordinate/relative clause
    - is_periodic:      heuristic flag — root verb appears in final 25% of sentence

Output:
    data/processed/dependency_parses.feather

Runtime:
    ~1 hour on RTX 3060 with en_core_web_trf.
    Run once and load from feather thereafter.

Usage:
    uv run src/parsing/dependency_parse.py
"""

from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / "data" / "processed" / "corpus.feather"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "dependency_parses.feather"

BATCH_SIZE = 32


def get_token_depth(token) -> int:
    """Compute the depth of a token in the dependency tree."""
    depth = 0
    current = token
    while current.head != current:
        current = current.head
        depth += 1
    return depth


def extract_sentence_features(sent) -> dict | None:
    """
    Extract dependency parse features from a single spaCy sentence span.

    Returns a dict of scalar features, or None if the sentence is empty.
    """
    tokens = [t for t in sent if not t.is_space]
    if not tokens:
        return None

    depths = [get_token_depth(t) for t in tokens]
    root_tokens = [t for t in tokens if t.dep_ == "ROOT"]

    n_clauses = sum(
        1 for t in tokens if t.dep_ in ("csubj", "ccomp", "advcl", "relcl", "acl")
    )
    n_subordinate = sum(1 for t in tokens if t.dep_ in ("advcl", "relcl", "acl"))
    n_coord = sum(1 for t in tokens if t.dep_ == "conj")

    # Loose sentence heuristic: final content token is a subordinate/relative clause head
    last_token = tokens[-1]
    is_loose = last_token.dep_ in ("advcl", "relcl", "acl")

    # Periodic sentence heuristic: root verb appears in final 25% of sentence
    is_periodic = False
    if root_tokens:
        root_position = root_tokens[0].i - sent.start
        is_periodic = root_position >= len(tokens) * 0.75

    return {
        "sent_text": sent.text.strip(),
        "n_tokens": len(tokens),
        "max_depth": max(depths),
        "mean_depth": round(sum(depths) / len(depths), 4),
        "n_clauses": n_clauses,
        "n_subordinate": n_subordinate,
        "n_coord": n_coord,
        "is_loose": is_loose,
        "is_periodic": is_periodic,
    }


def parse_corpus(df: pd.DataFrame, nlp) -> pd.DataFrame:
    """
    Parse all documents in the corpus and extract sentence-level features.

    Uses nlp.pipe() with as_tuples=True to pass document metadata through
    alongside each doc, avoiding the need to re-join on index afterwards.
    """
    records = []

    texts_and_contexts = [
        (
            row["text"],
            {
                "doc_id": row["id"],
                "domain": row["domain"],
                "source": row["source"],
                "author": row["author"],
            },
        )
        for _, row in df.iterrows()
    ]

    pipe = nlp.pipe(texts_and_contexts, as_tuples=True, batch_size=BATCH_SIZE)

    for doc, ctx in tqdm(pipe, total=len(df), desc="Parsing", unit="doc"):
        for sent in doc.sents:
            features = extract_sentence_features(sent)
            if features is None:
                continue
            features.update(ctx)
            records.append(features)

    return pd.DataFrame(records)


def main():
    print(f"Loading corpus from {INPUT}")
    df = pd.read_feather(INPUT)
    print(f"Loaded {len(df)} documents.")

    print("Loading spaCy model (en_core_web_trf)...")
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    print(f"GPU active: {spacy.prefer_gpu()}")

    print(f"Parsing corpus in batches of {BATCH_SIZE}...")
    parsed = parse_corpus(df, nlp)

    print(f"\nParsed {len(parsed)} sentences from {len(df)} documents.")
    print(f"Saving to {OUTPUT}...")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    parsed.to_feather(OUTPUT)
    print("Done.")


if __name__ == "__main__":
    main()
