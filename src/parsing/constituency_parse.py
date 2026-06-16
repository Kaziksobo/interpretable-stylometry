"""
Constituency parsing pipeline for the Ghostbuster corpus.

Runs benepar's en3 constituency parser (integrated with spaCy) over all
documents and saves the parsed output as a DataFrame with one row per
sentence, containing the sentence text and constituency parse features
directly relevant to RQ1.

Constituency parses provide phrase-structure trees (NP, VP, PP etc.),
enabling feature extraction that dependency parses cannot -- most
importantly the loose/periodic sentence classification from Feng et al.
(2012), which identifies whether the main clause comes first (loose)
or last (periodic) based on the position of the matrix S node.

Features extracted per sentence:
    - tree_depth:       maximum depth of the constituency tree
    - n_phrases:        total number of non-terminal phrase nodes
    - n_np:             number of noun phrases
    - n_vp:             number of verb phrases
    - n_pp:             number of prepositional phrases
    - n_sbar:           number of subordinate clauses (SBAR nodes)
    - n_s:              number of embedded S nodes (clause count)
    - is_loose:         True if sentence is loose (main clause precedes modifiers)
    - is_periodic:      True if sentence is periodic (main clause follows modifiers)
    - parse_str:        string representation of the parse tree (for inspection)

Notes:
    benepar has a hard limit of 510 tokens per sentence. Sentences
    exceeding this are skipped and counted in the final summary.
    batch_size=1 is used to prevent a single long sentence from
    killing an entire batch.

Output:
    data/processed/constituency_parses.feather

Runtime:
    Expect 2-4 hours on RTX 3060 due to batch_size=1.
    Run once and load from feather thereafter.

Usage:
    uv run src/parsing/constituency_parse.py
"""

from pathlib import Path

import pandas as pd
import spacy
from benepar import BeneparComponent  # noqa: F401 — triggers spaCy factory registration
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / "data" / "processed" / "corpus.feather"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "constituency_parses.feather"

BATCH_SIZE = 1


def get_tree_depth(parse_str: str) -> int:
    """Compute the maximum depth of the constituency parse tree."""
    depth = 0
    max_depth = 0
    for char in parse_str:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth -= 1
    return max_depth


def count_node(parse_str: str, label: str) -> int:
    """Count occurrences of a phrase label in a parse string."""
    return parse_str.count(f"({label} ")


def classify_sentence(parse_str: str) -> tuple[bool, bool]:
    """
    Classify a sentence as loose or periodic based on its constituency parse.

    Loose sentence: the matrix clause (S) appears first, followed by
    subordinate or adjunct material -- the main point comes first,
    elaboration follows. Characteristic of modern English prose.

    Periodic sentence: subordinate or adjunct material precedes the
    matrix clause -- elaboration builds to the main point at the end.
    Associated with more complex, literary prose styles.

    Based on the heuristic described in Feng et al. (2012): we examine
    the first immediate child of the top-level S node. If it is a
    fronted adjunct (SBAR, PP, ADVP) the sentence is periodic; if it
    is the subject NP or VP the sentence is loose.

    Returns:
        (is_loose, is_periodic): tuple of bools. Both can be False for
        simple sentences that do not fit either category clearly.
    """
    depth = 0
    top_children = []
    current = []

    for char in parse_str:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
            if depth == 1:
                node = "".join(current).strip()
                label = node.split()[0].lstrip("(")
                top_children.append(label)
                current = []
        elif depth >= 1:
            current.append(char)

    if not top_children:
        return False, False

    is_loose = top_children[0] in ("NP", "VP")
    is_periodic = top_children[0] in ("SBAR", "PP", "ADVP") and len(top_children) > 1

    return is_loose, is_periodic


def extract_sentence_features(sent) -> dict | None:
    """
    Extract constituency parse features from a single spaCy sentence span.

    Returns a dict of scalar features, or None if parsing failed.
    Raises ValueError if the sentence exceeds benepar's 510 token limit
    -- callers should catch this and skip the sentence.
    """
    parse_str = sent._.parse_string

    if not parse_str:
        return None

    is_loose, is_periodic = classify_sentence(parse_str)

    return {
        "sent_text": sent.text.strip(),
        "tree_depth": get_tree_depth(parse_str),
        "n_phrases": (
            count_node(parse_str, "NP")
            + count_node(parse_str, "VP")
            + count_node(parse_str, "PP")
            + count_node(parse_str, "SBAR")
            + count_node(parse_str, "S")
        ),
        "n_np": count_node(parse_str, "NP"),
        "n_vp": count_node(parse_str, "VP"),
        "n_pp": count_node(parse_str, "PP"),
        "n_sbar": count_node(parse_str, "SBAR"),
        "n_s": count_node(parse_str, "S"),
        "is_loose": is_loose,
        "is_periodic": is_periodic,
        "parse_str": parse_str,
    }


def parse_corpus(df: pd.DataFrame, nlp) -> pd.DataFrame:
    """
    Parse all documents in the corpus and extract sentence-level features.

    Uses nlp.pipe() with as_tuples=True to pass document metadata through
    alongside each doc. batch_size=1 ensures that a single document with
    an overlong sentence does not kill an entire batch.
    """
    records = []
    skipped = 0

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
            try:
                features = extract_sentence_features(sent)
            except ValueError:
                skipped += 1
                continue
            if features is None:
                continue
            features.update(ctx)
            records.append(features)

    print(f"\nSkipped {skipped} sentences exceeding 510 token limit.")
    return pd.DataFrame(records)


def main():
    print(f"Loading corpus from {INPUT}")
    df = pd.read_feather(INPUT)
    print(f"Loaded {len(df)} documents.")

    print("Loading spaCy + benepar model...")
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    print(f"GPU active: {spacy.prefer_gpu()}")

    print(f"Parsing corpus (batch_size={BATCH_SIZE})...")
    parsed = parse_corpus(df, nlp)

    print(f"\nParsed {len(parsed)} sentences from {len(df)} documents.")
    print(f"Saving to {OUTPUT}...")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    parsed.to_feather(OUTPUT)
    print("Done.")


if __name__ == "__main__":
    main()
