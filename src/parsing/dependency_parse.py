from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.feather"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dependency_parses.feather"

BATCH_SIZE = 32

_nlp = None


def get_nlp():
    """Returns a spaCy nlp object for dependency parsing."""
    global _nlp
    if _nlp is None:
        spacy.prefer_gpu()
        _nlp = spacy.load("en_core_web_trf")
    return _nlp


def load_data(data_path: Path):
    """Loads data from feather file and returns dataframe"""
    return pd.read_feather(data_path)


def create_texts_and_contexts(
    df: pd.DataFrame,
) -> list[tuple[str, dict[str, str | int]]]:
    """Creates a list of tuples containing text and metadata for each row in the df.

    metadata structure:
    {
        id: int,
        domain: str,
        source: str
    }
    """
    texts_and_contexts = []
    for _, row in df.iterrows():
        text = row["text"]
        metadata = {"id": row["id"], "domain": row["domain"], "source": row["source"]}
        texts_and_contexts.append((text, metadata))

    return texts_and_contexts


def sent_to_conllu(sent) -> str:
    """
    Converts a spaCy sentence span into a compact CoNLL-U style string.

    One line per token, tab-separated:
        token_index    text    pos    head_index    dep_relation

    token_index is 1-based within the sentence. head_index of 0 marks
    the root token. This is the standard plain-text convention for
    serialising dependency trees, analogous to bracket notation for
    constituency trees.
    """
    lines = []
    sent_start = sent.start
    for i, token in enumerate(sent, start=1):
        if token.is_space:
            continue
        head_idx = 0 if token.head == token else (token.head.i - sent_start) + 1
        lines.append(f"{i}\t{token.text}\t{token.pos_}\t{head_idx}\t{token.dep_}")
    return "\n".join(lines)


def main():
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
