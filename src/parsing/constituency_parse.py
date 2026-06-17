from pathlib import Path

import benepar
import pandas as pd
import spacy
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.feather"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_parses.feather"

_nlp = None


def get_nlp():
    """Returns a spaCy nlp object with benepar pipeline added."""
    global _nlp
    if _nlp is None:
        spacy.prefer_gpu()
        _nlp = spacy.load("en_core_web_trf")
        _nlp.add_pipe("benepar", config={"model": "benepar_en3"})
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


def main():
    df = load_data(INPUT_PATH)

    records = []
    skipped_docs = 0
    skipped_sents = 0

    texts_and_contexts = create_texts_and_contexts(df)

    # batch_size=1 is required: if a sentence exceeds benepar's 510 token
    # limit, the ValueError is raised during the batch's forward pass and
    # would otherwise kill every document in that batch, not just the
    # offending one.
    pipe = get_nlp().pipe(texts_and_contexts, as_tuples=True, batch_size=1)

    # A plain `for doc, ctx in pipe` cannot recover from an exception
    # raised mid-batch — the generator is left in an undefined state.
    # Manually advancing with next() lets us catch the ValueError for
    # exactly the offending document and continue pulling from the
    # same pipe, skipping only that one document.
    doc_iter = iter(pipe)
    with tqdm(total=len(df), desc="Parsing", unit="doc") as pbar:
        while True:
            try:
                doc, ctx = next(doc_iter)
            except StopIteration:
                break
            except ValueError:
                skipped_docs += 1
                pbar.update(1)
                continue

            pbar.update(1)

            for sent_idx, sent in enumerate(doc.sents):
                try:
                    parse_str = sent._.parse_string
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
