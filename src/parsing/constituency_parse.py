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


def main():
    df = load_data(INPUT_PATH)
    nlp = get_nlp()

    records = []
    skipped_docs = 0
    skipped_sents = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing", unit="doc"):
        try:
            doc = nlp(row["text"])
        except ValueError:
            skipped_docs += 1
            continue

        for sent_idx, sent in enumerate(doc.sents):
            try:
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
