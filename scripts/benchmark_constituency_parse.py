"""
Benchmark script to estimate constituency parse runtime on the current machine.

Tests the benepar constituency parse pipeline on a small sample and
extrapolates to the full corpus. Run this before committing to a full
parse run with constituency_parse.py.

Usage:
    uv run scripts/benchmark_constituency.py
"""

import time
from pathlib import Path

import pandas as pd
import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data" / "processed" / "corpus.feather"

SAMPLE_SIZE = 50


def main():
    print(f"Loading corpus from {INPUT}")
    df = pd.read_feather(INPUT)
    df_sample = df.sample(SAMPLE_SIZE, random_state=42)
    n_total = len(df)
    print(f"Loaded {n_total} documents. Benchmarking on {SAMPLE_SIZE}.")

    spacy.prefer_gpu()
    print(f"GPU active: {spacy.prefer_gpu()}")

    print("\nLoading constituency parse pipeline (en_core_web_trf + benepar_en3)...")
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    texts = df_sample["text"].tolist()

    start = time.time()
    for _ in nlp.pipe(texts, batch_size=16):
        pass
    elapsed = time.time() - start

    per_doc = elapsed / SAMPLE_SIZE
    total_estimated = per_doc * n_total

    print("\nResults:")
    print(f"  Sample time ({SAMPLE_SIZE} docs): {elapsed:.1f}s")
    print(f"  Time per document:               {per_doc:.2f}s")
    print(
        (
            f"  Estimated total ({n_total} docs): {total_estimated / 60:.1f} minutes "
            f"/ {total_estimated / 3600:.1f} hours"
        )
    )


if __name__ == "__main__":
    main()
