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

import benepar
import pandas as pd
import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data" / "processed" / "corpus.feather"

SAMPLE_SIZE = 50

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

df = pd.read_feather(INPUT)
df_sample = df.sample(SAMPLE_SIZE, random_state=42)

start = time.time()
for _, row in df_sample.iterrows():
    doc = nlp(row["text"])
elapsed = time.time() - start

per_doc = elapsed / SAMPLE_SIZE
total_estimated = per_doc * len(df)

print(f"Time per document: {per_doc:.2f}s")
print(
    (
        f"Estimated total ({len(df)} docs): {total_estimated / 60:.1f} minutes "
        f"/ {total_estimated / 3600:.1f} hours"
    )
)
