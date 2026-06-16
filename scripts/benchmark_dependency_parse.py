"""Quick benchmark to estimate full parse runtime on this machine."""

import time
from pathlib import Path

import pandas as pd
import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data" / "processed" / "corpus.feather"

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")

df = pd.read_feather(INPUT)
df_sample = df.sample(50, random_state=42)

start = time.time()
for _, row in df_sample.iterrows():
    doc = nlp(row["text"])
elapsed = time.time() - start

per_doc = elapsed / 50
total_estimated = per_doc * len(df)

print(f"Time per document: {per_doc:.2f}s")
print(
    (
        f"Estimated total ({len(df)} docs): {total_estimated / 60:.1f} minutes "
        f"/ {total_estimated / 3600:.1f} hours"
    )
)
