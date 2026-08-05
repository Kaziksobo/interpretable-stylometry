# Interpretable Stylometry for Human and AI Prose

Research code for a micro-placement project supervised by Dr Paul Nulty (Birkbeck, University of London).

## Project Overview

Computational stylometry has optimised relentlessly for discriminative accuracy - the ability to tell authors apart, or to detect AI-generated text - at the expense of interpretability. This project addresses that gap by developing syntactic, lexical, and prosodic features that can *explain* how AI-generated prose differs from human writing, rather than merely detecting it.

The analysis is structured around three research questions:

- **RQ1:** Does AI prose show reduced variance and greater regularity across interpretable syntactic features (loose/periodic sentences, clause complexity) compared to human prose?
- **RQ2:** Does LLM-generated prose show a regularisation in rhythm (sentence length variance, cadence, stress patterns) analogous to what Heuser (2025) found in verse?
- **RQ3:** Does instruction tuning amplify formal conservatism in AI prose, as it does in AI verse?

A full literature review motivating these questions is available in `docs/`.

## Current Status

**RQ1 - In progress**. Two parallel analysis tracks are complete:

- **Theory-driven (Feng features)**: Constituency parsing and per-sentence classification of sentence type (SIMPLE/COMPLEX/COMPOUND/COMPLEX-COMPOUND) and structure (LOOSE/PERIODIC) using Feng et al. (2012). Formal dispersion and location testing (Brown-Forsythe, Mann-Whitney U, Benjamini-Hochberg FDR correction) across all three domains and both AI models. Results in `results/rq1_significance.txt`.
- **Data-driven (syntactic motif mining)**: Bottom-up discovery of discriminatory phrase-structure patterns using induced subtree extraction from constituency parses. PMI-based comparison of GPT and Claude against human baseline, with categorical absence detection. Results in `results/stylometric_report.txt`. Formal dispersion testing on candidate motif patterns is the immediate next step.

**RQ2 - Not yet started**. Dependency parses are available; prosodic feature extraction is pending.

**RQ3 - Not yet started**. Requires construction of base vs instruct Llama corpora.

## Repository Structure

```
└── 📁interpretable-stylometry
    └── 📁data
        └── 📁processed
            ├── constituency_features.feather
            ├── constituency_parses.feather
            ├── corpus.feather
            ├── dependency_parses.bak.feather
            ├── dependency_parses.feather
            ├── doc_features.feather
        └── 📁raw
    └── 📁docs
        ├── stylometry-litreview.pdf
    └── 📁notebooks
        ├── constituency_analysis.ipynb
        ├── feng_algorithm_dev.ipynb
        ├── ghostbuster_exploratory_analysis.ipynb
    └── 📁results
        ├── ghostbuster_sentence_metrics.png
        ├── ghostbuster_sentence_stats.csv
        ├── ghostbuster_word_count_distributions.png
        ├── ghostbuster_word_count_stats.csv
        ├── rq1_significance.csv
        ├── rq1_significance.txt
        ├── stylometric_report.txt
    └── 📁src
        └── 📁analysis
            └── 📁__pycache__
                ├── analyze_corpus.cpython-312.pyc
                ├── mining.cpython-312.pyc
            ├── analyze_corpus.py
            ├── build_constituency_features.py
            ├── build_doc_features.py
            ├── feng_classifiers.py
            ├── mining.py
            ├── run_feature_analysis.py
            ├── run_rq1_significance.py
        └── 📁parsing
            ├── constituency_parse.py
            ├── dependency_parse.py
            ├── fix_author_reuters.py
    ├── .gitignore
    ├── .python-version
    ├── pyproject.toml
    ├── README.md
    └── uv.lock
```

## Datasets

| Dataset | Use | Source |
|---|---|---|
| Ghostbuster (Verma et al., 2024) | RQ1, RQ2 | github.com/vivek3141/ghostbuster-data |
| Llama 3.1 8B base (self-constructed) | RQ3 | HuggingFace: meta-llama/Llama-3.1-8B |
| Llama 3.1 8B Instruct (self-constructed) | RQ3 | HuggingFace: meta-llama/Llama-3.1-8B-Instruct |

Raw data is not committed to this repository. See the sources above to obtain it and place it in `data/raw/`.

## Setup

Requires Python 3.13+. Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/kaziksobo/interpretable-stylometry.git
cd interpretable-stylometry
uv sync
```

## References

- Verma, V., Fleisig, E., Tomlin, N., and Klein, D. (2024). Ghostbuster: Detecting Text Ghostwritten by Large Language Models. *NAACL 2024*.
- Feng, S., Banerjee, R., and Choi, Y. (2012). Characterizing Stylistic Elements in Syntactic Structure. *EMNLP 2012*.
- Heuser, R. (2025). Generative Aesthetics: On Formal Stuckness in AI Verse. *Journal of Cultural Analytics*.
