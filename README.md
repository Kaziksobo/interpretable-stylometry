# Interpretable Stylometry for Human and AI Prose

Research code for a micro-placement project supervised by Dr Paul Nulty (Birkbeck, University of London).

## Project Overview

Computational stylometry has optimised relentlessly for discriminative accuracy - the ability to tell authors apart, or to detect AI-generated text - at the expense of interpretability. This project addresses that gap by developing syntactic, lexical, and prosodic features that can *explain* how AI-generated prose differs from human writing, rather than merely detecting it.

The analysis is structured around four research questions:

- **RQ0:** Do humans, GPT and Claude use each sentence type and structure at different rates?
- **RQ1:** Does AI prose show reduced variance and greater regularity across interpretable syntactic features (loose/periodic sentences, clause complexity) compared to human prose?
- **RQ2:** Does LLM-generated prose show a regularisation in rhythm (sentence length variance, cadence, stress patterns) analogous to what Heuser (2025) found in verse?
- **RQ3:** Does instruction tuning amplify formal conservatism in AI prose, as it does in AI verse?

A full literature review motivating these questions is available in `docs/`.

## Current Status

**RQ0 - Complete**. Mean rate of each Feng et al. (2012) sentence type (SIMPLE/COMPLEX/COMPOUND/COMPLEX-COMPOUND/OTHER) and structure (LOOSE/PERIODIC/OTHER) per document, averaged per domain and source with 95% intervals. Charts in `results/rq0/` (grouped bars per domain or combined, and stacked composition bars); method in `docs/rq0_methodology.md`.

**RQ1 - Variance analysis complete**.

- **Between-document variance - complete**: per category, the variance of document rates with each document's sampling noise removed, relative to its maximum possible; uncertainty by resampling (whole authors for Reuters); human vs AI z-tests with Benjamini-Hochberg FDR correction. Charts in `results/rq1/`; method, verification and results in `docs/rq1_methodology.md`.
- **Regularity**: treated as part of the variance question; see `docs/rq1_methodology.md` §5.

**Motif mining - explorer built**. Bottom-up discovery of phrase-structure patterns (induced subtrees of depth 2-4) that GPT and Claude use more or less than humans, on the same document pool as RQ0 and RQ1. Rates are per 1,000 tokens, ranked by a Poisson log-rate-ratio z-score, with near-duplicate variants folded under their parents. The result is a single self-contained interactive page, `results/motifs/motif_explorer.html`, showing for each pattern its tree, rates, document counts and sampled example sentences with each pattern slot marked. Method, verification and limitations in `docs/motif_methodology.md`. Next: repertoire diversity and concentration of the motif distribution, as a separate track testing the reduced-variance hypothesis in prose.

**RQ2 - Not yet started**. Dependency parses are available; prosodic feature extraction is pending.

**RQ3 - Not yet started**. Requires construction of base vs instruct Llama corpora.

## Repository Structure

```
└── 📁interpretable-stylometry
    └── 📁data
        └── 📁processed
            ├── corpus.feather
            └── 📁features
                ├── constituency_features.feather
                ├── doc_features.feather
            └── 📁motifs                  (git-ignored; rebuilt by build_motif_counts.py and build_motif_examples.py)
                ├── motif_counts.feather
                ├── motif_docs.feather
                ├── motif_examples.feather
                ├── motif_patterns.feather
                ├── motif_sentences.feather
            └── 📁parses
                ├── constituency_parses.feather
                ├── dependency_parses.bak.feather
                ├── dependency_parses.feather
        └── 📁raw
    └── 📁docs
        ├── motif_methodology.md
        ├── rq0_methodology.md
        ├── rq1_methodology.md
        ├── stylometry-litreview.pdf
    └── 📁notebooks
        ├── constituency_analysis.ipynb
        ├── feng_algorithm_dev.ipynb
        ├── ghostbuster_exploratory_analysis.ipynb
    └── 📁results
        └── 📁exploratory
            ├── ghostbuster_sentence_metrics.png
            ├── ghostbuster_sentence_stats.csv
            ├── ghostbuster_word_count_distributions.png
            ├── ghostbuster_word_count_stats.csv
        └── 📁motifs
            ├── motif_explorer.html
            ├── motif_pool.csv
            ├── motif_stats.csv
            ├── stylometric_report.txt
        └── 📁rq0
            ├── rq0_rates.csv
            ├── rq0_rates_{essay,reuter,wp,grouped,stacked}.png
        └── 📁rq1
            ├── rq1_variance.csv
            ├── rq1_variance_tests.csv
            ├── rq1_variance_tests_sensitivity.csv
            ├── rq1_variance_{essay,reuter,wp,grouped}.png
    └── 📁src
        └── 📁analysis
            └── 📁features
                ├── build_doc_features.py
                ├── feng_classifiers.py
            └── 📁motifs
                ├── analyze_corpus.py
                ├── build_explorer.py
                ├── build_motif_counts.py
                ├── build_motif_examples.py
                ├── compute_motif_stats.py
                ├── explorer_template.html
                ├── mining.py
                ├── run_feature_analysis.py
            └── 📁rq0
                ├── plot_rq0.py
                ├── run_rq0.py
            └── 📁rq1
                ├── plot_rq1_variance.py
                ├── run_rq1_variance.py
            └── 📁shared
                ├── filters.py
                ├── significance_utils.py
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

Requires Python 3.12 (see `.python-version`). Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/kaziksobo/interpretable-stylometry.git
cd interpretable-stylometry
uv sync
```

## Running the Analyses

Run from the project root. Each step reads the previous step's output.

```bash
# Parses -> features (RQ0, RQ1)
python src/parsing/constituency_parse.py
python src/analysis/features/feng_classifiers.py
python src/analysis/features/build_doc_features.py

# RQ0 and RQ1: compute, then plot (PNG by default; --pdf for vector PDFs instead)
python src/analysis/rq0/run_rq0.py
python src/analysis/rq0/plot_rq0.py [--grouped | --stacked] [--pdf]
python src/analysis/rq1/run_rq1_variance.py
python src/analysis/rq1/plot_rq1_variance.py [--grouped] [--pdf]

# Motif mining: counts -> statistics -> examples -> explorer
python src/analysis/motifs/build_motif_counts.py
python src/analysis/motifs/compute_motif_stats.py
python src/analysis/motifs/build_motif_examples.py
python src/analysis/motifs/build_explorer.py [--n-cut 250] [--per-source 5]
```

Open `results/motifs/motif_explorer.html` in a browser; it needs no server.

## References

- Verma, V., Fleisig, E., Tomlin, N., and Klein, D. (2024). Ghostbuster: Detecting Text Ghostwritten by Large Language Models. *NAACL 2024*.
- Feng, S., Banerjee, R., and Choi, Y. (2012). Characterizing Stylistic Elements in Syntactic Structure. *EMNLP 2012*.
- Monroe, B. L., Colaresi, M. P., and Quinn, K. M. (2008). Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict. *Political Analysis*.
- Heuser, R. (2025). Generative Aesthetics: On Formal Stuckness in AI Verse. *Journal of Cultural Analytics*.
