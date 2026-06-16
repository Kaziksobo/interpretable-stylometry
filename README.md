# Interpretable Stylometry for Human and AI Prose

Research code for a micro-placement project supervised by Dr Paul Nulty (Birkbeck, University of London), connected to the broader project *Interpretable Stylometry for Human and AI Prose* (Co-PIs: Dr Paul Nulty and Dr Ryan Heuser).

## Project Overview

Computational stylometry has optimised relentlessly for discriminative accuracy - the ability to tell authors apart, or to detect AI-generated text - at the expense of interpretability. This project addresses that gap by developing syntactic, lexical, and prosodic features that can *explain* how AI-generated prose differs from human writing, rather than merely detecting it.

The analysis is structured around three research questions:

- **RQ1:** Does AI prose show reduced variance and greater regularity across interpretable syntactic features (loose/periodic sentences, clause complexity) compared to human prose?
- **RQ2:** Does LLM-generated prose show a regularisation in rhythm (sentence length variance, cadence, stress patterns) analogous to what Heuser (2025) found in verse?
- **RQ3:** Does instruction tuning amplify formal conservatism in AI prose, as it does in AI verse?

A full literature review motivating these questions is available in `docs/`.

## Repository Structure

```
interpretable-stylometry/
├── data/
│   ├── raw/                  # Original datasets, never modified
│   │   ├── ghostbuster/      # Verma et al. (2024) — RQ1, RQ2
│   │   ├── llama-base/       # Self-constructed — RQ3
│   │   └── llama-instruct/   # Self-constructed — RQ3
│   └── processed/            # Parsed outputs, feature matrices
├── docs/
│   └── litreview.pdf         # Background literature review
├── notebooks/
│   ├── rq1/                  # Syntactic feature analysis
│   ├── rq2/                  # Prosodic feature analysis
│   └── rq3/                  # Instruction tuning comparison
├── src/
│   ├── parsing/              # spaCy parsing pipeline
│   └── features/
│       ├── syntactic/        # RQ1 feature extraction
│       ├── prosodic/         # RQ2 feature extraction
│       └── posttrain/        # RQ3 feature extraction
├── results/                  # Figures and output tables
├── pyproject.toml
└── README.md
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