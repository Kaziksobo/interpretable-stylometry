"""Computes the per-pattern statistics behind the motif explorer.

Reads the per-document motif count tables (build_motif_counts.py) and writes
one row per (domain, pattern), for every pattern that occurs in at least
MIN_DOCS documents of that domain. Each row has, for each source (human, GPT,
Claude): the occurrence count, the number of documents containing the pattern,
and its rate per 1,000 tokens. For GPT and Claude against human it also has the
log2 rate ratio and a z-score.

Rates use tokens (parse-tree leaves, so punctuation counts) as the exposure,
not sentences: a source with longer sentences would otherwise score higher on
almost every construction, and the rankings would partly measure sentence
length. Counts are occurrences (a pattern that occurs twice in a sentence
counts twice), so a rate is occurrences per 1,000 tokens, not the share of
sentences containing the pattern.

The z-score is a Poisson log-rate ratio in the style of Monroe et al.'s
"Fightin' Words", with SMOOTHING added to each count:

    z = ln(rate_AI / rate_human) / sqrt(1 / (c_AI + a) + 1 / (c_human + a))

It ranks patterns by weight of evidence, where the raw ratio would put the
rarest patterns first. It is a ranking statistic, not a significance test: it
treats occurrences as independent, so it is anti-conservative when a few
documents account for most of a pattern's occurrences. The per-source document
counts are there to catch that.

Inputs:
    data/processed/motifs/{motif_docs,motif_patterns,motif_counts}.feather

Outputs:
    results/motifs/motif_stats.csv   one row per (domain, pattern)
    results/motifs/motif_pool.csv    documents, sentences and tokens per
                                     (domain, source)

Usage (from the project root):
    python src/analysis/motifs/compute_motif_stats.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from mining import count_terminal_nodes, pattern_depth

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
MOTIF_DIR = PROJECT_ROOT / "data" / "processed" / "motifs"
OUTPUT_DIR = PROJECT_ROOT / "results" / "motifs"

SOURCES = ["human", "gpt", "claude"]
AI_SOURCES = ["gpt", "claude"]

# a pattern is kept if this many documents of its domain contain it, summed
# over the three sources
MIN_DOCS = 10
SMOOTHING = 0.5  # added to every count before taking log rates
RATE_PER = 1000  # rates are occurrences per this many tokens


def _aggregate(docs: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Sums the per-document counts to (domain, source, pattern).

    Each (document, pattern) row is unique, so the number of rows in a group
    is the number of documents in that source containing the pattern.

    Args:
        docs (pd.DataFrame): motif_docs, with doc_idx equal to row position.
        counts (pd.DataFrame): motif_counts.

    Returns:
        pd.DataFrame: Columns domain, source, pattern_id, count (total
            occurrences) and docs (documents containing the pattern).
    """
    doc_idx = counts["doc_idx"].to_numpy()
    keyed = pd.DataFrame(
        {
            "domain": docs["domain"].to_numpy()[doc_idx],
            "source": docs["source"].to_numpy()[doc_idx],
            "pattern_id": counts["pattern_id"].to_numpy(),
            "count": counts["count"].to_numpy(),
        }
    )
    return (
        keyed.groupby(["domain", "source", "pattern_id"])
        .agg(count=("count", "sum"), docs=("count", "size"))
        .reset_index()
    )


def _domain_stats(
    domain: str, agg: pd.DataFrame, pool: pd.DataFrame, patterns: pd.DataFrame
) -> pd.DataFrame:
    """Builds one domain's statistics table.

    Args:
        domain (str): "essay", "reuter" or "wp".
        agg (pd.DataFrame): Output of _aggregate, all domains.
        pool (pd.DataFrame): Documents, sentences and tokens per
            (domain, source).
        patterns (pd.DataFrame): motif_patterns, with pattern_id equal to row
            position.

    Returns:
        pd.DataFrame: One row per kept pattern; see the module docstring.
    """
    n_tokens = pool[pool["domain"] == domain].set_index("source")["n_tokens"]

    wide = agg[agg["domain"] == domain].pivot(
        index="pattern_id", columns="source", values=["count", "docs"]
    )
    wide.columns = [f"{value}_{source}" for value, source in wide.columns]
    expected = [f"{v}_{s}" for v in ("count", "docs") for s in SOURCES]
    # a pattern that a source never uses has no row for it, so it pivots to NaN
    wide = wide.reindex(columns=expected).fillna(0).astype(int)

    docs_total = wide[[f"docs_{s}" for s in SOURCES]].sum(axis=1)
    out = wide[docs_total >= MIN_DOCS].reset_index()

    for source in SOURCES:
        out[f"rate_{source}"] = RATE_PER * out[f"count_{source}"] / n_tokens[source]

    human_adj = out["count_human"] + SMOOTHING
    for ai in AI_SOURCES:
        ai_adj = out[f"count_{ai}"] + SMOOTHING
        delta = np.log(ai_adj / n_tokens[ai]) - np.log(human_adj / n_tokens["human"])
        out[f"log2_ratio_{ai}"] = delta / np.log(2)
        out[f"z_{ai}"] = delta / np.sqrt(1 / ai_adj + 1 / human_adj)

    out["pattern"] = patterns["pattern"].to_numpy()[out["pattern_id"].to_numpy()]
    out["depth"] = out["pattern"].map(pattern_depth)
    out["n_leaves"] = out["pattern"].map(count_terminal_nodes)
    out.insert(0, "domain", domain)

    columns = (
        ["domain", "pattern_id", "pattern", "depth", "n_leaves"]
        + [f"{v}_{s}" for s in SOURCES for v in ("count", "docs", "rate")]
        + [f"{v}_{ai}" for ai in AI_SOURCES for v in ("log2_ratio", "z")]
    )
    return out[columns]


def _print_top(stats: pd.DataFrame, n: int = 5) -> None:
    """Prints each domain's most over- and under-used patterns, as a check.

    Args:
        stats (pd.DataFrame): The full statistics table.
        n (int): How many patterns to show at each end.
    """
    for domain, group in stats.groupby("domain"):
        for ai in AI_SOURCES:
            ranked = group.sort_values(f"z_{ai}", ascending=False)
            for label, rows in (
                ("over-used by", ranked.head(n)),
                ("under-used by", ranked.tail(n).iloc[::-1]),
            ):
                print(f"\n{domain}: {label} {ai} (rate per {RATE_PER:,} tokens)")
                for _, r in rows.iterrows():
                    print(
                        f"  z={r[f'z_{ai}']:+7.1f}  human {r['rate_human']:8.2f}"
                        f"  {ai} {r[f'rate_{ai}']:8.2f}"
                        f"  docs {r['docs_human']}/{r[f'docs_{ai}']}  {r['pattern']}"
                    )


def main() -> None:
    """Computes and saves the pattern statistics and the pool sizes."""
    docs = pd.read_feather(MOTIF_DIR / "motif_docs.feather")
    patterns = pd.read_feather(MOTIF_DIR / "motif_patterns.feather")
    counts = pd.read_feather(MOTIF_DIR / "motif_counts.feather")

    # the tables are indexed by position below, so ids must equal row numbers
    assert (docs["doc_idx"].to_numpy() == np.arange(len(docs))).all()
    assert (patterns["pattern_id"].to_numpy() == np.arange(len(patterns))).all()

    pool = (
        docs.groupby(["domain", "source"])
        .agg(
            n_docs=("doc_idx", "size"),
            n_sents=("n_sents", "sum"),
            n_tokens=("n_tokens", "sum"),
        )
        .reset_index()
    )
    agg = _aggregate(docs, counts)

    stats = pd.concat(
        [
            _domain_stats(domain, agg, pool, patterns)
            for domain in sorted(docs["domain"].unique())
        ],
        ignore_index=True,
    )

    rounding = {c: 4 for c in stats if c.startswith("rate_")}
    rounding |= {c: 3 for c in stats if c.startswith("log2_ratio_")}
    rounding |= {c: 2 for c in stats if c.startswith("z_")}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUTPUT_DIR / "motif_stats.csv"
    stats.round(rounding).to_csv(stats_path, index=False)
    pool.to_csv(OUTPUT_DIR / "motif_pool.csv", index=False)

    print(
        f"Pool: {len(docs):,} documents, {int(pool['n_sents'].sum()):,} sentences, "
        f"{int(pool['n_tokens'].sum()):,} tokens"
    )
    print(f"Kept patterns (in >= {MIN_DOCS} documents of a domain):")
    print(stats.groupby("domain").size().to_string())
    print(f"Saved {stats_path.name}: {stats_path.stat().st_size / 1e6:.1f} MB")

    _print_top(stats)


if __name__ == "__main__":
    main()
