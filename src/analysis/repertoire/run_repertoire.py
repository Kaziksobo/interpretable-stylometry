"""Repertoire diversity and concentration of syntactic patterns.

How varied is the set of phrase-building blocks a source uses, and how much of
its writing rests on the commonest few? A building block is a depth-2 pattern:
a node and its immediate parts, e.g. (NP (DT) (NN)). Human is compared with GPT
and with Claude in each domain, at the same amount of text.

Variety grows with the amount of text read, and the sources differ in size, so
every source is compared at the same budget (tokens by default): documents are
drawn at random, whole, until the budget is full, 1,000 times, and each metric
is averaged over the draws. Reuters is drawn by whole author, as RQ1 resamples
by author. Draws are subsets of one finite pool, so they overlap and their
spread understates the uncertainty; the standard error is the spread divided by
sqrt(1 - f), where f is the fraction of the pool in a draw. --null-check tests
that correction on random human-vs-human splits.

Four headline metrics, all tested (human vs each AI, one BH-FDR family):
    types            distinct blocks used at all (one-offs included)
    types_min5       distinct blocks used at least MIN_USES times
    effective_types  exp(Shannon entropy): the number of equally common blocks
                     that would give the same variety
    cover_50         share of all occurrences taken by the 50 commonest blocks
The share covered by the top-k blocks for other k (cover_k) is reported untested
and draws the coverage curves.

Inputs:
    data/processed/motifs/{motif_docs,motif_patterns,motif_counts}.feather

Outputs (sensitivity runs add a suffix such as _depth3_tokens):
    results/repertoire/repertoire_metrics.csv       one row per (domain, source)
    results/repertoire/repertoire_differences.csv   one row per comparison x metric

Usage (from the project root):
    python src/analysis/repertoire/run_repertoire.py
    python src/analysis/repertoire/run_repertoire.py --depth 3
    python src/analysis/repertoire/run_repertoire.py --budget occurrences
    python src/analysis/repertoire/run_repertoire.py --null-check
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

from src.analysis.shared.significance_utils import apply_fdr

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
MOTIF_DIR = PROJECT_ROOT / "data" / "processed" / "motifs"
OUTPUT_DIR = PROJECT_ROOT / "results" / "repertoire"

SOURCES = ["human", "gpt", "claude"]
AI_SOURCES = ["gpt", "claude"]
DOMAINS = ["essay", "reuter", "wp"]

N_DRAWS = 1000
BUDGET_FRACTION = 0.8  # of the smallest source's text in the domain
MIN_USES = 5  # a block counts towards types_min5 if used this often
COVER_K = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
SEED = 0

# the headline metrics, with the sign of (AI - human) that means a narrower
# repertoire: fewer blocks or a lower effective number, or more concentration
NARROWER = {"types": -1, "types_min5": -1, "effective_types": -1, "cover_50": +1}
SIZE_COLUMNS = ["documents", "tokens"]  # realised draw sizes, not metrics


def load_matrix(depth: int) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Builds the documents x patterns count matrix for one pattern depth.

    Args:
        depth (int): Levels in the patterns to keep, root included (2, 3 or 4).

    Returns:
        tuple[pd.DataFrame, sparse.csr_matrix]: motif_docs, and a matrix with
            one row per document (row number = doc_idx) and one column per
            pattern of that depth, holding its occurrence count.
    """
    docs = pd.read_feather(MOTIF_DIR / "motif_docs.feather")
    patterns = pd.read_feather(MOTIF_DIR / "motif_patterns.feather")
    counts = pd.read_feather(MOTIF_DIR / "motif_counts.feather")
    assert (docs["doc_idx"].to_numpy() == np.arange(len(docs))).all()

    keep = (patterns["depth"] == depth).to_numpy()
    column = np.cumsum(keep) - 1  # pattern_id -> matrix column, for kept patterns
    rows = counts[keep[counts["pattern_id"].to_numpy()]]
    matrix = sparse.csr_matrix(
        (
            rows["count"].to_numpy(),
            (rows["doc_idx"].to_numpy(), column[rows["pattern_id"].to_numpy()]),
        ),
        shape=(len(docs), int(keep.sum())),
    )
    return docs, matrix


def group_data(
    docs: pd.DataFrame, matrix: sparse.csr_matrix, idx: np.ndarray, by_author: bool
) -> dict:
    """Collects what one (domain, source) group needs for drawing.

    Args:
        docs (pd.DataFrame): motif_docs.
        matrix (sparse.csr_matrix): Output of load_matrix.
        idx (np.ndarray): Row numbers of the group's documents.
        by_author (bool): Group documents by author, for cluster-aware draws.

    Returns:
        dict: counts_t (patterns x documents, so a draw is one product),
            tokens and occurrences per document, and clusters (a list of
            document-index arrays, one per author, or None).
    """
    sub = docs.iloc[idx]
    clusters = None
    if by_author:
        author = sub["author"].to_numpy()
        assert not pd.isna(author).any()
        clusters = [np.flatnonzero(author == a) for a in np.unique(author)]
    return {
        "counts_t": matrix[idx].T.tocsr(),
        "tokens": sub["n_tokens"].to_numpy(),
        "occurrences": np.asarray(matrix[idx].sum(axis=1)).ravel(),
        "clusters": clusters,
    }


def draw_documents(
    rng: np.random.Generator,
    weight: np.ndarray,
    budget: float,
    clusters: list[np.ndarray] | None,
) -> np.ndarray:
    """Draws whole documents at random until the next would exceed the budget.

    With clusters (Reuters), whole authors are taken in random order, each
    author's documents in random order, so draws keep the author clustering;
    only the last author may be cut off part-way.

    Args:
        rng (np.random.Generator): Source of randomness.
        weight (np.ndarray): Each document's size (tokens or occurrences).
        budget (float): Total size to fill.
        clusters (list[np.ndarray] | None): Document indices per author.

    Returns:
        np.ndarray: Indices of the drawn documents. Never repeats a document
            and never overshoots the budget.
    """
    if clusters is None:
        order = rng.permutation(len(weight))
    else:
        order = np.concatenate(
            [rng.permutation(clusters[i]) for i in rng.permutation(len(clusters))]
        )
    n = int(np.searchsorted(np.cumsum(weight[order]), budget, side="right"))
    assert n > 0, "budget is smaller than a single document"
    return order[:n]


def sample_metrics(v: np.ndarray) -> dict[str, float]:
    """Computes the repertoire metrics for one sample's pattern counts.

    Args:
        v (np.ndarray): Occurrence count of every pattern in the sample.

    Returns:
        dict[str, float]: types, types_min5, effective_types, and cover_k for
            each k in COVER_K (the share of all occurrences taken by the k
            commonest patterns, or all of them if fewer than k occur).
    """
    v = v[v > 0].astype(float)
    p = v / v.sum()
    entropy = -(p * np.log(p)).sum()
    share = np.cumsum(np.sort(v)[::-1]) / v.sum()

    out = {
        "types": len(v),
        "types_min5": int((v >= MIN_USES).sum()),
        "effective_types": float(np.exp(entropy)),
    }
    out.update({f"cover_{k}": share[min(k, len(v)) - 1] for k in COVER_K})
    return out


def run_group(
    group: dict,
    weight_key: str,
    budget: float,
    n_draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Repeats the random draw and computes the metrics for each draw.

    Args:
        group (dict): Output of group_data.
        weight_key (str): "tokens" or "occurrences": what the budget measures.
        budget (float): Size of every draw.
        n_draws (int): Number of draws.
        rng (np.random.Generator): Source of randomness.

    Returns:
        pd.DataFrame: One row per draw: the metrics plus the realised number
            of documents and tokens.
    """
    weight = group[weight_key]
    rows = []
    for _ in range(n_draws):
        idx = draw_documents(rng, weight, budget, group["clusters"])
        mask = np.zeros(len(weight))
        mask[idx] = 1.0
        row = sample_metrics(group["counts_t"] @ mask)
        row["documents"] = len(idx)
        row["tokens"] = group["tokens"][idx].sum()
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(draws: pd.DataFrame, fraction: float) -> pd.Series:
    """Averages the draws and gives each metric a standard error.

    The draws are subsets of one finite pool, so they overlap and their spread
    is smaller than the uncertainty from having only this pool: a subset of
    fraction f has variance too small by a factor (1 - f). Dividing the spread
    by sqrt(1 - f) restores it. Exact for means, approximate for richness and
    entropy; --null-check measures how well it works.

    Args:
        draws (pd.DataFrame): Output of run_group.
        fraction (float): The share of the group's pool in each draw.

    Returns:
        pd.Series: <column>_mean for every column, and <metric>_se for every
            metric (the size columns get no standard error).
    """
    assert fraction < 1, "a draw must be smaller than the pool"
    se = draws.drop(columns=SIZE_COLUMNS).std(ddof=1) / np.sqrt(1 - fraction)
    return pd.concat([draws.mean().add_suffix("_mean"), se.add_suffix("_se")])


def compare(summary: pd.DataFrame) -> pd.DataFrame:
    """Compares each AI source with human on the headline metrics.

    Args:
        summary (pd.DataFrame): Metric means and standard errors, indexed by
            (domain, source).

    Returns:
        pd.DataFrame: One row per domain x AI x metric, with the difference,
            z, p, the FDR-corrected p across all rows (one family), and a
            verdict: "AI narrower", "AI broader" or "no difference".
    """
    rows = []
    for domain in DOMAINS:
        human = summary.loc[(domain, "human")]
        for ai in AI_SOURCES:
            other = summary.loc[(domain, ai)]
            for metric, narrower_sign in NARROWER.items():
                diff = other[f"{metric}_mean"] - human[f"{metric}_mean"]
                se = np.hypot(other[f"{metric}_se"], human[f"{metric}_se"])
                rows.append(
                    {
                        "domain": domain,
                        "comparison": f"human_vs_{ai}",
                        "metric": metric,
                        "human": human[f"{metric}_mean"],
                        "ai": other[f"{metric}_mean"],
                        "diff": diff,
                        "relative_diff": diff / human[f"{metric}_mean"],
                        "se": se,
                        "z": diff / se,
                        "p": 2 * norm.sf(abs(diff / se)),
                        "narrower": bool(np.sign(diff) == narrower_sign),
                    }
                )
    out = apply_fdr(pd.DataFrame(rows), "p")
    out["verdict"] = np.select(
        [~out["p_significant"], out["narrower"]],
        ["no difference", "AI narrower"],
        default="AI broader",
    )
    return out


def random_halves(
    rng: np.random.Generator, docs: pd.DataFrame, idx: np.ndarray, by_author: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Splits a group's documents at random into two halves (by author if asked).

    Args:
        rng (np.random.Generator): Source of randomness.
        docs (pd.DataFrame): motif_docs.
        idx (np.ndarray): Row numbers of the group's documents.
        by_author (bool): Keep each author's documents in one half.

    Returns:
        tuple[np.ndarray, np.ndarray]: Row numbers of the two halves.
    """
    if by_author:
        author = docs["author"].to_numpy()[idx]
        authors = rng.permutation(np.unique(author))
        first = np.isin(author, authors[: len(authors) // 2])
        return idx[first], idx[~first]
    shuffled = rng.permutation(idx)
    return shuffled[: len(idx) // 2], shuffled[len(idx) // 2 :]


def null_check(
    docs: pd.DataFrame,
    matrix: sparse.csr_matrix,
    domain: str,
    n_splits: int = 200,
    n_draws: int = 200,
) -> dict[str, float]:
    """Tests the intervals on data with no true difference.

    Splits the human documents of one domain at random into two halves, runs
    the whole procedure on each, and counts how often the halves differ at the
    5% level. Both halves are human, so about 5% of splits should.

    Args:
        docs (pd.DataFrame): motif_docs.
        matrix (sparse.csr_matrix): Output of load_matrix.
        domain (str): "essay", "reuter" or "wp".
        n_splits (int): Number of random splits.
        n_draws (int): Draws per half and split.

    Returns:
        dict[str, float]: For each headline metric, the share of splits with
            |z| > 1.96.
    """
    idx = np.flatnonzero(
        ((docs["domain"] == domain) & (docs["source"] == "human")).to_numpy()
    )
    by_author = domain == "reuter"
    rng = np.random.default_rng([SEED, DOMAINS.index(domain)])
    hits = dict.fromkeys(NARROWER, 0)
    for _ in range(n_splits):
        halves = random_halves(rng, docs, idx, by_author)
        groups = [group_data(docs, matrix, half, by_author) for half in halves]
        budget = BUDGET_FRACTION * min(g["tokens"].sum() for g in groups)
        a, b = (
            summarise(
                run_group(g, "tokens", budget, n_draws, rng),
                budget / g["tokens"].sum(),
            )
            for g in groups
        )
        for metric in NARROWER:
            z = (a[f"{metric}_mean"] - b[f"{metric}_mean"]) / np.hypot(
                a[f"{metric}_se"], b[f"{metric}_se"]
            )
            hits[metric] += abs(z) > 1.96
    return {metric: count / n_splits for metric, count in hits.items()}


def main(depth: int = 2, budget_by: str = "tokens", n_draws: int = N_DRAWS) -> None:
    """Runs the comparison and saves the metrics and differences.

    Args:
        depth (int): Pattern depth to analyse. Defaults to 2.
        budget_by (str): "tokens" (default) or "occurrences".
        n_draws (int): Random draws per group.
    """
    docs, matrix = load_matrix(depth)
    rows = []
    for domain_i, domain in enumerate(DOMAINS):
        groups = {}
        for source in SOURCES:
            idx = np.flatnonzero(
                ((docs["domain"] == domain) & (docs["source"] == source)).to_numpy()
            )
            groups[source] = group_data(docs, matrix, idx, domain == "reuter")
        budget = BUDGET_FRACTION * min(g[budget_by].sum() for g in groups.values())

        for source_i, source in enumerate(SOURCES):
            group = groups[source]
            rng = np.random.default_rng([SEED, domain_i, source_i])
            draws = run_group(group, budget_by, budget, n_draws, rng)
            fraction = budget / group[budget_by].sum()
            rows.append(
                summarise(draws, fraction).to_dict()
                | {
                    "domain": domain,
                    "source": source,
                    "budget": budget,
                    "sampled_fraction": fraction,
                    "n_draws": n_draws,
                }
            )

    summary = pd.DataFrame(rows).set_index(["domain", "source"])
    differences = compare(summary)

    tag = "" if (depth, budget_by) == (2, "tokens") else f"_depth{depth}_{budget_by}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / f"repertoire_metrics{tag}.csv")
    differences.to_csv(OUTPUT_DIR / f"repertoire_differences{tag}.csv", index=False)

    print(f"depth {depth}, budget by {budget_by}, {n_draws} draws per group\n")
    print(
        differences[
            ["domain", "comparison", "metric", "relative_diff", "p_fdr", "verdict"]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repertoire diversity and concentration of syntactic patterns."
    )
    parser.add_argument(
        "--depth", type=int, choices=(2, 3, 4), default=2, help="pattern depth"
    )
    parser.add_argument(
        "--budget",
        choices=("tokens", "occurrences"),
        default="tokens",
        help="what the equal-size budget measures",
    )
    parser.add_argument("--draws", type=int, default=N_DRAWS, help="draws per group")
    parser.add_argument(
        "--null-check",
        action="store_true",
        help="test the intervals on random human-vs-human splits instead",
    )
    args = parser.parse_args()

    if args.null_check:
        docs, matrix = load_matrix(args.depth)
        for domain in DOMAINS:
            rates = null_check(docs, matrix, domain)
            print(domain, {m: f"{100 * r:.1f}%" for m, r in rates.items()})
    else:
        main(depth=args.depth, budget_by=args.budget, n_draws=args.draws)
