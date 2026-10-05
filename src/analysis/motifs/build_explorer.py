"""Builds the motif explorer: one self-contained HTML file.

Reads the per-pattern statistics (compute_motif_stats.py) and the sampled
examples (build_motif_examples.py) and writes results/motifs/motif_explorer.html,
which has its data embedded and needs no server, network or libraries: open it
in a browser, email it, or host it as a static page. The page's markup,
styling and logic live in explorer_template.html; this script only fills in the
data.

The embedded table holds every pattern that occurs in at least --min-docs
documents of its domain, as one array per column. Each row also records the
index of its parent (the same pattern cut off one level higher; see
mining.parent_pattern), which the page uses to fold echo variants under their
parents.

Examples are embedded only for the patterns that can come out near the top of a
view: in each domain, the --n-cut strongest patterns in each direction by z-score
and by log2 ratio, for GPT, for Claude, and for the two together (patterns where
both differ from human the same way). Embedding examples for every pattern would
make the file tens of megabytes. Each such pattern gets its first --per-source
examples per source (a fixed random draw; see build_motif_examples.py). Every
sentence is stored once and examples refer to it by index.

Inputs:
    results/motifs/motif_stats.csv
    results/motifs/motif_pool.csv
    data/processed/motifs/{motif_sentences,motif_examples}.feather
    src/analysis/motifs/explorer_template.html

Outputs:
    results/motifs/motif_explorer.html

Usage (from the project root):
    python src/analysis/motifs/build_explorer.py
    python src/analysis/motifs/build_explorer.py --n-cut 100 --per-source 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mining import parent_pattern

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
STATS_PATH = PROJECT_ROOT / "results" / "motifs" / "motif_stats.csv"
POOL_PATH = PROJECT_ROOT / "results" / "motifs" / "motif_pool.csv"
MOTIF_DIR = PROJECT_ROOT / "data" / "processed" / "motifs"
TEMPLATE_PATH = Path(__file__).with_name("explorer_template.html")
OUTPUT_PATH = PROJECT_ROOT / "results" / "motifs" / "motif_explorer.html"

DOMAIN_TITLES = {
    "essay": "Student essays",
    "reuter": "Reuters news",
    "wp": "WritingPrompts",
}
SOURCES = ["human", "gpt", "claude"]
SOURCE_KEYS = {"human": "H", "gpt": "G", "claude": "C"}
DEFAULT_MIN_DOCS = 20
DEFAULT_N_CUT = 250
DEFAULT_PER_SOURCE = 5


def _domain_columns(group: pd.DataFrame) -> dict[str, list]:
    """Turns one domain's rows into the column arrays the page reads.

    Args:
        group (pd.DataFrame): One domain's rows of motif_stats.csv, already
            filtered to the document floor.

    Returns:
        dict[str, list]: Column arrays, all the same length. p pattern, d
            depth, c/d/r + H/G/C the count, document count and rate of each
            source, lG/lC and zG/zC the log2 ratio and z-score of GPT and
            Claude, and par the row index of the parent pattern (-1 if the
            parent is not in the table).
    """
    patterns = group["pattern"].tolist()
    row_of = {pattern: i for i, pattern in enumerate(patterns)}
    parents = [parent_pattern(p) for p in patterns]

    columns: dict[str, list] = {"p": patterns, "d": group["depth"].tolist()}
    for source in SOURCES:
        key = SOURCE_KEYS[source]
        columns[f"c{key}"] = group[f"count_{source}"].tolist()
        columns[f"d{key}"] = group[f"docs_{source}"].tolist()
        columns[f"r{key}"] = group[f"rate_{source}"].tolist()
    for ai in ("gpt", "claude"):
        key = SOURCE_KEYS[ai]
        columns[f"l{key}"] = group[f"log2_ratio_{ai}"].tolist()
        columns[f"z{key}"] = group[f"z_{ai}"].tolist()
    columns["par"] = [row_of.get(p, -1) if p else -1 for p in parents]
    return columns


def _select_rows(group: pd.DataFrame, n_cut: int) -> set[int]:
    """Picks the rows of one domain that get embedded examples.

    Mirrors the views the page can show: for GPT, for Claude, and for the two
    agreeing in direction (scored by the weaker of the two, as in the page),
    the n_cut highest and n_cut lowest rows by z-score and by log2 ratio.

    Args:
        group (pd.DataFrame): One domain's rows, with a 0-based index equal to
            the page's row index.
        n_cut (int): How many rows to take at each end of each ranking.

    Returns:
        set[int]: Row indices.
    """
    z_gpt = group["z_gpt"].to_numpy()
    z_claude = group["z_claude"].to_numpy()
    l_gpt = group["log2_ratio_gpt"].to_numpy()
    l_claude = group["log2_ratio_claude"].to_numpy()

    agree = z_gpt * z_claude > 0
    both_z = np.where(agree, np.sign(z_gpt) * np.minimum(abs(z_gpt), abs(z_claude)), np.nan)
    both_l = np.where(agree, np.sign(l_gpt) * np.minimum(abs(l_gpt), abs(l_claude)), np.nan)

    chosen: set[int] = set()
    for scores in (z_gpt, z_claude, both_z, l_gpt, l_claude, both_l):
        ranked = pd.Series(scores)  # nlargest and nsmallest skip the NaNs
        chosen.update(int(i) for i in ranked.nlargest(n_cut).index)
        chosen.update(int(i) for i in ranked.nsmallest(n_cut).index)
    return chosen


def _build_examples(
    groups: dict[str, pd.DataFrame], n_cut: int, per_source: int
) -> tuple[dict, list[str]]:
    """Collects the embedded examples and the sentences they refer to.

    Args:
        groups (dict[str, pd.DataFrame]): Each domain's table, 0-based index.
        n_cut (int): See _select_rows.
        per_source (int): Examples embedded per source and pattern.

    Returns:
        tuple[dict, list[str]]: examples[domain][row][H|G|C] as a list of
            [sentence index, start, end, *slot boundaries], and the sentences
            as space-separated token strings.

    Raises:
        ValueError: If the sentence table is not indexed by sent_id.
    """
    sentences = pd.read_feather(MOTIF_DIR / "motif_sentences.feather")
    if not (sentences["sent_id"].to_numpy() == np.arange(len(sentences))).all():
        raise ValueError("motif_sentences is not in sent_id order.")

    examples = pd.read_feather(MOTIF_DIR / "motif_examples.feather")
    examples = examples[examples["rank"] < per_source].copy()
    examples["domain"] = examples["domain"].astype(str)
    examples["source"] = examples["source"].astype(str)

    selected = pd.concat(
        [
            pd.DataFrame(
                {
                    "domain": domain,
                    "pattern_id": group["pattern_id"].to_numpy()[sorted(rows)],
                    "row": sorted(rows),
                }
            )
            for domain, group in groups.items()
            for rows in [_select_rows(group, n_cut)]
        ],
        ignore_index=True,
    )
    merged = examples.merge(selected, on=["domain", "pattern_id"]).sort_values(
        ["domain", "row", "source", "rank"], ignore_index=True
    )

    used = np.sort(merged["sent_id"].unique())
    new_index = {int(s): i for i, s in enumerate(used)}
    token_strings = sentences["tokens"].to_numpy()[used].tolist()

    out: dict = {}
    for r in merged.itertuples(index=False):
        bounds = [int(b) for b in r.bounds.split(",")] if r.bounds else []
        entry = [new_index[int(r.sent_id)], int(r.start), int(r.end), *bounds]
        by_row = out.setdefault(r.domain, {}).setdefault(str(r.row), {})
        by_row.setdefault(SOURCE_KEYS[r.source], []).append(entry)
    return out, token_strings


def build_payload(min_docs: int, n_cut: int, per_source: int) -> dict:
    """Builds the JSON-serialisable data the page embeds.

    Args:
        min_docs (int): A pattern is included if the documents containing it,
            summed over the three sources, number at least this many.
        n_cut (int): Examples are embedded for this many patterns at each end
            of each ranking (see _select_rows).
        per_source (int): Examples embedded per source and pattern.

    Returns:
        dict: order, titles, minDocs, totals, pool, one column set per
            domain, the examples and the sentences.
    """
    stats = pd.read_csv(STATS_PATH, keep_default_na=False)
    pool = pd.read_csv(POOL_PATH)

    docs_total = stats["docs_human"] + stats["docs_gpt"] + stats["docs_claude"]
    stats = stats[docs_total >= min_docs]

    order = [d for d in DOMAIN_TITLES if d in set(stats["domain"])]
    groups = {
        domain: stats[stats["domain"] == domain].reset_index(drop=True)
        for domain in order
    }
    domains = {domain: _domain_columns(group) for domain, group in groups.items()}
    examples, sentences = _build_examples(groups, n_cut, per_source)

    pool_by_domain = {
        domain: {
            row.source: {
                "n_docs": int(row.n_docs),
                "n_sents": int(row.n_sents),
                "n_tokens": int(row.n_tokens),
            }
            for row in group.itertuples()
        }
        for domain, group in pool.groupby("domain")
    }
    return {
        "order": order,
        "titles": {d: DOMAIN_TITLES[d] for d in order},
        "minDocs": min_docs,
        "nCut": n_cut,
        "perSource": per_source,
        "totals": {"docs": int(pool["n_docs"].sum())},
        "pool": pool_by_domain,
        "domains": domains,
        "ex": examples,
        "sents": sentences,
    }


def main(
    min_docs: int = DEFAULT_MIN_DOCS,
    n_cut: int = DEFAULT_N_CUT,
    per_source: int = DEFAULT_PER_SOURCE,
) -> None:
    """Builds the explorer HTML file.

    Args:
        min_docs (int): Document floor for including a pattern.
        n_cut (int): Examples are embedded for this many patterns at each end
            of each ranking.
        per_source (int): Examples embedded per source and pattern.

    Raises:
        ValueError: If the template does not contain the data placeholder.
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if "__DATA__" not in template:
        raise ValueError(f"{TEMPLATE_PATH.name} has no __DATA__ placeholder.")

    payload = build_payload(min_docs, n_cut, per_source)
    # "</" inside a script element could end it early
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("__DATA__", data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")

    for domain, columns in payload["domains"].items():
        n_rows = len(payload["ex"].get(domain, {}))
        print(f"{domain}: {len(columns['p']):,} patterns, {n_rows:,} with examples")
    sent_bytes = sum(len(s) + 4 for s in payload["sents"])
    print(f"{len(payload['sents']):,} sentences embedded (~{sent_bytes / 1e6:.1f} MB)")
    print(f"Saved {OUTPUT_PATH.name}: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the motif explorer HTML.")
    parser.add_argument(
        "--min-docs",
        type=int,
        default=DEFAULT_MIN_DOCS,
        help="include patterns used in at least this many documents of a domain",
    )
    parser.add_argument(
        "--n-cut",
        type=int,
        default=DEFAULT_N_CUT,
        help="embed examples for this many patterns at each end of each ranking",
    )
    parser.add_argument(
        "--per-source",
        type=int,
        default=DEFAULT_PER_SOURCE,
        help="examples embedded per source and pattern",
    )
    args = parser.parse_args()
    main(min_docs=args.min_docs, n_cut=args.n_cut, per_source=args.per_source)
