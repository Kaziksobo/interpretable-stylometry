"""Builds the motif explorer: one self-contained HTML file.

Reads the per-pattern statistics (compute_motif_stats.py) and writes
results/motifs/motif_explorer.html, which has its data embedded and needs no
server, network or libraries: open it in a browser, email it, or host it as a
static page. The page's markup, styling and logic live in
explorer_template.html; this script only fills in the data.

The embedded table holds every pattern that occurs in at least --min-docs
documents of its domain, as one array per column. Each row also records the
index of its parent (the same pattern cut off one level higher; see
mining.parent_pattern), which the page uses to fold echo variants under their
parents.

Inputs:
    results/motifs/motif_stats.csv
    results/motifs/motif_pool.csv
    src/analysis/motifs/explorer_template.html

Outputs:
    results/motifs/motif_explorer.html

Usage (from the project root):
    python src/analysis/motifs/build_explorer.py
    python src/analysis/motifs/build_explorer.py --min-docs 40
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from mining import parent_pattern

PROJECT_ROOT = next(
    p for p in Path(__file__).parents if (p / "pyproject.toml").exists()
)
STATS_PATH = PROJECT_ROOT / "results" / "motifs" / "motif_stats.csv"
POOL_PATH = PROJECT_ROOT / "results" / "motifs" / "motif_pool.csv"
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


def build_payload(min_docs: int) -> dict:
    """Builds the JSON-serialisable data the page embeds.

    Args:
        min_docs (int): A pattern is included if the documents containing it,
            summed over the three sources, number at least this many.

    Returns:
        dict: order, titles, minDocs, totals, pool and one column set per
            domain.
    """
    stats = pd.read_csv(STATS_PATH, keep_default_na=False)
    pool = pd.read_csv(POOL_PATH)

    docs_total = stats["docs_human"] + stats["docs_gpt"] + stats["docs_claude"]
    stats = stats[docs_total >= min_docs]

    order = [d for d in DOMAIN_TITLES if d in set(stats["domain"])]
    domains = {
        domain: _domain_columns(stats[stats["domain"] == domain].reset_index(drop=True))
        for domain in order
    }
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
        "totals": {"docs": int(pool["n_docs"].sum())},
        "pool": pool_by_domain,
        "domains": domains,
    }


def main(min_docs: int = DEFAULT_MIN_DOCS) -> None:
    """Builds the explorer HTML file.

    Args:
        min_docs (int): Document floor for including a pattern.

    Raises:
        ValueError: If the template does not contain the data placeholder.
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if "__DATA__" not in template:
        raise ValueError(f"{TEMPLATE_PATH.name} has no __DATA__ placeholder.")

    payload = build_payload(min_docs)
    # "</" inside a script element could end it early
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("__DATA__", data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")

    for domain, columns in payload["domains"].items():
        n_with_parent = sum(p >= 0 for p in columns["par"])
        print(f"{domain}: {len(columns['p']):,} patterns, {n_with_parent:,} with a parent")
    print(f"Saved {OUTPUT_PATH.name}: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the motif explorer HTML.")
    parser.add_argument(
        "--min-docs",
        type=int,
        default=DEFAULT_MIN_DOCS,
        help="include patterns used in at least this many documents of a domain",
    )
    main(min_docs=parser.parse_args().min_docs)
