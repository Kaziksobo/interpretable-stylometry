"""Feng et al. (2012) sentence-type classifiers for RQ1.

Implements Algorithm 1 (Sentence Type-I: SIMPLE / COMPLEX / COMPOUND /
COMPLEX-COMPOUND) and Algorithm 2 (Sentence Type-II: LOOSE / PERIODIC)
from Feng, Banerjee & Choi (2012), "Characterizing Stylistic Elements
in Syntactic Structure", applied to the constituency parses produced
by constituency_parse.py.

Both algorithms operate on a single parse tree t(Nr) rooted at the
sentence node Nr, using two structures defined in the paper:

    L^top   the sequence of nodes directly below the root (the
            "top structural level"), e.g. [NP VP .]
    Omega(N) the set of all nodes in the subtree rooted at N
            (N itself, plus every descendant)

CRITICAL ASSUMPTION: both algorithms below assume the `tree` passed
in *is* Nr — i.e. the parse tree's root label is the sentence's own
"S", with no ROOT/TOP wrapper node above it. This was checked
manually against a sample of the Ghostbuster corpus's parse trees
before this was written, but it is not re-validated here at runtime.
If a tree with a ROOT/TOP wrapper were ever fed in, `tree`'s only
direct child would be that real "S" node, so feng_algo_1 would see
"S" in l_top on every single sentence and misclassify everything as
COMPOUND or COMPLEX-COMPOUND — silently, with no exception raised.
See the note inside feng_algo_1 below.

Output schema (constituency_parses.feather columns, plus):
    feng_algo_1 : str  - one of SIMPLE, COMPLEX, COMPOUND,
                          COMPLEX-COMPOUND, OTHER, or ERROR
    feng_algo_2 : str  - one of LOOSE, PERIODIC, OTHER, or ERROR
                          (ERROR rows are sentences whose parse_str
                          failed to parse back into an nltk.Tree)
"""

from pathlib import Path

import pandas as pd
from nltk import Tree
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_parses.feather"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_features.feather"


def load_data(data_path: Path) -> pd.DataFrame:
    """Loads constituency parses from a feather file.

    Args:
        data_path: Path to a .feather file with at least a
            "parse_str" column of bracketed constituency parses.

    Returns:
        The loaded DataFrame.
    """
    return pd.read_feather(data_path)


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Saves a DataFrame to a feather file.

    Args:
        df: DataFrame to save.
        output_path: Destination .feather path.
    """
    df.to_feather(output_path)


def feng_algo_1(tree: Tree) -> str:
    # sourcery skip: assign-if-exp, merge-else-if-into-elif,
    # remove-unnecessary-else, swap-if-else-branches
    """Sentence Type-I classification (Feng et al. 2012, Algorithm 1).

    Distinguishes SIMPLE / COMPLEX / COMPOUND / COMPLEX-COMPOUND
    sentences using two structural signals:

      - an independent clause beyond the main one, detected as an
        extra "S" node at the top structural level (L^top — the
        nodes directly below the root);
      - a dependent clause, detected as an "SBAR" node anywhere in
        the tree (Omega(Nr) — every node in the whole tree, root
        included).

    Deliberately mirrors the paper's pseudocode line-for-line, hence
    the `# sourcery skip` directives above: collapsing the nested
    if/else into more idiomatic Python (e.g. a ternary, or merging
    the else-if) would make it harder to check this against
    Algorithm 1 by eye, which matters more here than brevity.

    IMPORTANT: assumes `tree`'s root label is the sentence's own "S"
    (i.e. Nr itself), with no ROOT/TOP wrapper above it — see the
    module docstring. If that assumption is ever violated, this
    misclassifies silently rather than raising.

    Args:
        tree: Parse tree t(Nr), rooted directly at the sentence's "S".

    Returns:
        One of "SIMPLE", "COMPLEX", "COMPOUND", "COMPLEX-COMPOUND",
        or "OTHER" (neither an extra S nor a VP at the top level —
        e.g. a sentence fragment).
    """
    # L^top: labels of the direct children of the root (the "top
    # structural level"), e.g. ["NP", "VP", "."]. Only Tree children
    # are kept; leaf strings (terminal tokens) are excluded.
    l_top: list[str] = [c.label() for c in tree if isinstance(c, Tree)]
    # Omega(Nr): labels of every node in the tree, root included.
    # tree.subtrees() yields the tree itself plus all descendants.
    all_labels: set[str] = {s.label() for s in tree.subtrees()}
    if "S" in l_top:
        # An extra S beyond the root -> an independent clause beyond
        # the main one, i.e. compound coordination.
        if "SBAR" not in all_labels:
            return "COMPOUND"
        else:
            return "COMPLEX-COMPOUND"
    else:
        if "VP" in l_top:
            # A single main predicate at the top level, i.e. just one
            # independent clause.
            if "SBAR" not in all_labels:
                return "SIMPLE"
            else:
                return "COMPLEX"
    return "OTHER"


def feng_algo_2(tree: Tree) -> str:
    # sourcery skip: merge-else-if-into-elif, swap-if-else-branches, while-to-for
    """Sentence Type-II classification (Feng et al. 2012, Algorithm 2).

    Distinguishes LOOSE / PERIODIC sentences by walking the top
    structural level L^top = [N_1, ..., N_lambda] in order and
    returning on the first node N_k whose own subtree Omega(N_k)
    contains an "S" or "SBAR":

      - if N_k is a VP, the embedded clause sits inside the
        predicate (typically a trailing subordinate clause) ->
        LOOSE;
      - if N_k is anything else (typically a fronted subordinate
        clause preceding the main predicate) -> PERIODIC.

    Note the asymmetry with feng_algo_1: there, "SBAR" is checked
    against Omega(Nr) — the whole tree. Here it's checked against
    Omega(N_k) — just that one top-level child's own subtree — for
    each child in turn, stopping at the first hit. The same S/SBAR
    condition therefore means something different in each algorithm
    purely because of *where* it's allowed to fire.

    Deliberately mirrors the paper's pseudocode line-for-line (see
    feng_algo_1's docstring for why the sourcery-skip directives
    above are there).

    Args:
        tree: Parse tree t(Nr), rooted directly at the sentence's "S"
            (same assumption as feng_algo_1; see module docstring).

    Returns:
        One of "LOOSE", "PERIODIC", or "OTHER" (no top-level child's
        subtree contains an S or SBAR).
    """
    l_top = [c for c in tree if isinstance(c, Tree)]  # L^top, in order
    lam = len(l_top)  # lambda (|L^top|); named `lam` since `lambda` is a keyword
    k = 0
    while k < lam:
        node = l_top[k]
        # Omega(L^top_k): every node in this one child's own subtree,
        # including the child itself.
        labels_in_subtree = {s.label() for s in node.subtrees()}

        if node.label() != "VP":
            if "S" in labels_in_subtree or "SBAR" in labels_in_subtree:
                return "PERIODIC"
        else:
            if "S" in labels_in_subtree or "SBAR" in labels_in_subtree:
                return "LOOSE"
        k += 1
    return "OTHER"


def classify_sentence(parse_str: str) -> tuple[str, str]:
    """Parses a bracketed constituency string and classifies it.

    Parses `parse_str` into an nltk.Tree once and runs both Feng et
    al. algorithms against that same tree, rather than having each
    algorithm parse the string independently.

    Args:
        parse_str: A bracketed constituency parse, e.g.
            "(S (NP ...) (VP ...))".

    Returns:
        A (feng_algo_1_result, feng_algo_2_result) tuple. If
        `parse_str` doesn't parse into a valid tree — malformed
        bracketing raises ValueError, certain degenerate strings
        raise IndexError, both from Tree.fromstring — returns
        ("ERROR", "ERROR") instead of propagating the exception, so
        a handful of bad rows don't kill the whole run.
    """
    try:
        tree = Tree.fromstring(parse_str)
    except (ValueError, IndexError):
        return "ERROR", "ERROR"
    return feng_algo_1(tree), feng_algo_2(tree)


def main() -> None:
    """Classifies every sentence in constituency_parses.feather.

    Runs classify_sentence() over the "parse_str" column and adds two
    new columns, "feng_algo_1" and "feng_algo_2", before saving the
    result to OUTPUT_PATH.

    Uses a plain list comprehension wrapped in tqdm() rather than
    `df["parse_str"].progress_apply(...)`: tqdm.pandas()'s
    progress_apply has been incompatible with recent pandas versions
    (an `is_builtin_func` import error), and this sidesteps that
    entirely while keeping the progress bar.
    """
    df = load_data(INPUT_PATH)

    results = [
        classify_sentence(parse_str)
        for parse_str in tqdm(
            df["parse_str"], desc="Classifying sentences (Feng et al. 2012)"
        )
    ]

    df["feng_algo_1"], df["feng_algo_2"] = zip(*results)

    if n_errors := (df["feng_algo_1"] == "ERROR").sum():
        print(f"Warning: {n_errors} rows failed to parse and were marked ERROR.")

    save_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
