from pathlib import Path

import pandas as pd
from nltk import Tree
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_parses.feather"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_features.feather"


def load_data(data_path: Path):
    """Loads data from feather file and returns dataframe"""
    return pd.read_feather(data_path)


def save_data(df: pd.DataFrame, output_path: Path):
    """Saves dataframe to feather file"""
    df.to_feather(output_path)


def feng_algo_1(tree: Tree) -> str:
    # sourcery skip: assign-if-exp, merge-else-if-into-elif,
    # remove-unnecessary-else, swap-if-else-branches
    l_top: list[str] = [c.label() for c in tree if isinstance(c, Tree)]
    all_labels: set[str] = {s.label() for s in tree.subtrees()}
    if "S" in l_top:
        if "SBAR" not in all_labels:
            return "COMPOUND"
        else:
            return "COMPLEX-COMPOUND"
    else:
        if "VP" in l_top:
            if "SBAR" not in all_labels:
                return "SIMPLE"
            else:
                return "COMPLEX"
    return "OTHER"


def feng_algo_2(tree: Tree) -> str:
    # sourcery skip: merge-else-if-into-elif, swap-if-else-branches, while-to-for
    l_top = [c for c in tree if isinstance(c, Tree)]
    lam = len(l_top)
    k = 0
    while k < lam:
        node = l_top[k]
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
    try:
        tree = Tree.fromstring(parse_str)
    except (ValueError, IndexError):
        return "ERROR", "ERROR"
    return feng_algo_1(tree), feng_algo_2(tree)


def main():
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
