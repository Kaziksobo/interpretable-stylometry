import multiprocessing as mp
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from analyze_corpus import compute_pmi_vs_baseline, highlight_in_sentence
from mining import extract_patterns_with_examples
from nltk import Tree
from tqdm import tqdm


def _process_chunk_with_examples(
    pairs: list[tuple[str, str]],
) -> tuple[Counter, dict[str, list[dict[str, str]]]]:
    """
    Worker process: Extracts pattern counts AND illustrative text examples.
    """
    counts = Counter()
    examples = defaultdict(list)

    for parse_str, sentence_text in pairs:
        if not isinstance(parse_str, str) or not isinstance(sentence_text, str):
            continue

        try:
            tree = Tree.fromstring(parse_str)
            # Extracts pattern, the matched words, and full sentence
            for pattern, highlighted, sentence in extract_patterns_with_examples(
                tree, sentence_text, max_depth=4, min_depth=2, min_terminals=2
            ):
                counts[pattern] += 1

                # Keep a small pool of examples per chunk to manage memory
                if len(examples[pattern]) < 2 and all(
                    ex["words"] != highlighted for ex in examples[pattern]
                ):
                    examples[pattern].append(
                        {"words": highlighted, "sentence": sentence}
                    )
        except Exception:
            continue

    return counts, dict(examples)


def extract_data_parallel(
    df_subset: pd.DataFrame, parse_col: str, text_col: str
) -> tuple[Counter, int, dict[str, list[dict[str, str]]]]:
    """
    Coordinates multi-core execution to gather counts and text examples.
    """
    total_sentences = len(df_subset)
    if total_sentences == 0:
        return Counter(), 0, {}

    # Pair the parse strings and text sentences together
    pairs = list(zip(df_subset[parse_col].tolist(), df_subset[text_col].tolist()))

    n_cores = max(1, mp.cpu_count() - 1)
    chunk_size = max(1, len(pairs) // n_cores)
    chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    master_counts = Counter()
    master_examples = defaultdict(list)

    with mp.Pool(processes=n_cores) as pool:
        # Wrap the imap iterator with tqdm for a visual progress bar
        results_iterator = tqdm(
            pool.imap(_process_chunk_with_examples, chunks),
            total=len(chunks),
            desc="  Processing",
            leave=False,  # Removes the bar once finished to keep terminal clean
            ncols=80,
        )

        for count_res, example_res in results_iterator:
            master_counts |= count_res

            # Merge text examples safely
            for pattern, ex_list in example_res.items():
                if len(master_examples[pattern]) < 3:
                    master_examples[pattern].extend(
                        ex_list[: 3 - len(master_examples[pattern])]
                    )

    return master_counts, total_sentences, dict(master_examples)


def analyze_and_compare_domain(
    df: pd.DataFrame, domain: str, parse_col: str, text_col: str, min_count: int = 15
):
    print("\n" + "=" * 80)
    print(f" RUNNING STRONG-FEATURE ANALYSIS FOR DOMAIN: {domain.upper()}")
    print("=" * 80)

    domain_df = df[df["domain"] == domain]

    # 1. Establish Human Baseline
    print("[1/3] Processing Human Baseline...")
    human_df = domain_df[domain_df["source"] == "human"]
    human_counts, human_total, human_examples = extract_data_parallel(
        human_df, parse_col, text_col
    )

    # 2. Extract GPT Data
    print("[2/3] Processing GPT Patterns...")
    gpt_df = domain_df[domain_df["source"] == "gpt"]
    gpt_counts, gpt_total, gpt_examples = extract_data_parallel(
        gpt_df, parse_col, text_col
    )

    # 3. Extract Claude Data
    print("[3/3] Processing Claude Patterns...")
    claude_df = domain_df[domain_df["source"] == "claude"]
    claude_counts, claude_total, claude_examples = extract_data_parallel(
        claude_df, parse_col, text_col
    )

    # Compute PMI profiles relative to Human Baseline
    gpt_pmi = compute_pmi_vs_baseline(
        gpt_counts, gpt_total, human_counts, human_total, min_count
    )
    claude_pmi = compute_pmi_vs_baseline(
        claude_counts, claude_total, human_counts, human_total, min_count
    )

    shared_patterns = set(gpt_pmi.keys()) & set(claude_pmi.keys())

    comparison_report = []
    for pattern in shared_patterns:
        g_pmi = gpt_pmi[pattern]
        c_pmi = claude_pmi[pattern]

        # Aggregate available text examples across splits for display
        all_ex = (
            gpt_examples.get(pattern, [])
            + claude_examples.get(pattern, [])
            + human_examples.get(pattern, [])
        )

        comparison_report.append(
            {
                "pattern": pattern,
                "gpt_pmi": g_pmi,
                "claude_pmi": c_pmi,
                "model_gap": abs(g_pmi - c_pmi),  # Distance between the two AIs
                "gpt_deviation": abs(g_pmi),  # Absolute distance from humans
                "claude_deviation": abs(c_pmi),  # Absolute distance from humans
                "examples": all_ex[:2],
            }
        )

    return comparison_report


def smart_truncate(text: str, max_len: int = 140) -> str:
    """Truncates text but guarantees the **highlighted** portion remains visible."""
    if len(text) <= max_len:
        return text

    start_idx = text.find("**")

    # If no highlight found, just truncate normally
    if start_idx == -1:
        return text[:max_len] + "..."

    # If the highlight is already near the beginning, truncate the end
    if start_idx < (max_len // 2):
        return text[:max_len] + "..."

    # Otherwise, create a window with the highlight in the middle
    start_window = max(0, start_idx - (max_len // 2))
    end_window = start_window + max_len

    return "..." + text[start_window:end_window] + "..."


def main():
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "constituency_parses.feather"
    OUTPUT_PATH = PROJECT_ROOT / "results" / "stylometric_report.txt"
    print("Loading Feather file...")
    # Update to your actual dataset file path
    df = pd.read_feather(INPUT_PATH)

    PARSE_COL = "parse_str"
    TEXT_COL = "sent_text"
    domains = ["reuter", "wp", "essay"]

    # Open the file in write mode (this will overwrite any older versions of the report)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("=== AI vs HUMAN STYLOMETRIC ANALYSIS REPORT ===\n")

        for domain in domains:
            start_time = time.time()

            # The terminal acts as your progress monitor
            print(f"\nStarting deep analysis on domain: '{domain.upper()}'...")

            report = analyze_and_compare_domain(
                df, domain, PARSE_COL, TEXT_COL, min_count=15
            )

            if not report:
                msg = f"No shared patterns found for domain '{domain}'.\n"
                print(msg)
                f.write(msg)
                continue

            # --- 1. STRONGEST MODEL DIVERGENCE (GPT vs Claude) ---
            divergent_sorted = sorted(
                report, key=lambda x: x["model_gap"], reverse=True
            )

            f.write(
                (
                    "\n\n### TOP DIVERGENCE: WHERE GPT "
                    f"AND CLAUDE DISAGREE MOST ({domain.upper()})\n"
                )
            )
            f.write("-" * 80 + "\n")
            for item in divergent_sorted[:4]:
                f.write(f"\nPattern: {item['pattern']}\n")
                f.write(f"  -> GPT PMI:    {item['gpt_pmi']:+.3f}\n")
                f.write(f"  -> Claude PMI: {item['claude_pmi']:+.3f}\n")
                f.write(f"  -> Gap Size:   {item['model_gap']:.3f}\n")

                for i, ex in enumerate(item["examples"]):
                    highlighted = highlight_in_sentence(ex["sentence"], ex["words"])
                    f.write(f"     ex{i + 1}: {smart_truncate(highlighted, 140)}...\n")

            # --- 2. STRONGEST GPT SIGNATURES ---
            gpt_strongest = sorted(
                report, key=lambda x: x["gpt_deviation"], reverse=True
            )

            f.write(
                (
                    "\n\n### STRONGEST GPT SIGNATURES: "
                    f"FURTHEST FROM HUMANS ({domain.upper()})\n"
                )
            )
            f.write("-" * 80 + "\n")
            for item in gpt_strongest[:4]:
                f.write(f"\nPattern: {item['pattern']}\n")
                f.write(f"  -> GPT PMI:    {item['gpt_pmi']:+.3f} (Max Deviation)\n")
                f.write(f"  -> Claude PMI: {item['claude_pmi']:+.3f}\n")

                for i, ex in enumerate(item["examples"]):
                    highlighted = highlight_in_sentence(ex["sentence"], ex["words"])
                    f.write(f"     ex{i + 1}: {smart_truncate(highlighted, 140)}...\n")

            # --- 3. STRONGEST CLAUDE SIGNATURES ---
            claude_strongest = sorted(
                report, key=lambda x: x["claude_deviation"], reverse=True
            )

            f.write(
                (
                    "\n\n### STRONGEST CLAUDE SIGNATURES: "
                    f"FURTHEST FROM HUMANS ({domain.upper()})\n"
                )
            )
            f.write("-" * 80 + "\n")
            for item in claude_strongest[:4]:
                f.write(f"\nPattern: {item['pattern']}\n")
                f.write(f"  -> Claude PMI: {item['claude_pmi']:+.3f} (Max Deviation)\n")
                f.write(f"  -> GPT PMI:    {item['gpt_pmi']:+.3f}\n")

                for i, ex in enumerate(item["examples"]):
                    highlighted = highlight_in_sentence(ex["sentence"], ex["words"])
                    f.write(f"     ex{i + 1}: {smart_truncate(highlighted, 140)}...\n")

            domain_time = time.time() - start_time
            f.write(f"\n[ Domain '{domain}' completed in {domain_time:.1f}s ]\n")

            # Let the terminal know this chunk is safely written
            print(
                (
                    f"-> Finished '{domain.upper()}' in {domain_time:.1f}s."
                    f" Results appended to {OUTPUT_PATH}."
                )
            )

    print(f"\nDone! Full analysis successfully saved to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    mp.freeze_support()
    main()
