"""
Analyze a pre-parsed corpus for syntactic motifs.

Usage:
    python analyze_corpus.py gutenberg_parsed.jsonl --limit 1000
    python analyze_corpus.py gutenberg_parsed.jsonl --author Dickens --min-terminals 3
    python analyze_corpus.py gutenberg_parsed.jsonl --author Dickens --score pmi --top 50
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict

from mining import count_terminal_nodes, extract_patterns_with_examples
from nltk import Tree


def analyze_parsed_corpus(
    sentences: list[dict],
    max_depth: int = 4,
    min_terminals: int = 2,
    verbose: bool = True,
    collect_examples: int = 2,
):
    """
    Extract and count syntactic motifs from pre-parsed sentences.

    Args:
        sentences: List of dicts with 'sentence' and 'parse' keys
        max_depth: Maximum subtree depth
        min_terminals: Minimum terminal (leaf) nodes per pattern
        verbose: Print progress
        collect_examples: Number of examples to collect per pattern

    Returns:
        Tuple of (counts Counter, total_sentences, examples dict)
    """
    counts = Counter()
    examples = defaultdict(list)
    total_sentences = 0
    errors = 0

    for i, record in enumerate(sentences):
        if verbose and i % 5000 == 0:
            print(
                f"\rProcessing sentence {i + 1}/{len(sentences)}...", end="", flush=True
            )

        try:
            tree = Tree.fromstring(record["parse"])
            sent_text = record["sentence"]
            total_sentences += 1

            for pattern, highlighted, sentence in extract_patterns_with_examples(
                tree, sent_text, max_depth=max_depth, min_terminals=min_terminals
            ):
                counts[pattern] += 1
                if len(examples[pattern]) < collect_examples:
                    if not any(highlighted == ex[0] for ex in examples[pattern]):
                        examples[pattern].append((highlighted, sentence))

        except Exception as e:
            errors += 1
            if verbose and errors <= 3:
                print(f"\n  Warning: Failed to process sentence {i}: {e}")
            continue

    if verbose:
        print(f"\rProcessed {total_sentences} sentences ({errors} errors)")

    return counts, total_sentences, dict(examples)


def extract_labels(pattern: str) -> list[str]:
    """Extract all node labels from a pattern string."""
    return re.findall(r"\(([A-Z$]+[A-Z0-9$-]*)", pattern)


def compute_pmi_vs_baseline(
    author_counts: Counter,
    author_total: int,
    baseline_counts: Counter,
    baseline_total: int,
    min_count: int = 5,
) -> dict[str, float]:
    """
    Compute PMI comparing author frequencies to corpus baseline.

    PMI(pattern, author) = log2(freq_author / freq_baseline)

    Positive = author overuses this pattern relative to baseline
    Negative = author underuses this pattern
    Zero = matches baseline

    Args:
        author_counts: Pattern counts for the author
        author_total: Total sentences for author
        baseline_counts: Pattern counts for balanced corpus
        baseline_total: Total sentences in baseline
        min_count: Minimum count in author corpus to include

    Returns:
        Dict mapping patterns to PMI scores
    """
    pmi_scores = {}

    for pattern, count in author_counts.items():
        if count < min_count:
            continue

        # Author frequency (per sentence)
        freq_author = count / author_total

        # Baseline frequency (per sentence)
        baseline_count = baseline_counts.get(pattern, 0)
        if baseline_count == 0:
            # Pattern doesn't appear in baseline - very distinctive!
            # Use a small smoothed value
            freq_baseline = 0.5 / baseline_total
        else:
            freq_baseline = baseline_count / baseline_total

        pmi = math.log2(freq_author / freq_baseline)
        pmi_scores[pattern] = pmi

    return pmi_scores


def highlight_in_sentence(sentence: str, words: str) -> str:
    """Highlight the motif words in the sentence using **bold** markers."""
    word_list = words.split()
    if not word_list:
        return sentence

    pattern_parts = [re.escape(w) for w in word_list]
    pattern = r"\b" + r'[\s,;:\'"]*'.join(pattern_parts) + r"\b"

    def replacer(match):
        return f"**{match.group(0)}**"

    return re.sub(pattern, replacer, sentence, count=1, flags=re.IGNORECASE)
