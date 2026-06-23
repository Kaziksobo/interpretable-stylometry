"""
Subtree pattern extraction and counting.

Extracts induced subtrees from constituency parse trees and counts their frequencies.
"""

from collections import Counter
from typing import Iterator

from nltk import Tree


def extract_induced_subtrees(
    tree: Tree, max_depth: int = 3, min_depth: int = 2
) -> Iterator[Tree]:
    """
    Extract all induced subtrees from a parse tree up to a given depth.

    An induced subtree includes a node and all of its descendants down to
    some depth. We enumerate all subtrees rooted at each non-terminal node.

    Args:
        tree: NLTK Tree (constituency parse)
        max_depth: Maximum depth of extracted subtrees (1 = just the node,
                   2 = node + children, 3 = node + children + grandchildren)
        min_depth: Minimum depth of extracted subtrees (default 2 to exclude
                   single POS tags)

    Yields:
        Tree objects representing subtrees (with terminals stripped)
    """
    if not isinstance(tree, Tree):
        return

    # Extract subtrees rooted at this node for depths min_depth to max_depth
    for depth in range(min_depth, max_depth + 1):
        subtree = _extract_subtree_at_depth(tree, depth)
        if subtree is not None:
            yield subtree

    # Recurse into children
    for child in tree:
        if isinstance(child, Tree):
            yield from extract_induced_subtrees(child, max_depth, min_depth)


def _extract_subtree_at_depth(tree: Tree, depth: int) -> Tree | None:
    """
    Extract an induced subtree rooted at tree with exact depth limit.

    Args:
        tree: Root of the subtree
        depth: How many levels to include (1 = just this node's label)

    Returns:
        A new Tree with structure up to the given depth, terminals abstracted
    """
    if not isinstance(tree, Tree):
        return None

    if depth == 1:
        # Just return a leaf node with this label (no children)
        return Tree(tree.label(), [])

    # Include children up to depth-1
    children = []
    for child in tree:
        if isinstance(child, Tree):
            child_subtree = _extract_subtree_at_depth(child, depth - 1)
            if child_subtree is not None:
                children.append(child_subtree)
        else:
            # Terminal: represent as empty node with POS tag
            # We skip terminals entirely since we want abstract patterns
            pass

    if not children and depth > 1:
        # This node has no non-terminal children, depth-1 would be empty
        # Still return the node itself
        return Tree(tree.label(), [])

    return Tree(tree.label(), children)


def canonicalize(tree: Tree) -> str:
    """
    Convert a tree to its canonical string representation.

    The canonical form is a parenthesized string that captures the tree structure:
        (S (NP) (VP))
        (NP (DT) (NN))
        (VP (VBD) (NP))

    Args:
        tree: NLTK Tree object

    Returns:
        Canonical string representation
    """
    if not isinstance(tree, Tree):
        return str(tree)

    if len(tree) == 0:
        # Leaf node (either terminal or abstracted non-terminal)
        return f"({tree.label()})"

    children_str = " ".join(canonicalize(child) for child in tree)
    return f"({tree.label()} {children_str})"


def count_terminal_nodes(pattern: str) -> int:
    """
    Count terminal (leaf) nodes in a pattern string.

    Terminal nodes are represented as (TAG) with no children.
    E.g., in "(NP (DT) (NN))", DT and NN are terminals (2 total).
    In "(PP (IN) (NP (DT) (NN)))", IN, DT, NN are terminals (3 total).

    Args:
        pattern: Canonical pattern string

    Returns:
        Number of terminal nodes
    """
    import re

    # Terminal nodes match pattern: opening paren, label, closing paren
    # with no space (which would indicate children)
    terminals = re.findall(r"\([A-Z$][A-Z0-9$-]*\)", pattern)
    return len(terminals)


def extract_patterns(
    tree: Tree, max_depth: int = 4, min_depth: int = 2, min_terminals: int = 2
) -> Iterator[str]:
    """
    Extract canonical pattern strings from a parse tree.

    Convenience function that combines extraction and canonicalization.

    Args:
        tree: NLTK Tree
        max_depth: Maximum subtree depth
        min_depth: Minimum subtree depth (default 2 excludes single tags)
        min_terminals: Minimum number of terminal (leaf) nodes in the pattern

    Yields:
        Canonical pattern strings
    """
    for subtree in extract_induced_subtrees(tree, max_depth, min_depth):
        pattern = canonicalize(subtree)
        # Count only terminal (leaf) nodes
        terminal_count = count_terminal_nodes(pattern)
        if terminal_count >= min_terminals:
            yield pattern


def get_terminals(tree: Tree) -> list[str]:
    """Get all terminal (word) nodes from a tree."""
    if not isinstance(tree, Tree):
        return [str(tree)]
    terminals = []
    for child in tree:
        terminals.extend(get_terminals(child))
    return terminals


def get_terminals_at_depth(tree: Tree, depth: int) -> list[str]:
    """
    Get terminal words that would be covered by a subtree at given depth.

    For depth=2, gets the first terminal under each immediate child.
    For depth=3, gets terminals from children and grandchildren, etc.
    """
    if not isinstance(tree, Tree):
        return [str(tree)]

    if depth <= 1:
        # At depth 1, this node is a leaf in the pattern, so get ALL terminals
        # under it (the pattern collapses this entire subtree)
        return get_terminals(tree)

    terminals = []
    for child in tree:
        if isinstance(child, Tree):
            # Recurse with depth-1
            child_terms = get_terminals_at_depth(child, depth - 1)
            terminals.extend(child_terms)
        else:
            terminals.append(str(child))

    return terminals


def extract_patterns_with_examples(
    tree: Tree,
    sentence: str,
    max_depth: int = 4,
    min_depth: int = 2,
    min_terminals: int = 2,
) -> Iterator[tuple[str, str, str]]:
    """
    Extract patterns along with example words and the full sentence.

    Args:
        tree: NLTK Tree (full parse with terminals)
        sentence: The original sentence text
        max_depth: Maximum subtree depth
        min_depth: Minimum subtree depth
        min_terminals: Minimum terminal (leaf) nodes per pattern

    Yields:
        Tuples of (pattern, highlighted_words, sentence)
    """

    def walk_and_extract(t):
        if not isinstance(t, Tree):
            return

        for depth in range(min_depth, max_depth + 1):
            subtree = _extract_subtree_at_depth(t, depth)
            if subtree is not None:
                pattern = canonicalize(subtree)
                terminal_count = count_terminal_nodes(pattern)
                if terminal_count >= min_terminals:
                    # Get representative terminals for this depth
                    words = get_terminals_at_depth(t, depth)
                    highlighted = " ".join(words)
                    yield (pattern, highlighted, sentence)

        for child in t:
            if isinstance(child, Tree):
                yield from walk_and_extract(child)

    yield from walk_and_extract(tree)
