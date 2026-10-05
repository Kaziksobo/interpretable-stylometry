"""
Subtree pattern extraction and counting.

Extracts induced subtrees from constituency parse trees and counts their frequencies.
"""

import re
from collections import Counter
from typing import Iterator

from nltk import Tree

# a leaf label counts towards min_terminals only if it looks like a phrase or
# POS tag; punctuation tags such as "," "." ":" do not (this mirrors the regex
# in count_terminal_nodes)
_COUNTABLE_LABEL = re.compile(r"[A-Z$][A-Z0-9$-]*")


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


def get_pattern_anchor_words(tree: Tree, depth: int) -> list[str]:
    """
    Get the first word under each leaf node of the abstract pattern.

    Where get_terminals_at_depth returns ALL words under the pattern's
    coverage, this returns just the FIRST word under each structural
    position that appears as a leaf at this depth. This gives concise
    anchor words showing which word fills each slot in the pattern,
    rather than dumping every word under the entire subtree.

    Example: (S (NP (DT)) (VP (VBZ) (VP)) (.)) at depth=2
        -> ["This", "has", "."]   not the whole sentence
    """
    if not isinstance(tree, Tree):
        return [str(tree)]

    if depth <= 1:
        # This node is collapsed to a leaf in the pattern.
        # Return just the first terminal word under it.
        leaves = get_terminals(tree)
        return [leaves[0]] if leaves else []

    result = []
    for child in tree:
        if isinstance(child, Tree):
            result.extend(get_pattern_anchor_words(child, depth - 1))
        else:
            result.append(str(child))
    return result


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

        seen_at_this_node = set()
        for depth in range(min_depth, max_depth + 1):
            subtree = _extract_subtree_at_depth(t, depth)
            if subtree is not None:
                pattern = canonicalize(subtree)
                if pattern in seen_at_this_node:
                    continue
                seen_at_this_node.add(pattern)
                terminal_count = count_terminal_nodes(pattern)
                if terminal_count >= min_terminals:
                    # Get representative terminals for this depth
                    words = get_pattern_anchor_words(t, depth)
                    highlighted = " ".join(words)
                    yield (pattern, highlighted, sentence)

        for child in t:
            if isinstance(child, Tree):
                yield from walk_and_extract(child)

    yield from walk_and_extract(tree)


def count_patterns(
    tree: Tree, max_depth: int = 4, min_depth: int = 2, min_terminals: int = 2
) -> Counter:
    """
    Count the pattern occurrences in one parse tree.

    Finds the same patterns, with the same per-node de-duplication, as
    extract_patterns_with_examples, but builds each node's depth-d pattern
    string bottom-up from its children's depth-(d-1) strings instead of
    constructing Tree objects, and does no example or anchor-word work. A
    pattern is counted once per node it occurs at, so a pattern that occurs
    at two nodes of the tree counts twice.

    Args:
        tree: NLTK Tree (full constituency parse)
        max_depth: Maximum subtree depth
        min_depth: Minimum subtree depth (default 2 excludes single tags)
        min_terminals: Minimum number of countable leaf nodes in the pattern

    Returns:
        Counter mapping canonical pattern strings to occurrence counts
    """
    counts: Counter = Counter()
    _count_node(tree, max_depth, min_depth, min_terminals, counts)
    return counts


def _count_node(
    node: Tree,
    max_depth: int,
    min_depth: int,
    min_terminals: int,
    counts: Counter,
) -> list[tuple[str, int]]:
    """
    Count the patterns rooted at node and below, returning node's own forms.

    Args:
        node: Subtree to process
        max_depth: Maximum subtree depth
        min_depth: Minimum subtree depth
        min_terminals: Minimum number of countable leaf nodes in the pattern
        counts: Counter updated in place with every pattern found

    Returns:
        One (pattern string, countable leaf count) pair per depth 1..max_depth
        for this node, so the parent can build its own deeper patterns.
    """
    label = node.label()
    leaf_form = (f"({label})", int(_COUNTABLE_LABEL.fullmatch(label) is not None))

    child_forms = [
        _count_node(child, max_depth, min_depth, min_terminals, counts)
        for child in node
        if isinstance(child, Tree)
    ]

    if not child_forms:
        forms = [leaf_form] * max_depth
    else:
        forms = [leaf_form]
        for depth in range(2, max_depth + 1):
            # a child contributes its own depth-1 form at this node's depth
            kids = [cf[depth - 2] for cf in child_forms]
            pattern = f"({label} {' '.join(k[0] for k in kids)})"
            forms.append((pattern, sum(k[1] for k in kids)))

    seen_at_this_node = set()
    for depth in range(min_depth, max_depth + 1):
        pattern, n_terminals = forms[depth - 1]
        if pattern in seen_at_this_node:
            continue
        seen_at_this_node.add(pattern)
        if n_terminals >= min_terminals:
            counts[pattern] += 1

    return forms


def pattern_depth(pattern: str) -> int:
    """
    Return the depth of a canonical pattern string.

    Depth counts levels including the root, i.e. the deepest parenthesis
    nesting: "(NP (DT) (NN))" has depth 2 and "(S (NP (DT)) (VP))" depth 3.

    Args:
        pattern: Canonical pattern string

    Returns:
        Depth of the pattern
    """
    level = deepest = 0
    for char in pattern:
        if char == "(":
            level += 1
            deepest = max(deepest, level)
        elif char == ")":
            level -= 1
    return deepest
