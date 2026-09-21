"""Generic test-family runner shared across RQ1 Tracks 1, 2, and 3.

run_test_family(): takes a {(domain, feature/composition, comparison):
(human_values, ai_values)} dict and a scipy test function, returns raw
stat + p-value per cell.

apply_fdr(): Benjamini-Hochberg wrapper, adds _fdr and _significant columns.

See docs/rq1_methodology.md §7 for the FDR family structure this feeds
(four separate families -- don't pool across tracks).
Not a driver script -- imported, no __main__.
"""
