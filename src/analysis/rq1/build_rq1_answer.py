"""Synthesises the three RQ1 significance tables into one plain-English answer.

Reads rq1_variance_significance.csv, rq1_regularity_significance.csv, and
rq1_diagnostics_significance.csv only -- no doc_features.feather dependency,
no compositional maths, pure pandas + string formatting.

See docs/rq1_methodology.md §9.7 for the exact generation logic (the
significant x ratio-direction verdict table, and the diagnostic-row
selection rule).

Inputs:
    results/rq1/rq1_variance_significance.csv
    results/rq1/rq1_regularity_significance.csv
    results/rq1/rq1_diagnostics_significance.csv

Outputs:
    results/rq1/rq1_answer.txt
"""
