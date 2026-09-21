"""RQ1 Track 2 -- Regularity: within-document entropy over syntactic categories.

Tests whether a single AI-generated document spreads across recognised
sentence types/structures more or less evenly than a human document does.

See docs/rq1_methodology.md §5 for the full pipeline (Eq. 11-13) and
§9.2, §9.4 for exact output schemas.

Inputs:
    data/processed/rq1/doc_features.feather

Outputs:
    data/processed/rq1/doc_entropy.feather
    results/rq1/rq1_regularity_significance.csv   -- RQ1 answer file
"""
