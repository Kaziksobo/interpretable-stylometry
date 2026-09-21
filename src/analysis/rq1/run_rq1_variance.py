"""RQ1 Track 1 -- Variance: multivariate compositional dispersion.

Tests whether AI-generated prose shows reduced between-document variance
in syntactic composition, via CLR transform + distance-to-centroid, rather
than testing each category rate separately (see Track 3 for that).

See docs/rq1_methodology.md §4 for the full pipeline (Eq. 1-10) and
§9.1, §9.3, §9.6 for exact output schemas.

Inputs:
    data/processed/rq1/doc_features.feather

Outputs:
    data/processed/rq1/doc_distances.feather
    results/rq1/rq1_variance_significance.csv   -- RQ1 answer file
    results/rq1/centroids.csv
"""
