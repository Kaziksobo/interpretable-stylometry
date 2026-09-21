"""Shared compositional-data utilities for RQ1 Tracks 1 and 2.

Filtering, proportion computation, per-document zero-replacement, and the
CLR transform -- everything needed to turn doc_features.feather counts into
a valid composition ready for either the variance (Track 1) or regularity
(Track 2) pipeline.

See docs/rq1_methodology.md §4.1 (Eq. 1-5, 9) for the full specification.
Not a driver script -- imported by run_rq1_variance.py and
run_rq1_regularity.py, no __main__.
"""
