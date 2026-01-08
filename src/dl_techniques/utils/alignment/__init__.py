"""
Platonic Representation Alignment Module for dl_techniques.

This module implements various metrics for measuring alignment between
neural network representations, based on "The Platonic Representation
Hypothesis" (Huh et al., ICML 2024).

Main components:
    - AlignmentMetrics: Collection of alignment metrics
    - Alignment: High-level API for computing alignment
    - AlignmentLogger: Training callback for monitoring alignment

Example usage:
    ```python
    from dl_techniques.utils.alignment import Alignment, AlignmentMetrics
    
    # Initialize alignment scorer
    scorer = Alignment(metric="mutual_knn", topk=10)
    
    # Compute alignment between two representations
    score, indices = scorer.compute_pairwise_alignment(
        features_a, features_b
    )
    
    # Or use metrics directly
    score = AlignmentMetrics.mutual_knn(features_a, features_b, topk=10)
    ```

Available metrics:
    - mutual_knn: Mutual k-nearest neighbors
    - cycle_knn: Cycle k-nearest neighbors
    - lcs_knn: Longest common subsequence of k-NN
    - cka: Centered Kernel Alignment
    - unbiased_cka: Unbiased CKA
    - cknna: CKA with nearest neighbor attention
    - svcca: Singular Vector CCA
    - edit_distance_knn: Edit distance of k-NN orderings
"""

from .metrics import AlignmentMetrics, remove_outliers
from .alignment import Alignment, AlignmentLogger
from .utils import (
    prepare_features,
    compute_score,
    compute_alignment_matrix,
    normalize_features,
    extract_layer_features,
    save_features,
    load_features,
    pool_features,
    create_feature_filename,
    create_alignment_filename,
    compute_statistics,
    batch_generator
)

__version__ = "1.0.0"

__all__ = [
    # Main classes
    "AlignmentMetrics",
    "Alignment",
    "AlignmentLogger",
    
    # Utility functions
    "remove_outliers",
    "prepare_features",
    "compute_score",
    "compute_alignment_matrix",
    "normalize_features",
    "extract_layer_features",
    "save_features",
    "load_features",
    "pool_features",
    "create_feature_filename",
    "create_alignment_filename",
    "compute_statistics",
    "batch_generator",
]
