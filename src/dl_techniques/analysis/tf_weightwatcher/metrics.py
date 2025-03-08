"""
Metrics module for TensorFlow WeightWatcher

This module contains autonomous functions for calculating various metrics on neural network
weight matrices, used by the TensorFlow WeightWatcher for analysis.
"""

import numpy as np
from scipy import stats
from scipy.sparse.linalg import svds
from typing import Dict, List, Optional, Tuple, Union, Any

# Import constants
from .constants import (
    EPSILON, EVALS_THRESH, OVER_TRAINED_THRESH, UNDER_TRAINED_THRESH,
    DEFAULT_MIN_EVALS, DEFAULT_MAX_EVALS, DEFAULT_BINS
)

from dl_techniques.utils.logger import logger

def compute_eigenvalues(
        weight_matrices: List[np.ndarray],
        N: int,
        M: int,
        n_comp: int,
        normalize: bool = False
) -> Tuple[np.ndarray, float, float, int]:
    """
    Compute the eigenvalues for all weight matrices combined.

    Args:
        weight_matrices: List of weight matrices.
        N: Maximum dimension of matrices.
        M: Minimum dimension of matrices.
        n_comp: Number of components to consider.
        normalize: Whether to normalize eigenvalues.

    Returns:
        Tuple containing:
        - Array of sorted eigenvalues.
        - Maximum singular value.
        - Minimum singular value.
        - Rank loss.
    """
    all_evals = []
    max_sv = 0.0
    min_sv = float('inf')
    rank_loss = 0

    for W in weight_matrices:
        W = W.astype(float)

        # For large matrices, use truncated SVD
        if (n_comp < M) or (M > DEFAULT_MAX_EVALS):
            try:
                # Use scipy's sparse SVD for large matrices
                _, sv, _ = svds(W, k=min(n_comp, M - 1))
                sv = np.sort(sv)
            except Exception as e:
                logger.warning(f"SVD failed: {e}, falling back to full SVD")
                sv = np.linalg.svd(W, compute_uv=False)
                sv = sv[-n_comp:] if len(sv) > n_comp else sv
        else:
            # Use full SVD for smaller matrices
            sv = np.linalg.svd(W, compute_uv=False)
            sv = sv[-n_comp:] if len(sv) > n_comp else sv

        # Calculate eigenvalues from singular values
        evals = sv * sv

        if normalize:
            evals = evals / N

        all_evals.extend(evals)
        max_sv = max(max_sv, np.max(sv))
        if len(sv) > 0:
            min_sv = min(min_sv, np.min(sv))

        # Calculate rank loss
        tol = max_sv * np.finfo(np.float32).eps
        rank = np.sum(sv > tol)
        rank_loss += len(sv) - rank

    # Handle the case where min_sv was never updated
    if min_sv == float('inf'):
        min_sv = 0.0

    return np.sort(np.array(all_evals)), max_sv, min_sv, rank_loss


def calculate_matrix_entropy(singular_values: np.ndarray, N: int) -> float:
    """
    Calculate the matrix entropy from singular values.

    Args:
        singular_values: Array of singular values.
        N: Maximum dimension of the matrix.

    Returns:
        Matrix entropy (float).
    """
    try:
        # Calculate matrix rank
        tol = np.max(singular_values) * N * np.finfo(singular_values.dtype).eps
        rank = np.sum(singular_values > tol)

        # Calculate eigenvalues from singular values
        evals = singular_values * singular_values

        # Calculate probabilities and entropy
        p = evals / np.sum(evals) + EPSILON
        rank = max(rank, 1) + EPSILON
        entropy = -np.sum(p * np.log(p)) / np.log(rank)

        return entropy
    except Exception as e:
        logger.warning(f"Error calculating matrix entropy: {e}")
        return -1.0


def fit_powerlaw(
        evals: np.ndarray,
        xmin: Optional[float] = None
) -> Tuple[float, float, float, float, int, str, str]:
    """
    Fit eigenvalues to a power-law distribution.

    Args:
        evals: Array of eigenvalues.
        xmin: Minimum x value for fitting.

    Returns:
        Tuple containing:
        - alpha: Power-law exponent.
        - xmin: Minimum x value used.
        - D: Kolmogorov-Smirnov statistic.
        - sigma: Standard error of alpha.
        - num_pl_spikes: Number of eigenvalues in power-law tail.
        - status: Success or failure status.
        - warning: Warning message (if any).
    """
    # Initialize return values
    alpha = -1
    D = -1
    sigma = -1
    if xmin is None or xmin <= 0:
        xmin = np.min(evals[evals > EVALS_THRESH])
    num_pl_spikes = -1
    status = "failed"
    warning = ""

    # Ensure we have enough eigenvalues
    if len(evals) < DEFAULT_MIN_EVALS:
        logger.warning("Not enough eigenvalues for power-law fitting")
        return alpha, xmin, D, sigma, num_pl_spikes, status, warning

    try:
        # Filter out very small eigenvalues
        nz_evals = evals[evals > EVALS_THRESH]

        # Simple power-law estimation (using MLE for discrete power-law)
        # For continuous power-law, alpha = 1 + n / (∑log(xi/xmin))
        filtered_evals = nz_evals[nz_evals >= xmin]
        n = len(filtered_evals)

        if n > 0:
            alpha = 1 + n / np.sum(np.log(filtered_evals / xmin))
            sigma = (alpha - 1) / np.sqrt(n)

            # Calculate Kolmogorov-Smirnov statistic
            # Compare empirical CDF with theoretical power-law CDF
            empirical_cdf = np.arange(1, n + 1) / n
            theoretical_cdf = 1 - (filtered_evals / xmin) ** (1 - alpha)
            D = np.max(np.abs(empirical_cdf - theoretical_cdf))

            num_pl_spikes = n
            status = "success"

            # Add warning based on alpha value
            if alpha < OVER_TRAINED_THRESH:
                warning = "over-trained"
            elif alpha > UNDER_TRAINED_THRESH:
                warning = "under-trained"

    except Exception as e:
        logger.warning(f"Power-law fitting failed: {e}")

    return alpha, xmin, D, sigma, num_pl_spikes, status, warning


def calculate_stable_rank(evals: np.ndarray) -> float:
    """
    Calculate the stable rank of a matrix from its eigenvalues.

    Args:
        evals: Array of eigenvalues.

    Returns:
        Stable rank (float).
    """
    if len(evals) == 0:
        return 0.0

    # Stable rank is the sum of eigenvalues divided by the largest eigenvalue
    return np.sum(evals) / np.max(evals)


def calculate_spectral_metrics(evals: np.ndarray, alpha: float) -> Dict[str, float]:
    """
    Calculate various spectral metrics from eigenvalues.

    Args:
        evals: Array of eigenvalues.
        alpha: Power-law exponent.

    Returns:
        Dictionary of spectral metrics.
    """
    if len(evals) == 0:
        return {
            "norm": 0.0,
            "log_norm": 0.0,
            "spectral_norm": 0.0,
            "log_spectral_norm": 0.0,
            "alpha_weighted": 0.0,
            "log_alpha_norm": 0.0,
            "stable_rank": 0.0
        }

    norm = np.sum(evals)
    spectral_norm = np.max(evals)

    return {
        "norm": norm,
        "log_norm": np.log10(norm),
        "spectral_norm": spectral_norm,
        "log_spectral_norm": np.log10(spectral_norm),
        "alpha_weighted": alpha * np.log10(spectral_norm),
        "log_alpha_norm": np.log10(np.sum([ev ** alpha for ev in evals])),
        "stable_rank": norm / spectral_norm
    }


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Jensen-Shannon distance between two distributions.

    Args:
        p: First distribution.
        q: Second distribution.

    Returns:
        Jensen-Shannon distance.
    """
    # Create histograms with the same bins
    min_val = min(np.min(p), np.min(q))
    max_val = max(np.max(p), np.max(q))

    p_hist, _ = np.histogram(p, bins=DEFAULT_BINS, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q, bins=DEFAULT_BINS, range=(min_val, max_val), density=True)

    # Add small epsilon to avoid division by zero
    p_hist = p_hist + EPSILON
    q_hist = q_hist + EPSILON

    # Normalize
    p_hist = p_hist / np.sum(p_hist)
    q_hist = q_hist / np.sum(q_hist)

    # Calculate m = (p+q)/2
    m = (p_hist + q_hist) / 2

    # Calculate Jensen-Shannon Divergence
    divergence = (stats.entropy(p_hist, m) + stats.entropy(q_hist, m)) / 2

    # Calculate Jensen-Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def compute_detX_constraint(evals: np.ndarray) -> int:
    """
    Identify the number of eigenvalues necessary to satisfy det(X) = 1.

    Args:
        evals: Array of eigenvalues (sorted ascending).

    Returns:
        Number of eigenvalues in the tail.
    """
    # Sort eigenvalues in ascending order
    sorted_evals = np.sort(evals)
    num_evals = len(sorted_evals)

    # Find first index where product of eigenvalues >= 1
    prod = 1.0
    for idx in range(num_evals - 1, -1, -1):
        prod *= sorted_evals[idx]
        if prod >= 1.0:
            return num_evals - idx

    return num_evals  # If no constraint is satisfied, keep all eigenvalues


def smooth_matrix(W: np.ndarray, n_comp: int) -> np.ndarray:
    """
    Apply SVD smoothing to a weight matrix.

    Args:
        W: Weight matrix.
        n_comp: Number of components to keep.

    Returns:
        Smoothed weight matrix.
    """
    # Ensure matrix has correct dimensions (larger dimension first)
    transpose = False
    if W.shape[0] < W.shape[1]:
        W = W.T
        transpose = True

    # Compute SVD
    try:
        u, s, vh = np.linalg.svd(W, full_matrices=False)

        # Set smaller singular values to zero
        if n_comp < len(s):
            s[n_comp:] = 0

        # Reconstruct the matrix
        smoothed_W = np.dot(u * s, vh)

        # Restore original orientation if needed
        if transpose:
            smoothed_W = smoothed_W.T

        return smoothed_W

    except Exception as e:
        logger.warning(f"SVD failed: {e}, returning original matrix")
        return W


def calculate_glorot_normalization_factor(N: int, M: int, rf: int) -> float:
    """
    Calculate the Glorot normalization factor.

    Args:
        N: Maximum dimension of matrix.
        M: Minimum dimension of matrix.
        rf: Receptive field size.

    Returns:
        Normalization factor.
    """
    return np.sqrt(2 / ((N + M) * rf))