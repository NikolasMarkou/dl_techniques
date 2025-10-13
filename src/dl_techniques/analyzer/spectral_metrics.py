"""
Enhanced Metrics module for WeightWatcher integration

This module contains autonomous functions for calculating various spectral and statistical metrics
on neural network weight matrices, including advanced concentration and localization measures.
"""

import numpy as np
from scipy import stats
from scipy.sparse.linalg import svds
from typing import Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.analyzer.constants import (
    SPECTRAL_EPSILON, SPECTRAL_EVALS_THRESH, SPECTRAL_OVER_TRAINED_THRESH, SPECTRAL_UNDER_TRAINED_THRESH,
    SPECTRAL_DEFAULT_MIN_EVALS, SPECTRAL_DEFAULT_MAX_EVALS, SPECTRAL_DEFAULT_BINS,
    SPECTRAL_MAX_CRITICAL_WEIGHTS_REPORTED, SPECTRAL_CRITICAL_WEIGHT_THRESHOLD
)

# ---------------------------------------------------------------------

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
        weight_matrices: List of weight matrices to analyze.
        N: Maximum dimension of matrices.
        M: Minimum dimension of matrices.
        n_comp: Number of components to consider.
        normalize: Whether to normalize eigenvalues by N.

    Returns:
        Tuple containing:
            - Array of sorted eigenvalues (ascending order).
            - Maximum singular value across all matrices.
            - Minimum singular value across all matrices.
            - Total rank loss across all matrices.
    """
    all_evals = []
    max_sv = 0.0
    min_sv = float('inf')
    rank_loss = 0

    for W in weight_matrices:
        W = W.astype(float)

        # For large matrices, use truncated SVD
        if (n_comp < M) or (M > SPECTRAL_DEFAULT_MAX_EVALS):
            try:
                # Use scipy's sparse SVD for large matrices
                _, sv, _ = svds(W, k=min(n_comp, M - 1))
                sv = np.sort(sv)[::-1]  # Sort descending
            except Exception as e:
                logger.warning(f"SVD failed: {e}, falling back to full SVD")
                sv = np.linalg.svd(W, compute_uv=False)
                if len(sv) > n_comp:
                    sv = sv[:n_comp]
        else:
            # Use full SVD for smaller matrices
            sv = np.linalg.svd(W, compute_uv=False)
            if len(sv) > n_comp:
                sv = sv[:n_comp]

        # Calculate eigenvalues from singular values
        evals = sv * sv

        if normalize:
            evals = evals / N

        all_evals.extend(evals)

        # Update global min/max singular values
        current_max_sv = np.max(sv) if len(sv) > 0 else 0.0
        max_sv = max(max_sv, current_max_sv)
        if len(sv) > 0:
            min_sv = min(min_sv, np.min(sv))

        # Calculate rank loss using current matrix's max singular value
        tol = current_max_sv * np.finfo(W.dtype).eps
        rank = np.sum(sv > tol)
        rank_loss += len(sv) - rank

    # Handle the case where min_sv was never updated
    if min_sv == float('inf'):
        min_sv = 0.0

    return np.sort(np.array(all_evals)), max_sv, min_sv, rank_loss

# ---------------------------------------------------------------------

def fit_powerlaw(
        evals: np.ndarray,
        xmin: Optional[float] = None
) -> Tuple[float, float, float, float, int, str, str]:
    """
    Fit eigenvalues to a power-law distribution using maximum likelihood estimation.

    Args:
        evals: Array of eigenvalues to fit.
        xmin: Minimum x value for fitting. If None, automatically determined.

    Returns:
        Tuple containing:
            - alpha: Power-law exponent.
            - xmin: Minimum x value used for fitting.
            - D: Kolmogorov-Smirnov statistic.
            - sigma: Standard error of alpha.
            - num_pl_spikes: Number of eigenvalues in power-law tail.
            - status: Success or failure status string.
            - warning: Warning message string (if any).
    """
    # Initialize return values
    alpha = -1
    D = -1
    sigma = -1
    if xmin is None or xmin <= 0:
        xmin = np.min(evals[evals > SPECTRAL_EVALS_THRESH])
    num_pl_spikes = -1
    status = "failed"
    warning = ""

    # Ensure we have enough eigenvalues
    if len(evals) < SPECTRAL_DEFAULT_MIN_EVALS:
        logger.warning("Not enough eigenvalues for power-law fitting")
        return alpha, xmin, D, sigma, num_pl_spikes, status, warning

    try:
        # Filter out very small eigenvalues
        nz_evals = evals[evals > SPECTRAL_EVALS_THRESH]

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
            if alpha < SPECTRAL_OVER_TRAINED_THRESH:
                warning = "over-trained"
            elif alpha > SPECTRAL_UNDER_TRAINED_THRESH:
                warning = "under-trained"

    except Exception as e:
        logger.warning(f"Power-law fitting failed: {e}")

    return alpha, xmin, D, sigma, num_pl_spikes, status, warning

# ---------------------------------------------------------------------

def calculate_matrix_entropy(singular_values: np.ndarray, N: int) -> float:
    """
    Calculate the matrix entropy from singular values.

    The matrix entropy is computed as H = -sum(p_i * log(p_i)) / log(rank),
    where p_i are the normalized eigenvalues and rank is the effective rank.

    Args:
        singular_values: Array of singular values from SVD.
        N: Maximum dimension of the matrix.

    Returns:
        Matrix entropy (float). Returns -1.0 if calculation fails.
    """
    try:
        # Calculate matrix rank
        tol = np.max(singular_values) * N * np.finfo(singular_values.dtype).eps
        rank = np.sum(singular_values > tol)

        # Calculate eigenvalues from singular values
        evals = singular_values * singular_values

        # Calculate probabilities
        p = evals / (np.sum(evals) + SPECTRAL_EPSILON)

        # Remove practically zero probabilities
        p = p[p > 0]

        # Ensure rank is at least 1
        rank = max(rank, 1)

        # Calculate entropy with proper log(rank) handling
        log_rank = np.log(rank) if rank > 1 else np.log(2)  # Avoid log(1)=0 division
        entropy = -np.sum(p * np.log(p)) / log_rank

        return entropy
    except Exception as e:
        logger.warning(f"Error calculating matrix entropy: {e}")
        return -1.0

# ---------------------------------------------------------------------

def calculate_spectral_metrics(evals: np.ndarray, alpha: float) -> Dict[str, float]:
    """
    Calculate various spectral metrics from eigenvalues.

    Args:
        evals: Array of eigenvalues.
        alpha: Power-law exponent for alpha-weighted metrics.

    Returns:
        Dictionary containing spectral metrics.
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
        "log_alpha_norm": np.log10(np.sum(evals ** alpha)),
        "stable_rank": norm / spectral_norm
    }

# ---------------------------------------------------------------------

def calculate_gini_coefficient(evals: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of the eigenvalue distribution.

    The Gini coefficient measures inequality in distribution:
    - 0 indicates perfect equality (all eigenvalues are equal)
    - 1 indicates perfect inequality (one eigenvalue dominates)

    Args:
        evals: Array of eigenvalues.

    Returns:
        Gini coefficient between 0 and 1.
    """
    if len(evals) < 2:
        return 0.0

    # Ensure all values are non-negative and sort
    sorted_evals = np.sort(np.abs(evals))
    n = len(sorted_evals)

    # Calculate Lorenz curve
    cum_evals = np.cumsum(sorted_evals)

    # Calculate Gini coefficient
    denominator = n * sorted_evals.sum()
    if denominator < SPECTRAL_EPSILON:
        return 0.0

    return ((n + 1) / n) - (2 * np.sum(cum_evals)) / denominator

# ---------------------------------------------------------------------

def calculate_dominance_ratio(evals: np.ndarray) -> float:
    """
    Calculate the ratio of the largest eigenvalue to the sum of all other eigenvalues.

    This quantifies how much a single dimension dominates the spectrum.
    High values indicate potential concentration of information.

    Args:
        evals: Array of eigenvalues.

    Returns:
        Dominance ratio (λ_max / sum(λ_others)).
    """
    if len(evals) < 2:
        return float('inf')

    lambda_max = np.max(evals)
    sum_others = np.sum(evals) - lambda_max

    if sum_others < SPECTRAL_EPSILON:
        return float('inf')

    return lambda_max / sum_others

# ---------------------------------------------------------------------

def calculate_participation_ratio(vector: np.ndarray) -> float:
    """
    Calculate the participation ratio of a vector, a measure of localization.

    The participation ratio is defined as:
    PR = (sum(v_i^2))^2 / sum(v_i^4)

    A low participation ratio indicates the vector's energy is concentrated in
    a few elements (localized).

    Args:
        vector: Eigenvector or other vector to analyze.

    Returns:
        Participation ratio. Lower values indicate more localization.
    """
    # Normalize the vector
    vec_norm = np.linalg.norm(vector)
    if vec_norm < SPECTRAL_EPSILON:
        return float('inf')

    vec = vector / vec_norm

    # Calculate participation ratio
    vec_sq = vec ** 2
    numerator = np.sum(vec_sq) ** 2
    denominator = np.sum(vec_sq ** 2)

    if denominator < SPECTRAL_EPSILON:
        return float('inf')

    return numerator / denominator

# ---------------------------------------------------------------------

def get_top_eigenvectors(
        weight_matrix: np.ndarray,
        k: int = 1,
        method: str = 'direct'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the top k left singular vectors of a weight matrix.

    Args:
        weight_matrix: Weight matrix to analyze.
        k: Number of top eigenvectors to return.
        method: Method to use ('direct' or 'power_iteration').

    Returns:
        Tuple containing:
            - Array of top k eigenvalues of W @ W.T (squared singular values of W).
            - Array of top k eigenvectors of W @ W.T (left singular vectors of W).
    """
    n, m = weight_matrix.shape
    min_dim = min(n, m)

    k = min(k, min_dim - 1)
    if k <= 0:
        return np.array([]), np.array([])

    try:
        if method == 'direct':
            # Use SVD for general matrices
            u, s, vh = np.linalg.svd(weight_matrix, full_matrices=False)
            # Return squared singular values and left singular vectors
            return s[:k] ** 2, u[:, :k]
        else:
            # Use power iteration for very large matrices
            return _power_iteration(weight_matrix @ weight_matrix.T, k)
    except Exception as e:
        logger.error(f"Eigenvector computation failed: {e}")
        return np.array([]), np.array([])

# ---------------------------------------------------------------------

def _power_iteration(
        matrix: np.ndarray,
        k: int = 1,
        max_iter: int = 100,
        tol: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the top k eigenvalues and eigenvectors using power iteration.

    Args:
        matrix: Square matrix to analyze.
        k: Number of eigenvectors to compute.
        max_iter: Maximum number of iterations per eigenvector.
        tol: Convergence tolerance.

    Returns:
        Tuple containing arrays of eigenvalues and eigenvectors.
    """
    n = matrix.shape[0]
    eigvals = np.zeros(k)
    eigvecs = np.zeros((n, k))

    # Start with random vectors
    Q = np.random.randn(n, k)
    Q, _ = np.linalg.qr(Q)

    for i in range(k):
        q = Q[:, i].reshape(-1, 1)

        # Deflate previously computed eigenvectors
        for j in range(i):
            q = q - eigvecs[:, j].reshape(-1, 1) @ (eigvecs[:, j].reshape(1, -1) @ q)

        # Power iteration
        for _ in range(max_iter):
            z = matrix @ q
            lambda_i = np.linalg.norm(z)
            q_new = z / (lambda_i + SPECTRAL_EPSILON)

            # Check convergence
            if np.linalg.norm(q_new - q) < tol:
                break

            q = q_new

        # Store results
        eigvals[i] = lambda_i
        eigvecs[:, i] = q.flatten()

    return eigvals, eigvecs

# ---------------------------------------------------------------------

def find_critical_weights(
        weight_matrix: np.ndarray,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        threshold: float = SPECTRAL_CRITICAL_WEIGHT_THRESHOLD
) -> List[Tuple[int, int, float]]:
    """
    Find individual weights that contribute most to top eigenvectors.

    This heuristic identifies weights with large magnitudes that are located in
    the most important rows of the weight matrix. Row importance is determined
    by the components of the top eigenvectors of the matrix WW^T.

    Args:
        weight_matrix: Weight matrix to analyze (shape n x m).
        eigenvectors: Top eigenvectors of WW^T (left singular vectors of W), shape (n, k).
        eigenvalues: Corresponding eigenvalues.
        threshold: Contribution threshold for identifying critical weights.

    Returns:
        List of tuples (row_idx, col_idx, contribution) for critical weights.
    """
    if eigenvectors.size == 0 or eigenvalues.size == 0:
        return []

    n, m = weight_matrix.shape
    critical_weights = set()  # Use a set to avoid duplicate (i, j) entries

    # Calculate an overall importance score for each row by taking a weighted
    # average of the absolute eigenvector components.
    row_importance = np.zeros(n)
    total_eigenvalue_sum = np.sum(eigenvalues)
    if total_eigenvalue_sum > SPECTRAL_EPSILON:
        for eigval, eigvec in zip(eigenvalues, eigenvectors.T):
            # eigvec has shape (n,)
            row_importance += (eigval / total_eigenvalue_sum) * np.abs(eigvec)

    # Find rows with high importance scores.
    max_importance = np.max(row_importance)
    if max_importance < SPECTRAL_EPSILON:
        return []

    high_importance_row_indices = np.where(row_importance > threshold * max_importance)[0]

    # For each important row, find the weights with the largest magnitudes.
    for i in high_importance_row_indices:
        row_weights = weight_matrix[i, :]
        max_weight_in_row = np.max(np.abs(row_weights))

        if max_weight_in_row < SPECTRAL_EPSILON:
            continue

        # Find columns (weights) in this row with magnitudes above a threshold.
        high_magnitude_col_indices = np.where(np.abs(row_weights) > threshold * max_weight_in_row)[0]

        for j in high_magnitude_col_indices:
            # Define contribution as the product of the row's importance and the weight's value.
            contribution = row_importance[i] * weight_matrix[i, j]
            critical_weights.add((i, j, float(contribution)))

    # Sort by contribution magnitude (descending)
    sorted_critical_weights = sorted(list(critical_weights), key=lambda x: abs(x[2]), reverse=True)

    return sorted_critical_weights

# ---------------------------------------------------------------------

def calculate_concentration_metrics(
        weight_matrix: np.ndarray,
        num_eigenvectors: int = 3
) -> Dict[str, Union[float, List[Tuple[int, int, float]]]]:
    """
    Calculate comprehensive concentration metrics for a weight matrix.

    This function combines multiple measures to assess how concentrated
    the weight matrix information is in specific patterns or weights.

    Args:
        weight_matrix: Weight matrix to analyze.
        num_eigenvectors: Number of top eigenvectors to analyze.

    Returns:
        Dictionary containing concentration metrics.
    """
    if min(weight_matrix.shape) < SPECTRAL_DEFAULT_MIN_EVALS:
        return {}

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = get_top_eigenvectors(weight_matrix, k=num_eigenvectors)

    if len(eigenvalues) == 0:
        return {}

    # Calculate various concentration metrics
    gini = calculate_gini_coefficient(eigenvalues)
    dominance = calculate_dominance_ratio(eigenvalues)

    # Calculate participation ratios for top eigenvectors
    participation_ratios = []
    for i in range(min(num_eigenvectors, eigenvectors.shape[1])):
        pr = calculate_participation_ratio(eigenvectors[:, i])
        participation_ratios.append(pr)

    # Find critical weights
    critical_weights = find_critical_weights(weight_matrix, eigenvectors, eigenvalues)

    # Calculate concentration score
    # Higher score indicates more concentration
    concentration_score = gini * dominance / (np.mean(participation_ratios) + SPECTRAL_EPSILON)

    return {
        'gini_coefficient': gini,
        'dominance_ratio': dominance,
        'participation_ratio': np.mean(participation_ratios),
        'min_participation_ratio': np.min(participation_ratios) if participation_ratios else 0,
        'critical_weights': critical_weights[:10],  # Top 10 critical weights
        'critical_weight_count': len(critical_weights),
        'critical_weights': critical_weights[:SPECTRAL_MAX_CRITICAL_WEIGHTS_REPORTED],
        'concentration_score': np.log1p(concentration_score)  # Log to manage extreme values
    }

# ---------------------------------------------------------------------

def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Jensen-Shannon distance between two distributions.

    Args:
        p: First distribution (array of values).
        q: Second distribution (array of values).

    Returns:
        Jensen-Shannon distance (float). Range: [0, 1].
    """
    # Create histograms with the same bins
    min_val = min(np.min(p), np.min(q))
    max_val = max(np.max(p), np.max(q))

    p_hist, _ = np.histogram(p, bins=SPECTRAL_DEFAULT_BINS, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q, bins=SPECTRAL_DEFAULT_BINS, range=(min_val, max_val), density=True)

    # Normalize histograms to probability distributions
    p_hist = p_hist / (np.sum(p_hist) + SPECTRAL_EPSILON)
    q_hist = q_hist / (np.sum(q_hist) + SPECTRAL_EPSILON)

    # Calculate m = (p+q)/2
    m = (p_hist + q_hist) / 2

    # Calculate Jensen-Shannon Divergence
    divergence = (stats.entropy(p_hist, m) + stats.entropy(q_hist, m)) / 2

    # Calculate Jensen-Shannon Distance
    distance = np.sqrt(divergence)

    return distance

# ---------------------------------------------------------------------

def compute_detX_constraint(evals: np.ndarray) -> int:
    """
    Identify the number of eigenvalues necessary to satisfy det(X) = 1.

    Args:
        evals: Array of eigenvalues (assumed to be sorted in descending order).

    Returns:
        Number of eigenvalues in the tail needed to satisfy the constraint.
    """
    # Sort eigenvalues in descending order
    sorted_evals = np.sort(evals)[::-1]
    num_evals = len(sorted_evals)

    # Find first index where product of eigenvalues >= 1
    prod = 1.0
    for idx in range(num_evals):
        prod *= sorted_evals[idx]
        if prod >= 1.0:
            return idx + 1

    return num_evals  # If no constraint is satisfied, keep all eigenvalues

# ---------------------------------------------------------------------

def smooth_matrix(W: np.ndarray, n_comp: int) -> np.ndarray:
    """
    Apply SVD smoothing to a weight matrix by zeroing out small singular values.

    Args:
        W: Weight matrix to smooth.
        n_comp: Number of largest singular values to keep.

    Returns:
        Smoothed weight matrix with the same shape as input.
    """
    # Ensure matrix has correct dimensions (larger dimension first)
    transpose = False
    if W.shape[0] < W.shape[1]:
        W = W.T
        transpose = True

    try:
        # Compute SVD
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

# ---------------------------------------------------------------------

def calculate_glorot_normalization_factor(N: int, M: int, rf: int) -> float:
    """
    Calculate the Glorot normalization factor for weight initialization.

    Args:
        N: Maximum dimension of matrix (fan_out).
        M: Minimum dimension of matrix (fan_in).
        rf: Receptive field size.

    Returns:
        Glorot normalization factor (float).
    """
    return np.sqrt(2 / ((N + M) * rf))

# ---------------------------------------------------------------------