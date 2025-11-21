"""
The mathematical core of spectral analysis for neural networks.

This module provides a suite of functions for analyzing the spectral
properties of deep learning model weights, based on principles from Random
Matrix Theory and statistical physics. The central idea is that the
distribution of eigenvalues of a layer's weight matrix contains a wealth of
information about the quality of training and the model's potential for
generalization.

Architecture and Methodology
---------------------------
The analysis pipeline implemented here follows these primary steps:
1.  **Eigenvalue Computation**: For a given weight matrix `W`, the eigenvalues
    {λ_i} of its correlation matrix `WW^T` are computed. This is achieved
    efficiently via Singular Value Decomposition (SVD), leveraging the
    relationship λ_i = σ_i^2, where {σ_i} are the singular values of `W`.
    For large matrices, truncated SVD (`svds`) is used for performance.
2.  **Power-Law Fitting**: The core hypothesis is that the tail of the
    Empirical Spectral Density (ESD) of these eigenvalues follows a truncated
    power-law distribution, P(λ) ~ λ^(-α). The function `fit_powerlaw`
    estimates the exponent `α` using a robust Maximum Likelihood Estimation
    (MLE) technique. This exponent `α` serves as the primary metric for
    assessing training quality.
3.  **Statistical Characterization**: Beyond the power-law fit, a variety
    of metrics are computed to characterize the shape and properties of the
    entire spectrum, including:
    -   **Information Content**: Stable Rank and Matrix Entropy measure the
        effective dimensionality and the uniformity of information spread
        across eigenvalues.
    -   **Concentration**: The Gini Coefficient and Dominance Ratio quantify
        the inequality in the spectrum, indicating whether information is
        concentrated in a few dominant modes.
    -   **Localization**: The Participation Ratio of the top eigenvectors
        is used to determine if the principal components of the learned
        features are localized to specific neurons or distributed.

Foundational Mathematics
------------------------
-   **Power-Law Exponent (α)**: This is the key metric. The MLE for a
    continuous power-law distribution is given by:
    α = 1 + n * [ Σ_{i=1 to n} log(x_i / x_min) ]^(-1)
    Empirical studies have shown a strong correlation between the value of `α`
    and a model's generalization gap. An `α` in the range (2.0, 6.0) is
    often indicative of a well-trained model, with values below 2.0
    suggesting overfitting and values above 6.0 suggesting under-training.

-   **Stable Rank**: Defined as ||W||_F^2 / ||W||_2^2, which simplifies to
    (Σ λ_i) / max(λ_i). It provides a more robust measure of the "effective"
    rank of a matrix than the discrete matrix rank, indicating the
    dimensionality of the space spanned by the weights.

-   **Participation Ratio (PR)**: Derived from Anderson localization theory in
    physics, the PR of an eigenvector `v` is calculated as:
    PR(v) = (Σ v_i^2)^2 / (Σ v_i^4)
    It measures how many basis elements an eigenvector is spread across. A
    low PR indicates that a principal feature is "localized" to a small
    subset of neurons.

References
----------
1.  Martin, C., & Mahoney, M. W. (2021). "Heavy-Tailed Universals in
    Deep Neural Networks." arXiv preprint arXiv:2106.07590.
2.  Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). "Power-law
    distributions in empirical data." SIAM review, 51(4), 661-703.
3.  Sagun, L., Evci, U., Guney, V. U., Dauphin, Y., & Bottou, L. (2017).
    "Empirical analysis of the hessian of loss functions for deep neural
    networks." arXiv preprint arXiv:1706.04454.

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
            - Array of sorted eigenvalues (descending order).
            - Maximum singular value.
            - Minimum singular value.
            - Total rank loss.
    """
    all_evals = []
    max_sv = 0.0
    min_sv = float('inf')
    rank_loss = 0

    for W in weight_matrices:
        # Ensure float32/64 to avoid scipy issues with half-precision
        W = W.astype(float)

        # For large matrices, use truncated SVD
        if (n_comp < M) or (M > SPECTRAL_DEFAULT_MAX_EVALS):
            try:
                # Use scipy's sparse SVD for large matrices
                # k must be strictly less than min(W.shape) for svds
                k_target = min(n_comp, min(W.shape) - 1)
                if k_target < 1:
                    # Fallback for extremely small matrices
                    sv = np.linalg.svd(W, compute_uv=False)
                else:
                    _, sv, _ = svds(W, k=k_target)
                    sv = np.sort(sv)[::-1]  # Sort descending
            except Exception as e:
                logger.debug(f"Sparse SVD failed: {e}, falling back to full SVD")
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
        tol = current_max_sv * max(W.shape) * np.finfo(W.dtype).eps
        rank = np.sum(sv > tol)
        rank_loss += len(sv) - rank

    # Handle the case where min_sv was never updated
    if min_sv == float('inf'):
        min_sv = 0.0

    # Sort all combined eigenvalues descending
    return np.sort(np.array(all_evals))[::-1], max_sv, min_sv, rank_loss


# ---------------------------------------------------------------------

def fit_powerlaw(
        evals: np.ndarray,
        xmin: Optional[float] = None
) -> Tuple[float, float, float, float, int, str, str]:
    """
    Fit eigenvalues to a power-law distribution using a robust xmin search.

    OPTIMIZED VERSION: Uses pre-computed suffix sums (cumulative sum on reversed array)
    to calculate alpha in O(N) total time, avoiding the O(N^2) bottleneck of
    repeated summation in the original implementation.

    Args:
        evals: Array of eigenvalues to fit.
        xmin: Not used for fitting, as the optimal xmin is found automatically.

    Returns:
        Tuple containing (alpha, xmin, D, sigma, num_pl_spikes, status, warning).
    """
    # 1. Initialization
    alpha, D, sigma, optimal_xmin, num_pl_spikes = -1.0, -1.0, -1.0, -1.0, -1
    status, warning = "failed", ""

    if evals is None or len(evals) < SPECTRAL_DEFAULT_MIN_EVALS:
        logger.debug("Not enough eigenvalues for power-law fitting")
        return alpha, optimal_xmin, D, sigma, num_pl_spikes, status, warning

    try:
        # Prepare data: Sort ascending for xmin iteration
        data = np.sort(evals[evals > SPECTRAL_EVALS_THRESH])

        if len(data) < 2:
            return alpha, optimal_xmin, D, sigma, num_pl_spikes, status, "not enough data > thresh"

        N = len(data)
        log_data = np.log(data)

        # --- OPTIMIZATION START ---
        # Pre-compute the sum of log(x) for the tail starting at each index i.
        # tail_sums[i] == sum(log_data[i:])
        # We use cumsum on the reversed array, then reverse back.
        tail_sums = np.cumsum(log_data[::-1])[::-1]
        # --- OPTIMIZATION END ---

        xmins = data[:-1]
        alphas = np.zeros(N - 1, dtype=np.float64)
        Ds = np.ones(N - 1, dtype=np.float64)

        # Loop through possible xmins
        # Although we iterate, the heavy summation is now O(1) lookup
        for i, current_xmin in enumerate(xmins):
            n_tail = float(N - i)

            # Retrieve pre-computed sum
            sum_log_tail = tail_sums[i]

            # MLE for alpha: 1 + n / (sum(ln x) - n * ln(xmin))
            denominator = sum_log_tail - n_tail * log_data[i]

            if denominator <= SPECTRAL_EPSILON:
                # Avoid division by zero if all values in tail are identical
                current_alpha = 0.0
            else:
                current_alpha = 1.0 + n_tail / denominator

            alphas[i] = current_alpha

            # Calculate KS Distance (D) only if alpha is valid
            if current_alpha > 1.0:
                # Theoretical CDF: P(x) = 1 - (x/xmin)^(-alpha+1)
                # Note: data[i:] contains the tail elements
                theoretical_cdf = 1 - (data[i:] / current_xmin) ** (-current_alpha + 1.0)

                # Empirical CDF: 0 to 1 over n_tail steps
                empirical_cdf = np.arange(n_tail) / n_tail

                # KS distance is max absolute difference
                Ds[i] = np.max(np.abs(theoretical_cdf - empirical_cdf))
            else:
                Ds[i] = float('inf')  # Invalid fit

        # 3. Find the xmin that minimizes the KS distance D
        best_i = np.argmin(Ds)

        # If all fits failed
        if Ds[best_i] == float('inf'):
            return alpha, optimal_xmin, D, sigma, num_pl_spikes, status, "No valid power-law fit found"

        optimal_xmin = xmins[best_i]
        alpha = alphas[best_i]
        D = Ds[best_i]

        # Calculate sigma (standard error) for the best fit
        n_tail_optimal = N - best_i
        sigma = (alpha - 1.0) / np.sqrt(n_tail_optimal)
        num_pl_spikes = int(n_tail_optimal)
        status = "success"

        # 4. Add warning based on alpha value
        if alpha < SPECTRAL_OVER_TRAINED_THRESH:
            warning = "over-trained"
        elif alpha > SPECTRAL_UNDER_TRAINED_THRESH:
            warning = "under-trained"

    except Exception as e:
        logger.warning(f"Power-law fitting failed: {e}")
        status = "failed"
        warning = str(e)

    return alpha, optimal_xmin, D, sigma, num_pl_spikes, status, warning

# ---------------------------------------------------------------------

def calculate_matrix_entropy(singular_values: np.ndarray, N: int) -> float:
    """
    Calculate the matrix entropy from singular values.

    Updated to be more robust against NaNs and zeros.

    Args:
        singular_values: Array of singular values from SVD.
        N: Maximum dimension of the matrix for rank calculation.

    Returns:
        Matrix entropy (float). Returns 0.0 if calculation fails.
    """
    try:
        if len(singular_values) == 0 or np.max(singular_values) < SPECTRAL_EPSILON:
            return 0.0

        # 1. Calculate matrix rank using numpy-compatible tolerance
        S = singular_values
        tol = np.max(S) * N * np.finfo(S.dtype).eps
        rank = np.count_nonzero(S > tol)

        # 2. Calculate eigenvalues from singular values
        evals = S * S

        # 3. Calculate probabilities (normalized eigenvalues)
        evals_sum = np.sum(evals)
        if evals_sum < SPECTRAL_EPSILON:
            return 0.0

        p = evals / evals_sum

        # 4. Compute Entropy: -sum(p * log(p)) / log(rank)
        # Add epsilon to mask zeros before log
        p_valid = p[p > SPECTRAL_EPSILON]

        if len(p_valid) == 0:
            return 0.0

        entropy_unnormalized = -np.sum(p_valid * np.log(p_valid))

        # Avoid division by zero if rank is 1
        log_rank = np.log(rank) if rank > 1 else 1.0

        entropy = entropy_unnormalized / log_rank

        # Handle NaN results from numerical instability
        if np.isnan(entropy):
            return 0.0

        return float(entropy)
    except Exception as e:
        logger.debug(f"Error calculating matrix entropy: {e}")
        return 0.0


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
            "norm": 0.0, "log_norm": 0.0, "spectral_norm": 0.0,
            "log_spectral_norm": 0.0, "alpha_weighted": 0.0,
            "log_alpha_norm": 0.0, "stable_rank": 0.0
        }

    norm = np.sum(evals)
    spectral_norm = np.max(evals)

    # Avoid log(0)
    norm_safe = norm if norm > SPECTRAL_EPSILON else SPECTRAL_EPSILON
    spectral_norm_safe = spectral_norm if spectral_norm > SPECTRAL_EPSILON else SPECTRAL_EPSILON

    return {
        "norm": norm,
        "log_norm": np.log10(norm_safe),
        "spectral_norm": spectral_norm,
        "log_spectral_norm": np.log10(spectral_norm_safe),
        "alpha_weighted": alpha * np.log10(spectral_norm_safe),
        "log_alpha_norm": np.log10(np.sum(evals ** alpha)) if alpha > 0 else 0.0,
        "stable_rank": norm / spectral_norm_safe
    }


# ---------------------------------------------------------------------

def calculate_gini_coefficient(evals: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of the eigenvalue distribution.

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

    Args:
        evals: Array of eigenvalues.

    Returns:
        Dominance ratio.
    """
    if len(evals) < 2:
        return 0.0

    lambda_max = np.max(evals)
    sum_others = np.sum(evals) - lambda_max

    if sum_others < SPECTRAL_EPSILON:
        return 0.0  # Return 0 instead of inf for stability

    return lambda_max / sum_others


# ---------------------------------------------------------------------

def calculate_participation_ratio(vector: np.ndarray) -> float:
    """
    Calculate the Participation Ratio (PR) of a vector.

    Corrected Formula: PR = (sum(v^2))^2 / sum(v^4)
    This measures the effective number of non-zero elements in the vector.

    Args:
        vector: Eigenvector or other vector to analyze.

    Returns:
        Participation ratio.
    """
    # Avoid numerical underflow with extremely small values
    vec_squared = vector ** 2
    sum_sq = np.sum(vec_squared)

    if sum_sq < SPECTRAL_EPSILON:
        return 0.0

    # Numerator: Square of L2 norm (squared sum of squares)
    numerator = sum_sq ** 2

    # Denominator: Sum of fourth powers
    denominator = np.sum(vector ** 4)

    if denominator < SPECTRAL_EPSILON:
        return 0.0

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
            # For large matrices, consider randomized_svd if available, but standard is safer
            u, s, _ = np.linalg.svd(weight_matrix, full_matrices=False)
            # Return squared singular values and left singular vectors
            return s[:k] ** 2, u[:, :k]
        else:
            # Use power iteration for very large matrices (approximation)
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
            prev_q = eigvecs[:, j].reshape(-1, 1)
            q = q - prev_q @ (prev_q.T @ q)

        # Power iteration
        for _ in range(max_iter):
            z = matrix @ q
            z_norm = np.linalg.norm(z)
            if z_norm < SPECTRAL_EPSILON:
                break
            q_new = z / z_norm

            # Check convergence
            if np.linalg.norm(q_new - q) < tol:
                q = q_new
                break
            q = q_new

        # Rayleigh quotient
        lambda_i = (q.T @ matrix @ q)[0, 0]

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
    """
    if eigenvectors.size == 0 or eigenvalues.size == 0:
        return []

    n, m = weight_matrix.shape
    critical_weights = set()

    # Calculate an overall importance score for each row
    # Weighted average of absolute eigenvector components
    total_eigenvalue_sum = np.sum(eigenvalues)

    if total_eigenvalue_sum < SPECTRAL_EPSILON:
        return []

    # Normalize eigenvalues for weighting
    weights_ev = eigenvalues / total_eigenvalue_sum

    # Compute row importance: Sum(weight_k * |u_ik|) across k eigenvectors
    row_importance = np.sum(weights_ev * np.abs(eigenvectors), axis=1)

    # Find rows with high importance scores
    max_importance = np.max(row_importance)
    if max_importance < SPECTRAL_EPSILON:
        return []

    high_importance_row_indices = np.where(row_importance > threshold * max_importance)[0]

    # For each important row, find the weights with the largest magnitudes
    for i in high_importance_row_indices:
        row_weights = weight_matrix[i, :]
        max_weight_in_row = np.max(np.abs(row_weights))

        if max_weight_in_row < SPECTRAL_EPSILON:
            continue

        high_magnitude_col_indices = np.where(np.abs(row_weights) > threshold * max_weight_in_row)[0]

        for j in high_magnitude_col_indices:
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
    """
    min_dim = min(weight_matrix.shape)
    if min_dim < SPECTRAL_DEFAULT_MIN_EVALS:
        return {}

    # Get eigenvalues and eigenvectors
    # Limit num_eigenvectors to actual matrix rank/dimensions
    k_safe = min(num_eigenvectors, min_dim - 1)
    if k_safe < 1:
        return {}

    eigenvalues, eigenvectors = get_top_eigenvectors(weight_matrix, k=k_safe)

    if len(eigenvalues) == 0:
        return {}

    gini = calculate_gini_coefficient(eigenvalues)
    dominance = calculate_dominance_ratio(eigenvalues)

    # Calculate participation ratios for top eigenvectors
    participation_ratios = []
    for i in range(eigenvectors.shape[1]):
        pr = calculate_participation_ratio(eigenvectors[:, i])
        participation_ratios.append(pr)

    critical_weights = find_critical_weights(weight_matrix, eigenvectors, eigenvalues)

    mean_pr = np.mean(participation_ratios) if participation_ratios else 0.0

    # Concentration score: Higher score = More concentrated (bad for robustness)
    # High Gini (inequality) * High Dominance / Low Participation
    concentration_score = (gini * dominance) / (mean_pr + SPECTRAL_EPSILON)

    return {
        'gini_coefficient': gini,
        'dominance_ratio': dominance,
        'participation_ratio': mean_pr,
        'min_participation_ratio': np.min(participation_ratios) if participation_ratios else 0,
        'critical_weight_count': len(critical_weights),
        'critical_weights': critical_weights[:SPECTRAL_MAX_CRITICAL_WEIGHTS_REPORTED],
        'concentration_score': np.log1p(concentration_score)  # Log to manage extreme values
    }


# ---------------------------------------------------------------------

def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Jensen-Shannon distance between two distributions.
    """
    if len(p) == 0 or len(q) == 0:
        return 1.0

    min_val = min(np.min(p), np.min(q))
    max_val = max(np.max(p), np.max(q))

    if min_val == max_val:
        return 0.0

    p_hist, _ = np.histogram(p, bins=SPECTRAL_DEFAULT_BINS, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q, bins=SPECTRAL_DEFAULT_BINS, range=(min_val, max_val), density=True)

    # Normalize histograms to probability distributions
    p_prob = p_hist / (np.sum(p_hist) + SPECTRAL_EPSILON)
    q_prob = q_hist / (np.sum(q_hist) + SPECTRAL_EPSILON)

    # Calculate m = (p+q)/2
    m = (p_prob + q_prob) / 2

    # Calculate Jensen-Shannon Divergence
    divergence = (stats.entropy(p_prob, m) + stats.entropy(q_prob, m)) / 2

    # Calculate Jensen-Shannon Distance
    distance = np.sqrt(max(0.0, divergence))  # Ensure non-negative

    return distance


# ---------------------------------------------------------------------

def rescale_eigenvalues(evals: np.ndarray) -> Tuple[np.ndarray, float]:
    """Rescale eigenvalues by their L2 norm to sqrt(N)."""
    N = len(evals)
    if N == 0:
        return evals, 1.0

    wnorm = np.sqrt(np.sum(evals))
    if wnorm < SPECTRAL_EPSILON:
        return evals, 1.0

    wscale = np.sqrt(N) / wnorm
    rescaled_evals = (wscale * wscale) * evals
    return rescaled_evals, wscale


def compute_detX_constraint(evals: np.ndarray) -> int:
    """
    Identify the number of eigenvalues necessary to satisfy det(X) = 1.
    """
    if evals is None or len(evals) < 2:
        return 0

    rescaled_evals, _ = rescale_eigenvalues(evals)
    sorted_evals = np.sort(rescaled_evals)

    for idx in range(len(sorted_evals) - 1, 0, -1):
        detX = np.prod(sorted_evals[idx:])
        if detX < 1.0:
            num_evals_in_tail = len(sorted_evals) - idx
            return num_evals_in_tail

    return len(sorted_evals)


# ---------------------------------------------------------------------

def smooth_matrix(W: np.ndarray, n_comp: int) -> np.ndarray:
    """
    Apply SVD smoothing to a weight matrix by zeroing out small singular values.
    """
    transpose = False
    if W.shape[0] < W.shape[1]:
        W = W.T
        transpose = True

    try:
        u, s, vh = np.linalg.svd(W, full_matrices=False)

        if n_comp < len(s):
            s[n_comp:] = 0

        smoothed_W = np.dot(u * s[:, np.newaxis] if s.ndim == 1 else u * s, vh)

        if transpose:
            smoothed_W = smoothed_W.T

        return smoothed_W

    except Exception as e:
        logger.warning(f"SVD failed during smoothing: {e}, returning original matrix")
        return W if not transpose else W.T


# ---------------------------------------------------------------------

def calculate_glorot_normalization_factor(N: int, M: int, rf: int) -> float:
    """
    Calculate the Glorot normalization factor for weight initialization.
    """
    if (N + M) * rf == 0:
        return 1.0
    return np.sqrt(2 / ((N + M) * rf))

# ---------------------------------------------------------------------