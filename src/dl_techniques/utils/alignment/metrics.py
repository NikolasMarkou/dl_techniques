"""
Alignment metrics for measuring representation similarity.

This module implements various metrics for quantifying alignment between
neural network representations, including:
- k-NN based metrics (mutual_knn, cycle_knn, lcs_knn)
- Kernel-based metrics (CKA, unbiased CKA, CKNNA)
- SVCCA
- Edit distance

References:
    Huh et al. "The Platonic Representation Hypothesis" ICML 2024
"""

import keras
import tensorflow as tf
import numpy as np
from typing import Union, Tuple, Optional, Callable
from sklearn.cross_decomposition import CCA


class AlignmentMetrics:
    """
    Collection of alignment metrics for neural network representations.
    
    All metrics expect inputs of shape (N, D) where N is the number of samples
    and D is the feature dimension. Features should typically be L2-normalized.
    """
    
    SUPPORTED_METRICS = [
        "cycle_knn",
        "mutual_knn",
        "lcs_knn",
        "cka",
        "unbiased_cka",
        "cknna",
        "svcca",
        "edit_distance_knn",
    ]

    @staticmethod
    def measure(
        metric: str,
        *args,
        **kwargs
    ) -> float:
        """
        Compute alignment using the specified metric.
        
        Args:
            metric: Name of the metric to use
            *args: Positional arguments passed to the metric function
            **kwargs: Keyword arguments passed to the metric function
            
        Returns:
            Alignment score (higher is better for all metrics)
            
        Raises:
            ValueError: If metric name is not recognized
        """
        if metric not in AlignmentMetrics.SUPPORTED_METRICS:
            raise ValueError(
                f"Unrecognized metric: {metric}. "
                f"Supported metrics: {AlignmentMetrics.SUPPORTED_METRICS}"
            )
        
        return getattr(AlignmentMetrics, metric)(*args, **kwargs)

    @staticmethod
    def cycle_knn(
        feats_a: Union[tf.Tensor, keras.ops.Tensor],
        feats_b: Union[tf.Tensor, keras.ops.Tensor],
        topk: int
    ) -> float:
        """
        Cycle KNN: A->B nearest neighbors, then query B->A nearest neighbors.
        
        Measures how well the k-nearest neighbor structure is preserved when
        cycling from representation A to B and back to A.
        
        Args:
            feats_a: Features from model A, shape (N, D)
            feats_b: Features from model B, shape (N, D)
            topk: Number of nearest neighbors to consider
            
        Returns:
            Accuracy score in [0, 1]
        """
        feats_a = keras.ops.convert_to_tensor(feats_a)
        feats_b = keras.ops.convert_to_tensor(feats_b)
        
        knn_a = _compute_nearest_neighbors(feats_a, topk)
        knn_b = _compute_nearest_neighbors(feats_b, topk)
        
        # Index knn_b with knn_a results
        # knn_a shape: (N, topk), values are indices
        # We want knn_b[knn_a] which requires gathering
        n = keras.ops.shape(feats_a)[0]
        
        # Reshape for gathering
        knn_a_flat = keras.ops.reshape(knn_a, [-1])
        knn_b_gathered = keras.ops.take(knn_b, knn_a_flat, axis=0)
        knn_b_gathered = keras.ops.reshape(knn_b_gathered, [n, topk, topk])
        
        return float(_compute_knn_accuracy(knn_b_gathered))

    @staticmethod
    def mutual_knn(
        feats_a: Union[tf.Tensor, keras.ops.Tensor],
        feats_b: Union[tf.Tensor, keras.ops.Tensor],
        topk: int
    ) -> float:
        """
        Mutual KNN: Intersection of nearest neighbors from both representations.
        
        Measures the overlap between the k-nearest neighbors in representation A
        and representation B.
        
        Args:
            feats_a: Features from model A, shape (N, D)
            feats_b: Features from model B, shape (N, D)
            topk: Number of nearest neighbors to consider
            
        Returns:
            Average mutual KNN accuracy in [0, 1]
        """
        feats_a = keras.ops.convert_to_tensor(feats_a)
        feats_b = keras.ops.convert_to_tensor(feats_b)
        
        knn_a = _compute_nearest_neighbors(feats_a, topk)
        knn_b = _compute_nearest_neighbors(feats_b, topk)
        
        n = keras.ops.shape(feats_a)[0]
        
        # Create range tensor for indexing
        range_tensor = keras.ops.arange(n)
        range_tensor = keras.ops.reshape(range_tensor, [-1, 1])
        
        # Create binary masks
        # Initialize with zeros
        mask_a = keras.ops.zeros([n, n])
        mask_b = keras.ops.zeros([n, n])
        
        # Convert to numpy for scatter operation (Keras doesn't have scatter_nd)
        mask_a_np = keras.ops.convert_to_numpy(mask_a)
        mask_b_np = keras.ops.convert_to_numpy(mask_b)
        knn_a_np = keras.ops.convert_to_numpy(knn_a)
        knn_b_np = keras.ops.convert_to_numpy(knn_b)
        range_np = keras.ops.convert_to_numpy(range_tensor)
        
        # Set positions to 1
        for i in range(n):
            mask_a_np[i, knn_a_np[i]] = 1.0
            mask_b_np[i, knn_b_np[i]] = 1.0
        
        mask_a = keras.ops.convert_to_tensor(mask_a_np)
        mask_b = keras.ops.convert_to_tensor(mask_b_np)
        
        # Compute intersection
        acc = keras.ops.sum(mask_a * mask_b, axis=1) / float(topk)
        
        return float(keras.ops.mean(acc))

    @staticmethod
    def lcs_knn(
        feats_a: Union[tf.Tensor, keras.ops.Tensor],
        feats_b: Union[tf.Tensor, keras.ops.Tensor],
        topk: int
    ) -> float:
        """
        Longest Common Subsequence (LCS) of k-nearest neighbors.
        
        Measures the length of the longest common subsequence between the
        k-nearest neighbor orderings from both representations.
        
        Args:
            feats_a: Features from model A, shape (N, D)
            feats_b: Features from model B, shape (N, D)
            topk: Number of nearest neighbors to consider
            
        Returns:
            Average normalized LCS length in [0, 1]
        """
        feats_a = keras.ops.convert_to_tensor(feats_a)
        feats_b = keras.ops.convert_to_tensor(feats_b)
        
        knn_a = _compute_nearest_neighbors(feats_a, topk)
        knn_b = _compute_nearest_neighbors(feats_b, topk)
        
        score = _longest_ordinal_sequence(knn_a, knn_b)
        return float(keras.ops.mean(score))

    @staticmethod
    def cka(
        feats_a: Union[tf.Tensor, keras.ops.Tensor],
        feats_b: Union[tf.Tensor, keras.ops.Tensor],
        kernel_metric: str = 'ip',
        rbf_sigma: float = 1.0,
        unbiased: bool = False
    ) -> float:
        """
        Centered Kernel Alignment (CKA).
        
        Measures similarity between representations using kernel methods.
        
        Args:
            feats_a: Features from model A, shape (N, D)
            feats_b: Features from model B, shape (N, D)
            kernel_metric: 'ip' for inner product or 'rbf' for RBF kernel
            rbf_sigma: Bandwidth parameter for RBF kernel
            unbiased: Whether to use unbiased HSIC estimator
            
        Returns:
            CKA score in [0, 1]
        """
        feats_a = keras.ops.convert_to_tensor(feats_a)
        feats_b = keras.ops.convert_to_tensor(feats_b)
        
        # Compute kernel matrices
        if kernel_metric == 'ip':
            k_mat = keras.ops.matmul(feats_a, keras.ops.transpose(feats_a))
            l_mat = keras.ops.matmul(feats_b, keras.ops.transpose(feats_b))
        elif kernel_metric == 'rbf':
            # RBF kernel: exp(-||x_i - x_j||^2 / (2 * sigma^2))
            k_mat = _rbf_kernel(feats_a, feats_a, rbf_sigma)
            l_mat = _rbf_kernel(feats_b, feats_b, rbf_sigma)
        else:
            raise ValueError(f"Invalid kernel metric: {kernel_metric}")
        
        # Compute HSIC values
        hsic_fn = _hsic_unbiased if unbiased else _hsic_biased
        hsic_kk = hsic_fn(k_mat, k_mat)
        hsic_ll = hsic_fn(l_mat, l_mat)
        hsic_kl = hsic_fn(k_mat, l_mat)
        
        # Compute CKA
        denominator = keras.ops.sqrt(hsic_kk * hsic_ll) + 1e-6
        cka_value = hsic_kl / denominator
        
        return float(cka_value)

    @staticmethod
    def unbiased_cka(*args, **kwargs) -> float:
        """
        Unbiased Centered Kernel Alignment.
        
        Same as CKA but uses unbiased HSIC estimator.
        
        Args:
            *args: Positional arguments passed to cka()
            **kwargs: Keyword arguments passed to cka()
            
        Returns:
            Unbiased CKA score in [0, 1]
        """
        kwargs['unbiased'] = True
        return AlignmentMetrics.cka(*args, **kwargs)

    @staticmethod
    def svcca(
        feats_a: Union[tf.Tensor, keras.ops.Tensor],
        feats_b: Union[tf.Tensor, keras.ops.Tensor],
        cca_dim: int = 10
    ) -> float:
        """
        Singular Vector Canonical Correlation Analysis (SVCCA).
        
        Combines SVD dimensionality reduction with CCA to measure
        representation similarity.
        
        Args:
            feats_a: Features from model A, shape (N, D)
            feats_b: Features from model B, shape (N, D)
            cca_dim: Number of CCA components to use
            
        Returns:
            SVCCA similarity score in [0, 1]
        """
        # Convert to numpy for sklearn
        feats_a_np = keras.ops.convert_to_numpy(feats_a)
        feats_b_np = keras.ops.convert_to_numpy(feats_b)
        
        # Center and scale
        feats_a_np = _preprocess_activations(feats_a_np)
        feats_b_np = _preprocess_activations(feats_b_np)
        
        # Compute SVD (low-rank approximation)
        u1, s1, vt1 = np.linalg.svd(feats_a_np, full_matrices=False)
        u2, s2, vt2 = np.linalg.svd(feats_b_np, full_matrices=False)
        
        # Take top cca_dim components
        u1 = u1[:, :cca_dim] * s1[:cca_dim]
        u2 = u2[:, :cca_dim] * s2[:cca_dim]
        
        # Add small noise to avoid numerical issues
        u1 += 1e-10 * np.random.randn(*u1.shape)
        u2 += 1e-10 * np.random.randn(*u2.shape)
        
        # Compute CCA
        cca = CCA(n_components=cca_dim)
        cca.fit(u1, u2)
        u1_c, u2_c = cca.transform(u1, u2)
        
        # Compute SVCCA similarity as mean correlation
        correlations = []
        for i in range(cca_dim):
            corr = np.corrcoef(u1_c[:, i], u2_c[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 0.0

    @staticmethod
    def edit_distance_knn(
        feats_a: Union[tf.Tensor, keras.ops.Tensor],
        feats_b: Union[tf.Tensor, keras.ops.Tensor],
        topk: int
    ) -> float:
        """
        Edit distance between k-nearest neighbor orderings.
        
        Measures the minimum number of edits needed to transform one
        k-NN ordering into another.
        
        Args:
            feats_a: Features from model A, shape (N, D)
            feats_b: Features from model B, shape (N, D)
            topk: Number of nearest neighbors to consider
            
        Returns:
            Normalized edit distance similarity in [0, 1]
        """
        feats_a = keras.ops.convert_to_tensor(feats_a)
        feats_b = keras.ops.convert_to_tensor(feats_b)
        
        knn_a = _compute_nearest_neighbors(feats_a, topk)
        knn_b = _compute_nearest_neighbors(feats_b, topk)
        
        # Convert to numpy for edit distance computation
        knn_a_np = keras.ops.convert_to_numpy(knn_a)
        knn_b_np = keras.ops.convert_to_numpy(knn_b)
        
        distances = _compute_batch_edit_distance(knn_a_np, knn_b_np)
        
        # Normalize by sequence length and convert to similarity
        similarity = 1.0 - (np.mean(distances) / topk)
        return float(similarity)

    @staticmethod
    def cknna(
        feats_a: Union[tf.Tensor, keras.ops.Tensor],
        feats_b: Union[tf.Tensor, keras.ops.Tensor],
        topk: Optional[int] = None,
        distance_agnostic: bool = False,
        unbiased: bool = True
    ) -> float:
        """
        Centered Kernel Alignment with Nearest Neighbor Attention (CKNNA).
        
        A variant of CKA that only considers nearest neighbors.
        
        Args:
            feats_a: Features from model A, shape (N, D)
            feats_b: Features from model B, shape (N, D)
            topk: Number of nearest neighbors (default: N-1)
            distance_agnostic: Whether to ignore distance magnitudes
            unbiased: Whether to exclude diagonal in computation
            
        Returns:
            CKNNA score in [0, 1]
        """
        feats_a = keras.ops.convert_to_tensor(feats_a)
        feats_b = keras.ops.convert_to_tensor(feats_b)
        
        n = keras.ops.shape(feats_a)[0]
        
        if topk is None:
            topk = int(n - 1)
        
        if topk < 2:
            raise ValueError("CKNNA requires topk >= 2")
        
        # Compute similarity matrices
        k_mat = keras.ops.matmul(feats_a, keras.ops.transpose(feats_a))
        l_mat = keras.ops.matmul(feats_b, keras.ops.transpose(feats_b))
        
        # Helper to create nearest neighbor mask
        def _similarity_with_mask(k, l, topk_val):
            if unbiased:
                # Set diagonal to -inf for topk selection
                k_hat = keras.ops.where(
                    keras.ops.eye(n, dtype=keras.ops.dtype(k)) > 0,
                    keras.ops.full([n, n], float('-inf')),
                    k
                )
                l_hat = keras.ops.where(
                    keras.ops.eye(n, dtype=keras.ops.dtype(l)) > 0,
                    keras.ops.full([n, n], float('-inf')),
                    l
                )
            else:
                k_hat, l_hat = k, l
            
            # Get topk indices
            _, topk_k_indices = tf.nn.top_k(k_hat, topk_val)
            _, topk_l_indices = tf.nn.top_k(l_hat, topk_val)
            
            # Create masks
            mask_k = keras.ops.zeros([n, n])
            mask_l = keras.ops.zeros([n, n])
            
            # Convert to numpy for scatter
            mask_k_np = keras.ops.convert_to_numpy(mask_k)
            mask_l_np = keras.ops.convert_to_numpy(mask_l)
            topk_k_np = keras.ops.convert_to_numpy(topk_k_indices)
            topk_l_np = keras.ops.convert_to_numpy(topk_l_indices)
            
            for i in range(int(n)):
                mask_k_np[i, topk_k_np[i]] = 1.0
                mask_l_np[i, topk_l_np[i]] = 1.0
            
            mask_k = keras.ops.convert_to_tensor(mask_k_np)
            mask_l = keras.ops.convert_to_tensor(mask_l_np)
            
            # Intersection mask
            mask = mask_k * mask_l
            
            if distance_agnostic:
                sim = mask * 1.0
            else:
                if unbiased:
                    sim = _hsic_unbiased(mask * k, mask * l)
                else:
                    sim = _hsic_biased(mask * k, mask * l)
            
            return sim
        
        sim_kl = _similarity_with_mask(k_mat, l_mat, topk)
        sim_kk = _similarity_with_mask(k_mat, k_mat, topk)
        sim_ll = _similarity_with_mask(l_mat, l_mat, topk)
        
        denominator = keras.ops.sqrt(sim_kk * sim_ll) + 1e-6
        score = sim_kl / denominator
        
        return float(score)


# Helper functions

def _compute_nearest_neighbors(
    feats: Union[tf.Tensor, keras.ops.Tensor],
    topk: int
) -> tf.Tensor:
    """
    Compute k-nearest neighbors for each sample.
    
    Args:
        feats: Features, shape (N, D)
        topk: Number of nearest neighbors
        
    Returns:
        Indices of nearest neighbors, shape (N, topk)
    """
    feats = keras.ops.convert_to_tensor(feats)
    
    # Compute similarity matrix
    sim_matrix = keras.ops.matmul(feats, keras.ops.transpose(feats))
    
    # Set diagonal to very negative value
    n = keras.ops.shape(feats)[0]
    diagonal_mask = keras.ops.eye(n, dtype=keras.ops.dtype(sim_matrix))
    sim_matrix = keras.ops.where(
        diagonal_mask > 0,
        keras.ops.full([n, n], -1e8),
        sim_matrix
    )
    
    # Get topk indices
    _, indices = tf.nn.top_k(sim_matrix, topk)
    
    return indices


def _compute_knn_accuracy(knn: Union[tf.Tensor, keras.ops.Tensor]) -> tf.Tensor:
    """
    Compute accuracy of k-NN predictions.
    
    Args:
        knn: k-NN indices, shape (N, K) or (N, K, K)
        
    Returns:
        Accuracy score
    """
    knn = keras.ops.convert_to_tensor(knn)
    n = keras.ops.shape(knn)[0]
    
    # Ground truth indices
    gt = keras.ops.arange(n)
    gt = keras.ops.reshape(gt, [-1, 1, 1])
    
    # Check if any k-NN matches ground truth
    matches = keras.ops.equal(knn, gt)
    matches = keras.ops.cast(matches, dtype='float32')
    
    # Reshape and compute accuracy
    matches = keras.ops.reshape(matches, [n, -1])
    acc = keras.ops.max(matches, axis=1)
    
    return keras.ops.mean(acc)


def _hsic_unbiased(
    k_mat: Union[tf.Tensor, keras.ops.Tensor],
    l_mat: Union[tf.Tensor, keras.ops.Tensor]
) -> tf.Tensor:
    """
    Unbiased Hilbert-Schmidt Independence Criterion (HSIC).
    
    Reference: Song et al. "Feature Selection via Dependence Maximization" JMLR 2012
    
    Args:
        k_mat: Kernel matrix K, shape (N, N)
        l_mat: Kernel matrix L, shape (N, N)
        
    Returns:
        Unbiased HSIC value
    """
    k_mat = keras.ops.convert_to_tensor(k_mat)
    l_mat = keras.ops.convert_to_tensor(l_mat)
    
    m = keras.ops.shape(k_mat)[0]
    m_float = keras.ops.cast(m, dtype=k_mat.dtype)
    
    # Zero out diagonal
    eye = keras.ops.eye(m, dtype=k_mat.dtype)
    k_tilde = keras.ops.where(eye > 0, keras.ops.zeros_like(k_mat), k_mat)
    l_tilde = keras.ops.where(eye > 0, keras.ops.zeros_like(l_mat), l_mat)
    
    # Compute terms
    term1 = keras.ops.sum(k_tilde * keras.ops.transpose(l_tilde))
    
    sum_k = keras.ops.sum(k_tilde)
    sum_l = keras.ops.sum(l_tilde)
    term2 = (sum_k * sum_l) / ((m_float - 1.0) * (m_float - 2.0))
    
    kl_product = keras.ops.matmul(k_tilde, l_tilde)
    term3 = (2.0 * keras.ops.sum(kl_product)) / (m_float - 2.0)
    
    # Combine terms
    hsic_value = term1 + term2 - term3
    hsic_value = hsic_value / (m_float * (m_float - 3.0))
    
    return hsic_value


def _hsic_biased(
    k_mat: Union[tf.Tensor, keras.ops.Tensor],
    l_mat: Union[tf.Tensor, keras.ops.Tensor]
) -> tf.Tensor:
    """
    Biased Hilbert-Schmidt Independence Criterion (HSIC).
    
    This is the original CKA formulation.
    
    Args:
        k_mat: Kernel matrix K, shape (N, N)
        l_mat: Kernel matrix L, shape (N, N)
        
    Returns:
        Biased HSIC value
    """
    k_mat = keras.ops.convert_to_tensor(k_mat)
    l_mat = keras.ops.convert_to_tensor(l_mat)
    
    n = keras.ops.shape(k_mat)[0]
    n_float = keras.ops.cast(n, dtype=k_mat.dtype)
    
    # Centering matrix H = I - 1/n * 11^T
    eye = keras.ops.eye(n, dtype=k_mat.dtype)
    ones = keras.ops.ones([n, n], dtype=k_mat.dtype)
    h_mat = eye - ones / n_float
    
    # HSIC = trace(K H L H)
    centered_k = keras.ops.matmul(keras.ops.matmul(k_mat, h_mat), l_mat)
    hsic_value = keras.ops.trace(keras.ops.matmul(centered_k, h_mat))
    
    return hsic_value


def _rbf_kernel(
    x: Union[tf.Tensor, keras.ops.Tensor],
    y: Union[tf.Tensor, keras.ops.Tensor],
    sigma: float
) -> tf.Tensor:
    """
    RBF (Gaussian) kernel.
    
    Args:
        x: Features, shape (N, D)
        y: Features, shape (M, D)
        sigma: Bandwidth parameter
        
    Returns:
        Kernel matrix, shape (N, M)
    """
    x = keras.ops.convert_to_tensor(x)
    y = keras.ops.convert_to_tensor(y)
    
    # Compute pairwise squared distances
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    x_norm = keras.ops.sum(x ** 2, axis=1, keepdims=True)
    y_norm = keras.ops.sum(y ** 2, axis=1, keepdims=True)
    
    dist_sq = x_norm + keras.ops.transpose(y_norm) - 2.0 * keras.ops.matmul(x, keras.ops.transpose(y))
    
    # RBF kernel
    kernel = keras.ops.exp(-dist_sq / (2.0 * sigma ** 2))
    
    return kernel


def _preprocess_activations(act: np.ndarray) -> np.ndarray:
    """
    Center and scale activations for SVCCA.
    
    Args:
        act: Activations, shape (N, D)
        
    Returns:
        Preprocessed activations
    """
    act = act - np.mean(act, axis=0, keepdims=True)
    act = act / (np.std(act, axis=0, keepdims=True) + 1e-8)
    return act


def _longest_ordinal_sequence(
    x: Union[tf.Tensor, keras.ops.Tensor],
    y: Union[tf.Tensor, keras.ops.Tensor]
) -> tf.Tensor:
    """
    Compute longest common subsequence (LCS) length for each pair.
    
    Args:
        x: Sequence indices, shape (N, K)
        y: Sequence indices, shape (N, K)
        
    Returns:
        LCS lengths normalized by K, shape (N,)
    """
    x_np = keras.ops.convert_to_numpy(x)
    y_np = keras.ops.convert_to_numpy(y)
    
    n, k = x_np.shape
    lcs_lengths = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        lcs_lengths[i] = _lcs_length(x_np[i], y_np[i])
    
    # Normalize by sequence length
    lcs_lengths = lcs_lengths / float(k)
    
    return keras.ops.convert_to_tensor(lcs_lengths)


def _lcs_length(seq_x: np.ndarray, seq_y: np.ndarray) -> int:
    """
    Compute longest common subsequence length using dynamic programming.
    
    Args:
        seq_x: First sequence
        seq_y: Second sequence
        
    Returns:
        LCS length
    """
    m, n = len(seq_x), len(seq_y)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_x[i - 1] == seq_y[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    
    return int(dp[m, n])


def _compute_batch_edit_distance(
    x: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Compute edit distance for each pair in batch.
    
    Args:
        x: Sequence indices, shape (N, K)
        y: Sequence indices, shape (N, K)
        
    Returns:
        Edit distances, shape (N,)
    """
    n = x.shape[0]
    distances = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        distances[i] = _edit_distance(x[i], y[i])
    
    return distances


def _edit_distance(seq_x: np.ndarray, seq_y: np.ndarray) -> int:
    """
    Compute edit distance (Levenshtein distance) between two sequences.
    
    Args:
        seq_x: First sequence
        seq_y: Second sequence
        
    Returns:
        Edit distance
    """
    m, n = len(seq_x), len(seq_y)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    
    # Initialize
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_x[i - 1] == seq_y[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(
                    dp[i - 1, j],      # deletion
                    dp[i, j - 1],      # insertion
                    dp[i - 1, j - 1]   # substitution
                )
    
    return int(dp[m, n])


def remove_outliers(
    feats: Union[tf.Tensor, keras.ops.Tensor],
    q: float = 0.95,
    exact: bool = False,
    max_threshold: Optional[float] = None
) -> tf.Tensor:
    """
    Remove outliers by clipping to quantile.
    
    Args:
        feats: Features of any shape
        q: Quantile for clipping (default: 0.95)
        exact: Whether to compute exact quantile (slower)
        max_threshold: Optional maximum threshold
        
    Returns:
        Clipped features
    """
    if q == 1.0:
        return feats
    
    feats = keras.ops.convert_to_tensor(feats)
    
    if exact:
        # Exact quantile computation
        feats_flat = keras.ops.reshape(keras.ops.abs(feats), [-1])
        feats_sorted = keras.ops.sort(feats_flat)
        idx = int(q * keras.ops.shape(feats_sorted)[0])
        q_val = feats_sorted[idx]
    else:
        # Approximate: mean of per-sample quantiles
        feats_abs = keras.ops.abs(feats)
        # Flatten all but first dimension
        original_shape = keras.ops.shape(feats)
        if len(feats.shape) > 2:
            feats_reshaped = keras.ops.reshape(feats_abs, [original_shape[0], -1])
        else:
            feats_reshaped = feats_abs
        
        # Compute quantile per sample using TensorFlow
        feats_tf = tf.convert_to_tensor(feats_reshaped)
        quantiles = []
        for i in range(feats_tf.shape[0]):
            q_i = tfp.stats.quantiles(feats_tf[i], q, interpolation='linear')
            quantiles.append(q_i)
        q_val = keras.ops.mean(keras.ops.stack(quantiles))
    
    if max_threshold is not None:
        q_val = keras.ops.maximum(q_val, max_threshold)
    
    # Clip
    feats_clipped = keras.ops.clip(feats, -q_val, q_val)
    
    return feats_clipped


# Import tensorflow_probability for quantile computation
try:
    import tensorflow_probability as tfp
except ImportError:
    # Fallback without tfp
    def remove_outliers(
        feats: Union[tf.Tensor, keras.ops.Tensor],
        q: float = 0.95,
        exact: bool = False,
        max_threshold: Optional[float] = None
    ) -> tf.Tensor:
        """Simplified version without tensorflow_probability."""
        if q == 1.0:
            return feats
        
        feats = keras.ops.convert_to_tensor(feats)
        feats_abs = keras.ops.abs(feats)
        
        # Use max as approximate quantile
        q_val = keras.ops.max(feats_abs) * q
        
        if max_threshold is not None:
            q_val = keras.ops.maximum(q_val, max_threshold)
        
        return keras.ops.clip(feats, -q_val, q_val)
