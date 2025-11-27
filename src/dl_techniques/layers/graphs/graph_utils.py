"""
Graph Data Utilities for sHGCN.

This module provides utility functions for preparing graph data, including
adjacency matrix normalization and sparse tensor creation, which are essential
for efficient graph neural network operations.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Union
import scipy.sparse as sp


def normalize_adjacency_symmetric(
        adj_matrix: Union[np.ndarray, sp.spmatrix],
        add_self_loops: bool = True
) -> sp.coo_matrix:
    """
    Symmetric normalization of adjacency matrix: D^{-1/2} A D^{-1/2}.

    This is the standard normalization used in Graph Convolutional Networks (GCN)
    and many graph neural network variants. It ensures that the aggregation
    operation preserves scale across different node degrees.

    Mathematical Operation:
        1. Add self-loops: Ã = A + I (optional)
        2. Compute degree matrix: D_ii = sum_j Ã_ij
        3. Compute normalization: Ã_norm = D^{-1/2} Ã D^{-1/2}

    where D^{-1/2} is the diagonal matrix with D^{-1/2}_ii = 1/sqrt(D_ii).

    Args:
        adj_matrix: Adjacency matrix as numpy array or scipy sparse matrix.
            Shape (num_nodes, num_nodes). Can be weighted or binary.
        add_self_loops: Whether to add self-loops (A + I). Recommended for GNNs
            to ensure each node includes its own features in aggregation.
            Defaults to True.

    Returns:
        Normalized sparse adjacency matrix in COO format.

    Example:
        ```python
        # Create adjacency matrix (undirected graph with 5 nodes)
        adj = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0]
        ])

        # Normalize
        adj_norm = normalize_adjacency_symmetric(adj)

        # Convert to TensorFlow sparse tensor
        adj_sparse = sparse_to_tf_sparse(adj_norm)
        ```

    Note:
        - Input can be dense or sparse; output is always sparse COO format
        - Self-loops are added before normalization if requested
        - Handles disconnected nodes by setting their degree to 1
    """
    # Convert to scipy sparse matrix if needed
    if not sp.issparse(adj_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix)

    # Add self-loops: A_tilde = A + I
    if add_self_loops:
        adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0], format='csr')

    # Compute degree matrix D
    # D_ii = sum_j A_ij (row sum)
    degree = np.array(adj_matrix.sum(axis=1)).flatten()

    # Handle isolated nodes (degree = 0) by setting to 1
    degree[degree == 0] = 1.0

    # Compute D^{-1/2}
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt_mat = sp.diags(degree_inv_sqrt, format='csr')

    # Compute D^{-1/2} A D^{-1/2}
    adj_normalized = degree_inv_sqrt_mat @ adj_matrix @ degree_inv_sqrt_mat

    return adj_normalized.tocoo()


def normalize_adjacency_row(
        adj_matrix: Union[np.ndarray, sp.spmatrix],
        add_self_loops: bool = True
) -> sp.coo_matrix:
    """
    Row normalization of adjacency matrix: D^{-1} A.

    This normalization ensures that aggregated features from neighbors have
    consistent scale by dividing each row by the node's degree. It's equivalent
    to computing the mean of neighbor features.

    Mathematical Operation:
        1. Add self-loops: Ã = A + I (optional)
        2. Compute degree matrix: D_ii = sum_j Ã_ij
        3. Compute normalization: Ã_norm = D^{-1} Ã

    Args:
        adj_matrix: Adjacency matrix as numpy array or scipy sparse matrix.
            Shape (num_nodes, num_nodes).
        add_self_loops: Whether to add self-loops. Defaults to True.

    Returns:
        Row-normalized sparse adjacency matrix in COO format.

    Example:
        ```python
        adj = create_adjacency_matrix(...)
        adj_norm = normalize_adjacency_row(adj)
        adj_sparse = sparse_to_tf_sparse(adj_norm)
        ```

    Note:
        - Each row sums to 1.0 after normalization
        - Useful when treating aggregation as averaging neighbor features
    """
    # Convert to scipy sparse matrix if needed
    if not sp.issparse(adj_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix)

    # Add self-loops
    if add_self_loops:
        adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0], format='csr')

    # Compute degree matrix D
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0

    # Compute D^{-1}
    degree_inv = np.power(degree, -1.0)
    degree_inv_mat = sp.diags(degree_inv, format='csr')

    # Compute D^{-1} A
    adj_normalized = degree_inv_mat @ adj_matrix

    return adj_normalized.tocoo()


def sparse_to_tf_sparse(
        sparse_matrix: sp.spmatrix
) -> tf.sparse.SparseTensor:
    """
    Convert scipy sparse matrix to TensorFlow SparseTensor.

    TensorFlow's sparse operations require SparseTensor format, which stores
    non-zero values with their indices. This is essential for efficient
    large-scale graph neural network computations.

    Args:
        sparse_matrix: Scipy sparse matrix in any format (will be converted to COO).

    Returns:
        TensorFlow SparseTensor with the same values and structure.

    Example:
        ```python
        # From scipy sparse
        scipy_sparse = sp.coo_matrix(([[1, 2], [3, 4]]))
        tf_sparse = sparse_to_tf_sparse(scipy_sparse)

        # Use in model
        output = model([features, tf_sparse])
        ```

    Note:
        - Output tensor is automatically reordered for efficiency
        - Preserves data type from input matrix
    """
    # Convert to COO format for easier indexing
    coo = sp.coo_matrix(sparse_matrix)

    # Extract indices and values
    indices = np.column_stack((coo.row, coo.col)).astype(np.int64)
    values = coo.data.astype(np.float32)
    shape = coo.shape

    # Create TensorFlow SparseTensor
    sparse_tensor = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=shape
    )

    # Reorder for efficient operations
    sparse_tensor = tf.sparse.reorder(sparse_tensor)

    return sparse_tensor


def create_random_graph(
        num_nodes: int,
        edge_probability: float = 0.1,
        add_self_loops: bool = True,
        symmetric: bool = True
) -> Tuple[sp.coo_matrix, tf.sparse.SparseTensor]:
    """
    Create a random graph for testing or synthetic experiments.

    Generates an Erdős-Rényi random graph where each potential edge exists
    independently with probability p. Useful for testing graph neural networks
    or creating synthetic datasets.

    Args:
        num_nodes: Number of nodes in the graph.
        edge_probability: Probability of edge between any two nodes. Should be
            in range (0, 1). Defaults to 0.1.
        add_self_loops: Whether to add self-loops to all nodes. Defaults to True.
        symmetric: Whether to make the graph undirected (symmetric adjacency).
            Defaults to True.

    Returns:
        Tuple of:
        - Scipy sparse adjacency matrix (COO format)
        - TensorFlow sparse tensor (reordered)

    Example:
        ```python
        # Create random graph
        adj_scipy, adj_tf = create_random_graph(
            num_nodes=1000,
            edge_probability=0.05,
            symmetric=True
        )

        # Use in model
        features = ops.random.normal((1000, 64))
        output = model([features, adj_tf])
        ```
    """
    # Create random adjacency matrix
    adj = sp.random(
        num_nodes,
        num_nodes,
        density=edge_probability,
        format='csr'
    )

    # Make binary (0 or 1)
    adj.data = np.ones_like(adj.data)

    # Make symmetric for undirected graph
    if symmetric:
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)  # Remove double edges

    # Add self-loops
    if add_self_loops:
        adj = adj + sp.eye(num_nodes, format='csr')
        adj.data = np.ones_like(adj.data)

    # Convert to COO
    adj_coo = adj.tocoo()

    # Create TensorFlow version
    adj_tf = sparse_to_tf_sparse(adj_coo)

    return adj_coo, adj_tf


def preprocess_features(
        features: np.ndarray,
        normalize: bool = True
) -> np.ndarray:
    """
    Preprocess node features for graph neural networks.

    Optionally normalizes features to have zero mean and unit variance per
    feature dimension. This standardization improves training stability.

    Args:
        features: Node feature matrix of shape (num_nodes, num_features).
        normalize: Whether to standardize features (z-score normalization).
            When True, each feature dimension is normalized to mean=0, std=1.
            Defaults to True.

    Returns:
        Preprocessed feature matrix with same shape as input.

    Example:
        ```python
        # Raw features
        features = np.random.randn(1000, 64)

        # Preprocess
        features_norm = preprocess_features(features, normalize=True)

        # Verify normalization
        print(features_norm.mean(axis=0))  # Close to 0
        print(features_norm.std(axis=0))   # Close to 1
        ```

    Note:
        - Handles zero-variance features by avoiding division by zero
        - Preserves input dtype
    """
    if normalize:
        # Compute mean and std per feature
        mean = features.mean(axis=0)
        std = features.std(axis=0)

        # Avoid division by zero for constant features
        std[std == 0] = 1.0

        # Standardize
        features = (features - mean) / std

    return features


def sample_negative_edges(
        num_nodes: int,
        positive_edges: np.ndarray,
        num_samples: int,
        max_attempts: int = 1000
) -> np.ndarray:
    """
    Sample negative edges (non-existent edges) for link prediction training.

    For link prediction, we need both positive examples (actual edges) and
    negative examples (non-edges) to train the model to distinguish between them.
    This function samples node pairs that are NOT connected in the graph.

    Args:
        num_nodes: Total number of nodes in the graph.
        positive_edges: Array of shape (num_edges, 2) with existing edges.
        num_samples: Number of negative samples to generate.
        max_attempts: Maximum attempts per sample to avoid infinite loops on
            dense graphs. Defaults to 1000.

    Returns:
        Array of shape (num_samples, 2) with negative edge pairs [src, tgt].

    Example:
        ```python
        # Existing edges in graph
        pos_edges = np.array([[0, 1], [1, 2], [2, 3]])

        # Sample negative edges
        neg_edges = sample_negative_edges(
            num_nodes=100,
            positive_edges=pos_edges,
            num_samples=100
        )

        # Create training data
        edge_pairs = np.vstack([pos_edges, neg_edges])
        labels = np.array([1] * len(pos_edges) + [0] * len(neg_edges))
        ```

    Note:
        - Avoids sampling edges that already exist
        - For very dense graphs, consider reducing num_samples or increasing max_attempts
        - May return fewer samples than requested if max_attempts is reached
    """
    # Create set of existing edges for fast lookup
    positive_set = set(map(tuple, positive_edges))

    negative_edges = []
    attempts = 0

    while len(negative_edges) < num_samples and attempts < max_attempts:
        # Sample random node pairs
        src = np.random.randint(0, num_nodes)
        tgt = np.random.randint(0, num_nodes)

        # Check if this is a valid negative edge
        # (not in positive set, not self-loop)
        if src != tgt and (src, tgt) not in positive_set:
            negative_edges.append([src, tgt])

        attempts += 1

    if len(negative_edges) < num_samples:
        print(
            f"Warning: Only sampled {len(negative_edges)} negative edges "
            f"out of {num_samples} requested."
        )

    return np.array(negative_edges, dtype=np.int64)