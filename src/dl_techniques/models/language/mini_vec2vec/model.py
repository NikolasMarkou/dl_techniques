"""
Unsupervised alignment of two embedding spaces by a single square linear
map, fitted with clustering, quadratic assignment and Procrustes rather
than by gradient descent.

The premise is that independently trained encoders of the same data arrive
at nearly the same relative geometry, differing mainly in coordinate frame.
If so, the map from space A to space B needs no capacity beyond a rotation
and reflection, and the work moves from function fitting to correspondence:
knowing which point of A is which point of B. `align` runs five numpy
stages, none of them a gradient step: center and L2-normalize both spaces,
build an initial correspondence by clustering each space and matching
centroids with quadratic assignment, solve orthogonal Procrustes for an
initial `W`, then refine it with two alternating rounds — one that
resamples nearest neighbours under the current `W`, one that seeds
k-means on B from `W`-transformed centroids of A. Each refinement step
blends into `W` by exponential smoothing rather than replacing it outright,
which is why the shipped `W` is only approximately orthogonal: a convex
blend of two orthogonal matrices is not itself orthogonal. Do not assume
`W^-1 = W^T` when mapping back from B to A.

`W` is a trainable weight but is only ever written by `assign` from numpy;
the entry point is `align`, not `fit`. A caller transforming new embeddings
must reproduce `align`'s centering and normalization itself, since only
`X @ W` runs in `call`. All heavy work runs on CPU through scikit-learn and
scipy. `get_config` carries only `embedding_dim`; the fitted matrix travels
in the saved file's weights.

References:
    - mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations,
      2025. (https://arxiv.org/abs/2510.02348)
    - Huh et al., 2024. The Platonic Representation Hypothesis.
      (https://arxiv.org/abs/2405.07987)
    - Jha et al., 2025. Harnessing the Universal Geometry of Embeddings.
    - Schonemann, 1966. A Generalized Solution of the Orthogonal Procrustes Problem.
      Psychometrika 31(1):1-10.
    - Conneau et al., 2017. Word Translation Without Parallel Data.
      (https://arxiv.org/abs/1710.04087)
    - Artetxe et al., 2018. A Robust Self-Learning Method for Fully Unsupervised
      Cross-Lingual Mappings of Word Embeddings. ACL 2018.
    - Moschella et al., 2022. Relative Representations Enable Zero-Shot Latent Space
      Communication. (https://arxiv.org/abs/2209.15430)
"""

import keras
from keras import ops, initializers
from typing import Optional, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import quadratic_assignment

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

@register_dl_technique("dl_techniques.models.mini_vec2vec.model")
class MiniVec2VecAligner(keras.Model):
    """
    Keras implementation of the mini-vec2vec unsupervised alignment algorithm.

    This model learns a linear transformation to align two embedding spaces
    (A and B) without parallel data. Alignment runs in three stages: build
    pseudo-parallel pairs by clustering both spaces and matching centroids
    with the quadratic assignment problem, fit an initial orthogonal
    transform `W` from those pairs by Procrustes analysis, then refine `W`
    with a matching-based round and a clustering-based round.

    Architecture:

    .. code-block:: text

        Input (Space A)
               ↓
        Linear Transform: X_A @ W
               ↓
        Output (Aligned to Space B)

    :param embedding_dim: Dimensionality of the embedding spaces to align.
        Sets the size of the transformation matrix W. Must be positive.
    :type embedding_dim: int
    :param kwargs: Additional arguments for the keras.Model base class.

    :ivar W: The transformation matrix, shape `(embedding_dim, embedding_dim)`.
    :vartype W: keras.Variable

    Example:
        >>> # Create aligner. `align` builds it; an explicit build() is only
        >>> # needed to read `W` before fitting.
        >>> aligner = MiniVec2VecAligner(embedding_dim=128)
        >>>
        >>> # Align two embedding spaces
        >>> history = aligner.align(
        ...     XA=source_embeddings,  # shape: (n_samples_A, 128)
        ...     XB=target_embeddings,  # shape: (n_samples_B, 128)
        ...     approx_clusters=20,
        ...     approx_runs=30,
        ...     refine1_iterations=50
        ... )
        >>>
        >>> # Transform new embeddings IN THE FITTED FRAME: `align` centered
        >>> # and L2-normalized both spaces and did not keep the means, so
        >>> # reproduce them from the alignment set (see the module docstring
        >>> # and `example_alignment.align_frame`).
        >>> mean_A = source_embeddings.mean(axis=0, keepdims=True)
        >>> centered = source_embeddings - mean_A
        >>> in_frame = centered / np.linalg.norm(centered, axis=1, keepdims=True)
        >>> aligned_embeddings = aligner(in_frame)
        >>>
        >>> # Save model
        >>> aligner.save('mini_vec2vec_aligner.keras')

    Note:
        The `align` method is used to fit the transformation matrix W, not
        the standard Keras `fit` method. This is because the alignment
        procedure follows a specific algorithmic approach rather than
        gradient-based optimization.
    """

    def __init__(self, embedding_dim: int, **kwargs: Any) -> None:
        """
        Initialize the MiniVec2VecAligner model.

        :param embedding_dim: Dimensionality of the embedding spaces.
        :param kwargs: Additional arguments for keras.Model.
        :raises ValueError: If embedding_dim is not positive.
        """
        super().__init__(**kwargs)

        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {embedding_dim}"
            )

        # Store configuration
        self.embedding_dim = embedding_dim

        # Transformation matrix will be created in build()
        self.W = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the transformation matrix W.

        :param input_shape: Shape of input tensor; last dimension must equal
            embedding_dim.
        :raises ValueError: If input shape's last dimension doesn't match
            embedding_dim.
        """
        if input_shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Input shape's last dimension ({input_shape[-1]}) must match "
                f"the embedding_dim ({self.embedding_dim}) provided at initialization."
            )

        # Create transformation matrix initialized to identity
        self.W = self.add_weight(
            name="transformation_matrix_W",
            shape=(self.embedding_dim, self.embedding_dim),
            initializer=initializers.Identity(),
            trainable=True,
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply the learned linear transformation W to input embeddings.

        :param inputs: Input embeddings, shape `(batch_size, embedding_dim)`.
        :param training: Unused; present for API consistency.
        :return: Transformed embeddings, shape `(batch_size, embedding_dim)`.
        """
        return ops.matmul(inputs, self.W)

    def _procrustes(
            self,
            XA: np.ndarray,
            XB: np.ndarray
    ) -> np.ndarray:
        """
        Compute the optimal orthogonal transformation using Procrustes analysis.

        Finds the orthogonal matrix W that minimizes ||XA @ W - XB||_F.

        Args:
            XA: Source embeddings, shape `(n_samples, embedding_dim)`.
            XB: Target embeddings, shape `(n_samples, embedding_dim)`.

        Returns:
            Optimal orthogonal transformation matrix W.
        """
        # Compute cross-covariance matrix
        A_T_B = np.dot(XA.T, XB)

        # SVD decomposition
        U, _, Vt = np.linalg.svd(A_T_B)

        # Optimal orthogonal transformation
        return np.dot(U, Vt)

    def _create_pseudo_pairs(
            self,
            XA: np.ndarray,
            XB: np.ndarray,
            num_clusters: int,
            num_runs: int,
            num_neighbors: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create pseudo-parallel pairs using noisy anchor alignment (Algorithm 2).

        This method implements the anchor-based matching strategy that uses
        clustering and the Quadratic Assignment Problem to create pseudo-pairs
        without requiring true parallel data.

        Args:
            XA: Source embeddings, shape `(n_samples_A, embedding_dim)`.
            XB: Target embeddings, shape `(n_samples_B, embedding_dim)`.
            num_clusters: Number of clusters for anchor generation.
            num_runs: Number of runs for ensemble creation.
            num_neighbors: Number of neighbors to average for pseudo-targets.

        Returns:
            Tuple of (source_pairs, target_pairs) where both are numpy arrays
            of shape `(n_samples_A, embedding_dim)`.
        """
        logger.info(f"Creating pseudo-pairs with {num_runs} runs of anchor alignment...")
        all_relative_A = []
        all_relative_B = []

        for _ in tqdm(range(num_runs), desc="Anchor Alignment Runs"):
            # Step 1: Cluster both spaces independently
            kmeans_A = KMeans(
                n_clusters=num_clusters,
                n_init="auto",
                random_state=None
            ).fit(XA)
            kmeans_B = KMeans(
                n_clusters=num_clusters,
                n_init="auto",
                random_state=None
            ).fit(XB)

            centroids_A = kmeans_A.cluster_centers_
            centroids_B = kmeans_B.cluster_centers_

            # Step 2: Compute cosine similarity matrices between centroids
            sim_A = centroids_A @ centroids_A.T
            sim_B = centroids_B @ centroids_B.T

            # Step 3: Find correspondence with QAP.
            # We maximize Tr(P @ sim_A @ P.T @ sim_B) by setting A=-sim_A.
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-050: method='faq', not
            # '2opt' — '2opt' returned a worse objective on 6 of 12 exactly
            # solvable test instances. See decisions.md.
            res = quadratic_assignment(
                -sim_A,
                sim_B,
                method='faq',
            )
            permutation_indices = res.col_ind

            # Reorder B's centroids to match A's
            aligned_centroids_B = centroids_B[permutation_indices]

            # Step 4: Build relative representations (anchor-based features)
            relative_A = XA @ centroids_A.T
            relative_B = XB @ aligned_centroids_B.T
            all_relative_A.append(relative_A)
            all_relative_B.append(relative_B)

        # Step 5: Concatenate relative representations from all runs
        concat_relative_A = np.concatenate(all_relative_A, axis=1)
        concat_relative_B = np.concatenate(all_relative_B, axis=1)

        # Step 6: Match embeddings using nearest neighbors and create pseudo-pairs
        nn = NearestNeighbors(
            n_neighbors=num_neighbors,
            metric='cosine',
            n_jobs=-1
        )
        nn.fit(concat_relative_B)
        distances, indices = nn.kneighbors(concat_relative_A)

        # Average neighbors to get robust pseudo-targets
        matched_XB = XB[indices].mean(axis=1)

        return XA, matched_XB

    def _refine_matching_based(
            self,
            XA: np.ndarray,
            XB: np.ndarray,
            iterations: int,
            sample_size: int,
            num_neighbors: int,
            smoothing: float,
    ) -> None:
        """
        Apply matching-based refinement (Algorithm 3 / Refine-1).

        Iteratively refines the transformation matrix by:
        1. Sampling source embeddings
        2. Transforming them
        3. Finding nearest neighbors in target space
        4. Re-estimating transformation
        5. Smoothly updating W

        Args:
            XA: Source embeddings, shape `(n_samples_A, embedding_dim)`.
            XB: Target embeddings, shape `(n_samples_B, embedding_dim)`.
            iterations: Number of refinement iterations.
            sample_size: Number of samples per iteration.
            num_neighbors: Number of neighbors to average.
            smoothing: Exponential smoothing factor (0 < smoothing <= 1).
        """
        logger.info("Starting Refine-1: Matching-Based Refinement...")
        current_W = ops.convert_to_numpy(self.W)

        for i in tqdm(range(iterations), desc="Refine-1 Iterations"):
            # Step 1: Sample from source embeddings
            sample_indices = np.random.choice(
                XA.shape[0],
                size=min(sample_size, XA.shape[0]),
                replace=False
            )
            X_sample = XA[sample_indices]

            # Step 2: Transform samples with current W
            X_transformed = X_sample @ current_W

            # Step 3: Find nearest neighbors in target space
            nn = NearestNeighbors(
                n_neighbors=num_neighbors,
                metric='cosine',
                n_jobs=-1
            )
            nn.fit(XB)
            _, indices = nn.kneighbors(X_transformed)

            # Step 4: Average neighbors to create pseudo-targets
            X_matched = XB[indices].mean(axis=1)

            # Step 5: Estimate new mapping using Procrustes
            W_new = self._procrustes(X_sample, X_matched)

            # Step 6: Update with exponential smoothing
            current_W = (1 - smoothing) * current_W + smoothing * W_new

        # Update model weight
        self.W.assign(current_W)

    def _refine_clustering_based(
            self,
            XA: np.ndarray,
            XB: np.ndarray,
            num_clusters: int,
            smoothing: float,
    ) -> None:
        """
        Apply clustering-based refinement (Algorithm 4 / Refine-2).

        Refines the transformation by:
        1. Clustering source space
        2. Transforming centroids
        3. Using transformed centroids as seeds for clustering target space
        4. Matching centroid pairs
        5. Re-estimating transformation

        Args:
            XA: Source embeddings, shape `(n_samples_A, embedding_dim)`.
            XB: Target embeddings, shape `(n_samples_B, embedding_dim)`.
            num_clusters: Number of clusters.
            smoothing: Exponential smoothing factor (0 < smoothing <= 1).
        """
        logger.info("Starting Refine-2: Clustering-Based Refinement...")
        current_W = ops.convert_to_numpy(self.W)

        # Step 1: Cluster source space A
        kmeans_A = KMeans(
            n_clusters=num_clusters,
            n_init="auto",
            random_state=None
        ).fit(XA)
        centroids_A = kmeans_A.cluster_centers_

        # Step 2: Transform source centroids
        transformed_centroids_A = centroids_A @ current_W

        # Step 3: Cluster target space B using transformed centroids as initialization
        kmeans_B = KMeans(
            n_clusters=num_clusters,
            init=transformed_centroids_A,
            n_init=1
        ).fit(XB)
        centroids_B = kmeans_B.cluster_centers_

        # Step 4: Estimate new mapping from matched centroid pairs
        W_new = self._procrustes(centroids_A, centroids_B)

        # Step 5: Update with exponential smoothing
        final_W = (1 - smoothing) * current_W + smoothing * W_new
        self.W.assign(final_W)

    def align(
            self,
            XA: np.ndarray,
            XB: np.ndarray,
            # Params for Approximate Matching (Algorithm 2)
            approx_clusters: int = 20,
            approx_runs: int = 30,
            approx_neighbors: int = 50,
            # Params for Refine-1 (Algorithm 3)
            refine1_iterations: int = 75,
            refine1_sample_size: int = 10000,
            refine1_neighbors: int = 50,
            # Params for Refine-2 (Algorithm 4)
            refine2_clusters: int = 500,
            # General params
            smoothing_alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Execute the full mini-vec2vec alignment pipeline (Algorithm 1).

        Runs all three stages: approximate matching, initial mapping
        estimation, and iterative refinement.

        :param XA: Source embeddings, shape `(n_samples_A, embedding_dim)`.
        :param XB: Target embeddings, shape `(n_samples_B, embedding_dim)`.
        :param approx_clusters: Number of clusters for anchor alignment.
        :param approx_runs: Number of ensemble runs for anchor alignment.
        :param approx_neighbors: Number of neighbors averaged for pseudo-pairs.
        :param refine1_iterations: Number of matching-based refinement iterations.
        :param refine1_sample_size: Number of samples per Refine-1 iteration.
        :param refine1_neighbors: Number of neighbors for matching in Refine-1.
        :param refine2_clusters: Number of clusters for clustering-based
            refinement; should exceed approx_clusters.
        :param smoothing_alpha: Exponential smoothing factor for updating W,
            in (0, 1]. Closer to 1 weights new estimates more.
        :return: Dictionary with W's history: 'initial_W', 'refine1_W', 'final_W'.
        :raises ValueError: If input arrays have incompatible shapes or
            hyperparameters are invalid.

        Note:
            Modifies the model's W weight in place. Embeddings are
            centered and normalized within this method.
        """
        # Validate inputs
        if XA.shape[1] != self.embedding_dim or XB.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Input embeddings must have shape (*, {self.embedding_dim}), "
                f"got XA: {XA.shape}, XB: {XB.shape}"
            )

        if not (0 < smoothing_alpha <= 1):
            raise ValueError(
                f"smoothing_alpha must be in (0, 1], got {smoothing_alpha}"
            )

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-049: align builds the model
        # itself — the shape is fully determined by embedding_dim, so there is
        # nothing for a caller to decide. Do not replace with a raise. See decisions.md.
        if not self.built:
            self.build((None, self.embedding_dim))

        # Ensure input is numpy for sklearn/scipy compatibility
        XA = ops.convert_to_numpy(XA)
        XB = ops.convert_to_numpy(XB)

        history = {}

        # ===== Stage 1: Preprocessing =====
        logger.info("Step 1: Preprocessing embeddings...")
        mean_A = XA.mean(axis=0, keepdims=True)
        mean_B = XB.mean(axis=0, keepdims=True)
        XA_proc = XA - mean_A
        XB_proc = XB - mean_B

        # Normalize to unit sphere
        XA_proc = XA_proc / np.linalg.norm(XA_proc, axis=1, keepdims=True)
        XB_proc = XB_proc / np.linalg.norm(XB_proc, axis=1, keepdims=True)

        # ===== Stage 2: Approximate Matching =====
        logger.info("\nStep 2: Approximate Matching...")
        source_pairs, target_pairs = self._create_pseudo_pairs(
            XA_proc, XB_proc, approx_clusters, approx_runs, approx_neighbors
        )

        # ===== Stage 3: Estimate Initial Mapping =====
        logger.info("\nStep 3: Estimating initial transformation...")
        initial_W = self._procrustes(source_pairs, target_pairs)
        self.W.assign(initial_W)
        history["initial_W"] = initial_W
        logger.info("Initial mapping estimated.")

        # ===== Stage 4: Refine-1 (Matching-Based) =====
        logger.info("\nStep 4: Applying Matching-Based Refinement (Refine-1)...")
        self._refine_matching_based(
            XA_proc,
            XB_proc,
            iterations=refine1_iterations,
            sample_size=refine1_sample_size,
            num_neighbors=refine1_neighbors,
            smoothing=smoothing_alpha,
        )
        history["refine1_W"] = ops.convert_to_numpy(self.W)
        logger.info("Refine-1 complete.")

        # ===== Stage 5: Refine-2 (Clustering-Based) =====
        logger.info("\nStep 5: Applying Clustering-Based Refinement (Refine-2)...")
        self._refine_clustering_based(
            XA_proc,
            XB_proc,
            num_clusters=refine2_clusters,
            smoothing=smoothing_alpha,
        )
        history["final_W"] = ops.convert_to_numpy(self.W)
        logger.info("Refine-2 complete.")

        logger.info("\n✓ Alignment finished successfully!")
        return history

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        :return: Dictionary containing the model configuration.
        """
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim
        })
        return config


def create_mini_vec2vec_aligner(
        embedding_dim: int,
        **kwargs: Any
) -> MiniVec2VecAligner:
    """
    Factory function to create a MiniVec2VecAligner model.

    :param embedding_dim: Dimensionality of the embedding spaces.
    :param kwargs: Additional arguments for MiniVec2VecAligner.
    :return: Initialized MiniVec2VecAligner model.

    Example:
        >>> aligner = create_mini_vec2vec_aligner(embedding_dim=128)
        >>> aligner.build(input_shape=(None, 128))
    """
    return MiniVec2VecAligner(embedding_dim=embedding_dim, **kwargs)