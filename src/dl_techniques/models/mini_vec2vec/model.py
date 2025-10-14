"""
Mini-Vec2Vec Alignment Model for Unsupervised Embedding Space Alignment.

This module implements the mini-vec2vec algorithm for aligning two embedding
spaces without parallel data, following the procedure described in
"mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations".
"""

import keras
from keras import ops, initializers
from typing import Optional, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import quadratic_assignment


@keras.saving.register_keras_serializable(package="dl_techniques.models.mini_vec2vec")
class MiniVec2VecAligner(keras.Model):
    """
    Keras implementation of the mini-vec2vec unsupervised alignment algorithm.

    This model learns a linear transformation to align two embedding spaces (A and B)
    without access to parallel data, following the procedure described in
    "mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations".

    The alignment is achieved through a three-stage process:

    1. **Approximate Matching**: Creates pseudo-parallel pairs of embeddings using
       a robust anchor-based method involving clustering and the Quadratic
       Assignment Problem (QAP).
    2. **Mapping Estimation**: Learns an initial orthogonal transformation (W)
       from these pseudo-pairs using Procrustes analysis.
    3. **Iterative Refinement**: Refines the transformation matrix W using two
       complementary strategies: matching-based and clustering-based refinement.

    **Intent**: Provide a robust, efficient, and Keras-native implementation for
    unsupervised embedding space alignment. The model's primary weight is the
    transformation matrix `W`.

    **Architecture**:

    .. code-block:: text

        Input (Space A)
               ↓
        Linear Transform: X_A @ W
               ↓
        Output (Aligned to Space B)

    Args:
        embedding_dim: Integer, the dimensionality of the embedding spaces to be
            aligned. This determines the size of the transformation matrix W.
            Must be positive.
        **kwargs: Additional arguments for the keras.Model base class.

    Attributes:
        W: keras.Variable, the transformation matrix of shape
            `(embedding_dim, embedding_dim)`. This is the core learnable weight
            of the model.

    Example:
        >>> # Create aligner
        >>> aligner = MiniVec2VecAligner(embedding_dim=128)
        >>>
        >>> # Build model
        >>> aligner.build(input_shape=(None, 128))
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
        >>> # Transform new embeddings
        >>> aligned_embeddings = aligner(source_embeddings)
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

        Args:
            embedding_dim: Dimensionality of the embedding spaces.
            **kwargs: Additional arguments for keras.Model.

        Raises:
            ValueError: If embedding_dim is not positive.
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

        Args:
            input_shape: Shape of input tensor, must have last dimension
                equal to embedding_dim.

        Raises:
            ValueError: If input shape's last dimension doesn't match
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

        Args:
            inputs: Input embeddings of shape `(batch_size, embedding_dim)`.
            training: Boolean or None, whether the call is in training mode.
                Not used in this model but included for API consistency.

        Returns:
            Transformed embeddings of shape `(batch_size, embedding_dim)`.
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
        print(f"Creating pseudo-pairs with {num_runs} runs of anchor alignment...")
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

            # Step 3: Find correspondence with QAP (using 2-OPT)
            # We maximize Tr(P @ sim_A @ P.T @ sim_B) by setting A=-sim_A
            res = quadratic_assignment(
                -sim_A,
                sim_B,
                method='2opt',
                options={'rng': None}
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
        print("Starting Refine-1: Matching-Based Refinement...")
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
        print("Starting Refine-2: Clustering-Based Refinement...")
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

        This method orchestrates all three stages of the alignment procedure:
        approximate matching, initial mapping estimation, and iterative refinement.

        Args:
            XA: Source embeddings, shape `(n_samples_A, embedding_dim)`.
            XB: Target embeddings, shape `(n_samples_B, embedding_dim)`.
            approx_clusters: Number of clusters for anchor alignment. Higher values
                may improve alignment quality but increase computation time.
            approx_runs: Number of runs for ensembling in anchor alignment. More
                runs provide robustness to clustering randomness.
            approx_neighbors: Number of neighbors to average for pseudo-pairs.
                Higher values create more robust but less precise matches.
            refine1_iterations: Number of iterations for matching-based refinement.
                More iterations allow finer adjustments but increase runtime.
            refine1_sample_size: Number of samples per Refine-1 iteration.
                Larger samples improve stability but increase per-iteration cost.
            refine1_neighbors: Number of neighbors for matching in Refine-1.
            refine2_clusters: Number of clusters for clustering-based refinement.
                Should be larger than approx_clusters for fine-grained adjustment.
            smoothing_alpha: Exponential smoothing factor for updating W.
                Values closer to 1 give more weight to new estimates,
                closer to 0 preserve previous estimates. Range: (0, 1].

        Returns:
            Dictionary containing the history of the transformation matrix W
            at different stages: 'initial_W', 'refine1_W', 'final_W'.

        Raises:
            ValueError: If input arrays have incompatible shapes or if
                hyperparameters are invalid.

        Note:
            This method modifies the model's W weight in-place. The embeddings
            should be preprocessed (centered and normalized) within this method.
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

        # Ensure input is numpy for sklearn/scipy compatibility
        XA = ops.convert_to_numpy(XA)
        XB = ops.convert_to_numpy(XB)

        history = {}

        # ===== Stage 1: Preprocessing =====
        print("Step 1: Preprocessing embeddings...")
        mean_A = XA.mean(axis=0, keepdims=True)
        mean_B = XB.mean(axis=0, keepdims=True)
        XA_proc = XA - mean_A
        XB_proc = XB - mean_B

        # Normalize to unit sphere
        XA_proc = XA_proc / np.linalg.norm(XA_proc, axis=1, keepdims=True)
        XB_proc = XB_proc / np.linalg.norm(XB_proc, axis=1, keepdims=True)

        # ===== Stage 2: Approximate Matching =====
        print("\nStep 2: Approximate Matching...")
        source_pairs, target_pairs = self._create_pseudo_pairs(
            XA_proc, XB_proc, approx_clusters, approx_runs, approx_neighbors
        )

        # ===== Stage 3: Estimate Initial Mapping =====
        print("\nStep 3: Estimating initial transformation...")
        initial_W = self._procrustes(source_pairs, target_pairs)
        self.W.assign(initial_W)
        history["initial_W"] = initial_W
        print("Initial mapping estimated.")

        # ===== Stage 4: Refine-1 (Matching-Based) =====
        print("\nStep 4: Applying Matching-Based Refinement (Refine-1)...")
        self._refine_matching_based(
            XA_proc,
            XB_proc,
            iterations=refine1_iterations,
            sample_size=refine1_sample_size,
            num_neighbors=refine1_neighbors,
            smoothing=smoothing_alpha,
        )
        history["refine1_W"] = ops.convert_to_numpy(self.W)
        print("Refine-1 complete.")

        # ===== Stage 5: Refine-2 (Clustering-Based) =====
        print("\nStep 5: Applying Clustering-Based Refinement (Refine-2)...")
        self._refine_clustering_based(
            XA_proc,
            XB_proc,
            num_clusters=refine2_clusters,
            smoothing=smoothing_alpha,
        )
        history["final_W"] = ops.convert_to_numpy(self.W)
        print("Refine-2 complete.")

        print("\n✓ Alignment finished successfully!")
        return history

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        Returns:
            Dictionary containing the model configuration.
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

    Args:
        embedding_dim: Dimensionality of the embedding spaces.
        **kwargs: Additional arguments for MiniVec2VecAligner.

    Returns:
        Initialized MiniVec2VecAligner model.

    Example:
        >>> aligner = create_mini_vec2vec_aligner(embedding_dim=128)
        >>> aligner.build(input_shape=(None, 128))
    """
    return MiniVec2VecAligner(embedding_dim=embedding_dim, **kwargs)