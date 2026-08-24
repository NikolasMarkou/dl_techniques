"""
Unsupervised alignment of two embedding spaces by a single square linear map, fitted
with clustering, quadratic assignment and Procrustes rather than by gradient descent.

The premise is the universal-geometry, or Platonic representation, hypothesis:
independently trained encoders of the same underlying data arrive at nearly the same
*relative* geometry, differing mainly in the arbitrary coordinate frame they express it
in. If that holds, the map from space A to space B needs no capacity beyond a rotation
and reflection, and the entire difficulty moves from function fitting to correspondence:
knowing which point of A is which point of B. Every stage below exists to manufacture a
correspondence, because once one exists the map is closed-form. Orthogonal Procrustes
solves `min_W ||X_A W - X_B||_F` subject to `W^T W = I` by `W = U V^T` where
`U S V^T = SVD(X_A^T X_B)` - no learning rate, no adversarial game, no convergence
question.

`align` runs five stages over numpy arrays; nothing here is a gradient step.

Preprocessing mean-centers each space over the arrays it is given and then L2-normalizes
every row onto the unit sphere. Note what is *not* stored: the two mean vectors are
local to `align`, and `call` applies only `X @ W`. A caller transforming new embeddings
must reproduce the same centering and normalization themselves, or the map is being
applied in a frame it was not fitted in.

Approximate matching produces the first correspondence. Each of `approx_runs` (30)
rounds clusters both spaces independently into `approx_clusters` (20) centroids, then
matches the two centroid sets using only frame-invariant information: the centroid Gram
matrices `C_A C_A^T` and `C_B C_B^T` are unchanged by any orthogonal transform of their
space, so the permutation that makes them agree is recoverable without knowing the
transform. That is a quadratic assignment problem, solved by scipy's 2-opt heuristic;
the first argument is negated because `quadratic_assignment` minimizes, and minimizing
against `-sim_A` maximizes `tr(P sim_A P^T sim_B)`. With B's centroids permuted into A's
order, every embedding is re-expressed by its similarities to its own anchor set
(`X_A C_A^T`, `X_B C_B_perm^T`), coordinates that by construction mean the same thing in
both spaces. One round of k-means plus a heuristic QAP is far too noisy to trust, so the
rounds are concatenated along the feature axis: the ensemble of 30 imperfect anchor sets
is a much more discriminative descriptor than any single one. Pseudo-pairs then come
from a cosine nearest-neighbour search in that descriptor space, and each source row's
pseudo-target is the *mean* of its `approx_neighbors` (50) neighbours in B, trading
precision for immunity to individual mismatches. Every source row is paired; nothing is
filtered on match quality, and the neighbour distances are discarded.

Refinement then alternates correspondence and map, the self-learning loop that makes
unsupervised bilingual dictionary induction work. Refine-1 repeats `refine1_iterations`
(75) times: sample rows from A, push them through the current `W`, take the mean of
their `refine1_neighbors` cosine neighbours in B as targets, re-solve Procrustes, and
blend. Because neighbours are recomputed under the improved map each iteration, the
correspondence and the map bootstrap each other. Refine-2 runs once and gets its
correspondence for free rather than by search: cluster A into `refine2_clusters` (500)
centroids, push them through `W`, and use the result as the *initialization* of a
single-restart k-means on B (`n_init=1`). The i-th B centroid is then by definition the
one seeded from the i-th A centroid, so the 500 centroid pairs are matched with no
assignment step at all. The progression is deliberate: 20 coarse anchors to establish
the frame, 500 fine ones to sharpen it.

The orthogonality of the result is the subtle point. Every Procrustes solve returns an
exactly orthogonal matrix, but each update is an exponential-smoothing blend
`W <- (1 - alpha) W + alpha W_new`, and a convex combination of two orthogonal matrices
is not orthogonal - the orthogonal group is not convex. No re-orthogonalization follows
the blend. The shipped `W` is therefore only approximately orthogonal, and the
approximation is good exactly to the extent the iterates have converged and the two
blended matrices already agree. Treat `W` as a general linear map: in particular do not
assume `W^-1 = W^T` when mapping back from B to A. The trade is what buys stability,
since undamped updates would let one bad neighbour draw discard the whole map.

`W` is registered through `add_weight` with an identity initializer and is nominally
trainable, but it is only ever written by `assign` from numpy; `fit` is not the entry
point and `align` is. All heavy work runs on CPU through scikit-learn and scipy, so
sequence length and dimensionality drive the cost, not accelerator memory. `align`
returns the matrix at three checkpoints (`initial_W`, `refine1_W`, `final_W`), which is
the practical way to see whether refinement helped or drifted. `get_config` carries only
`embedding_dim`; the fitted matrix travels in the weights of the saved file.

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

@keras.saving.register_keras_serializable()
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
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-050: `method='faq'`,
            # NOT `'2opt'`. MEASURED on EXACTLY solvable instances (sim_B is a
            # true permutation of sim_A, so the optimum is known and reachable):
            # over 12 instances at k in {5, 8, 12, 20}, '2opt' recovered the
            # permutation in 6 and returned a strictly WORSE objective in the
            # other 6 — at k = 20, this package's own `approx_clusters` default,
            # it failed 2 of 3 (objective 25.97 vs the optimal 30.72). 'faq'
            # was exact in 12 of 12. With the anchor permutation wrong the
            # relative representations of the two spaces are not comparable,
            # every pseudo-pair is noise, and the whole pipeline returns a map
            # no better than chance. Do not revert to '2opt' without re-running
            # that comparison.
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

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-049: `align` builds the
        # model itself. Every stage below writes through `self.W.assign(...)`,
        # so on a FRESH aligner — the state both this method's and the class's
        # docstring examples start from — it used to die with
        # `AttributeError: 'NoneType' object has no attribute 'assign'`, from
        # inside stage 3, after minutes of k-means and QAP work. The shape is
        # fully determined by `embedding_dim`, which is a constructor argument,
        # so there is nothing to infer and nothing for the caller to decide.
        # Do NOT replace this with a raise: `build` takes no information the
        # object does not already have.
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