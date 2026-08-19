"""
Latent-GMM Point Cloud Registration model.

ACCEPTED RAW-TF EXCEPTION (production-map §L2-5 / H10):
    The weighted-Procrustes rotation solver in the forward path uses
    ``tf.linalg.svd`` / ``tf.linalg.det`` / ``tf.linalg.diag`` to recover the
    optimal rigid rotation from the cross-covariance matrix. This cannot migrate
    to ``keras.ops``: there is no ``keras.ops.svd`` (nor a backend-agnostic
    determinant/diag-construct path suitable here), so the SVD-based closed-form
    rotation is not expressible without the raw ``tf.linalg`` ops. The raw-TF
    linear-algebra path is therefore an accepted, documented exception to the
    keras.ops-only (H10) rule for the forward pass.

FLOAT32 ONLY -- this model does NOT run under ``mixed_float16``.
    MEASURED (2026-08-19, TF 2.18 / Keras 3.8, RTX 4070 and CPU): a single
    forward pass under ``keras.mixed_precision.set_global_policy(
    "mixed_float16")`` raises, inside ``compute_rigid_transform``::

        NotFoundError: Could not find device for node:
        {{node Svd}} = Svd[T=DT_HALF, compute_uv=true, full_matrices=false]

    This is NOT a dtype-plumbing bug in this package. TensorFlow registers NO
    ``Svd`` kernel for ``DT_HALF`` on ``CPU`` or ``GPU`` -- the op's kernel list
    is ``{CPU: float, double, complex64, complex128}`` and
    ``{GPU: double, float}``; only the XLA JIT devices accept ``DT_HALF``. Half
    precision is also the wrong arithmetic for this step on its own merits: the
    weighted-Procrustes solve orthogonalizes a 3x3 cross-covariance and its
    ``det``-based reflection correction switches on the SIGN of a quantity that
    is near zero exactly when the rotation is near-degenerate.

    Train and infer this model under the default ``float32`` policy. See
    plan ``plan-2026-08-19T163559-499b6f0e`` decisions.md D-011.
"""

import keras
import tensorflow as tf
from typing import Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.losses.chamfer_loss import ChamferLoss
from dl_techniques.layers.geometric.point_cloud_autoencoder import (
    PointCloudAutoencoder, CorrespondenceNetwork)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class LatentGMMRegistration(keras.Model):
    """Robust Semi-Supervised Point Cloud Registration via Latent GMM.

    This model implements the complete architecture from the paper, combining a
    feature-learning autoencoder with a GMM-based correspondence network to
    estimate the rigid transformation between two point clouds.

    **Intent**: To provide an end-to-end, learning-based solution for point
    cloud registration that is robust to noise and large transformations.

    **Architecture**:
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │  Input: (Source PC, Target PC)                               │
    │         Shape: (B, N, 3) each                                │
    └────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  PointCloudAutoencoder                                       │
    │  ├─ Reconstructions: x_rec, y_rec  (B, N, 3)                 │
    │  ├─ Local Features: local_x, local_y  (B, N, F_local)        │
    │  └─ Global Features: global_x, global_y  (B, F_global)       │
    └────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  CorrespondenceNetwork (shared weights)                      │
    │  Inputs: (local_features, global_features)                   │
    │  Outputs: gamma_x, gamma_y  (B, N, K)                        │
    │  where K = num_gaussians (soft assignments)                  │
    └────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  GMM Parameter Estimation (non-trainable ops)                │
    │  ├─ Mixing coefficients: pi_x, pi_y  (B, K)                  │
    │  └─ Component means: mu_x, mu_y  (B, K, 3)                   │
    └────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Rigid Transform Estimation (weighted Procrustes)            │
    │  ├─ Rotation: R  (B, 3, 3)                                   │
    │  └─ Translation: t  (B, 3)                                   │
    └────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  Output Dictionary                                           │
    │  ├─ reconstruction_x: (B, N, 3)                              │
    │  ├─ reconstruction_y: (B, N, 3)                              │
    │  ├─ estimated_r: (B, 3, 3)                                   │
    │  └─ estimated_t: (B, 3)                                      │
    └──────────────────────────────────────────────────────────────┘

    Legend: B=batch_size, N=num_points, K=num_gaussians, F=feature_dim
    ```

    **Key Components**:
    - **PointCloudAutoencoder**: Extracts local and global features from point clouds
    - **CorrespondenceNetwork**: Computes soft assignments to GMM components
    - **GMM Parameter Estimation**: Non-trainable operations for computing GMM statistics
    - **Rigid Transform Estimation**: Closed-form solution for optimal transformation

    Args:
        num_gaussians: Number of latent GMM components. Must be positive.
            Determines the expressiveness of the latent correspondence space.
        k_neighbors: Number of neighbors for feature extraction. Must be positive.
            Controls the receptive field of local feature computation.
        chamfer_weight: Weight for the Chamfer reconstruction loss. Default 1.0.
            Balances reconstruction quality vs transformation accuracy.
        transform_weight: Weight for the transformation loss. Default 1.0.
            Balances transformation accuracy vs reconstruction quality.
        **kwargs: Additional arguments for Model base class.

    Examples:
        >>> model = LatentGMMRegistration(
        ...     num_gaussians=32,
        ...     k_neighbors=16,
        ...     chamfer_weight=1.0,
        ...     transform_weight=0.5
        ... )
        >>> source = keras.random.normal((8, 1024, 3))
        >>> target = keras.random.normal((8, 1024, 3))
        >>> outputs = model((source, target))
        >>> R_est = outputs["estimated_r"]  # Shape: (8, 3, 3)
        >>> t_est = outputs["estimated_t"]  # Shape: (8, 3)
    """

    def __init__(
            self,
            num_gaussians: int,
            k_neighbors: int,
            chamfer_weight: float = 1.0,
            transform_weight: float = 1.0,
            **kwargs: Any
    ) -> None:
        """Initialize LatentGMMRegistration model.

        Args:
            num_gaussians: Number of latent GMM components.
            k_neighbors: Number of neighbors for feature extraction.
            chamfer_weight: Weight for the Chamfer reconstruction loss.
            transform_weight: Weight for the transformation loss.
            **kwargs: Additional arguments for Model base class.

        Raises:
            ValueError: If num_gaussians or k_neighbors are not positive.
        """
        super().__init__(**kwargs)

        if num_gaussians <= 0:
            raise ValueError(f"num_gaussians must be positive, got {num_gaussians}")
        if k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
        if chamfer_weight < 0:
            raise ValueError(f"chamfer_weight must be non-negative, got {chamfer_weight}")
        if transform_weight < 0:
            raise ValueError(f"transform_weight must be non-negative, got {transform_weight}")

        self.num_gaussians = num_gaussians
        self.k_neighbors = k_neighbors
        self.chamfer_weight = chamfer_weight
        self.transform_weight = transform_weight

        self.autoencoder = PointCloudAutoencoder(
            k_neighbors=k_neighbors,
            name="autoencoder"
        )
        self.correspondence_net = CorrespondenceNetwork(
            num_gaussians=num_gaussians,
            name="correspondence_net"
        )

        self.chamfer_loss_fn = ChamferLoss(
            reduction="sum_over_batch_size",
            name="chamfer_loss"
        )

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: bool = False
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of the model.

        Args:
            inputs: Tuple of (source_pc, target_pc) point clouds.
                Each point cloud has shape (batch_size, num_points, 3).
            training: Whether in training mode. Affects dropout and batch normalization.

        Returns:
            Dictionary containing:
                - reconstruction_x: Reconstructed source point cloud (batch_size, num_points, 3)
                - reconstruction_y: Reconstructed target point cloud (batch_size, num_points, 3)
                - estimated_r: Estimated rotation matrix (batch_size, 3, 3)
                - estimated_t: Estimated translation vector (batch_size, 3)
        """
        source_pc, target_pc = inputs

        # The autoencoder processes both point clouds simultaneously to extract:
        # - Reconstructions (x_rec, y_rec): Decoded point clouds for Chamfer loss
        # - Local features: Per-point features capturing neighborhood geometry
        # - Global features: Point cloud-level features capturing overall structure
        (x_rec, y_rec), (local_x, local_y), (global_x, global_y) = self.autoencoder(
            (source_pc, target_pc),
            training=training
        )

        # gamma[i,j,k] is the probability that point j belongs to component k.
        # The correspondence network is shared between source and target for consistency
        gamma_x = self.correspondence_net((local_x, global_x), training=training)
        gamma_y = self.correspondence_net((local_y, global_y), training=training)

        # pi/mu are differentiable but carry no trainable parameters.
        pi_x, mu_x = compute_gmm_params(source_pc, gamma_x)
        pi_y, mu_y = compute_gmm_params(target_pc, gamma_y)

        R_est, t_est = compute_rigid_transform(mu_x, pi_x, mu_y, pi_y)

        return {
            "reconstruction_x": x_rec,
            "reconstruction_y": y_rec,
            "estimated_r": R_est,
            "estimated_t": t_est
        }

    def train_step(
            self,
            data: Tuple[Tuple[keras.KerasTensor, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]]
    ) -> Dict[str, keras.KerasTensor]:
        """Custom training step with semi-supervised loss.

        The training combines two complementary objectives:
        1. Unsupervised: Chamfer distance for point cloud reconstruction quality
        2. Supervised: Transformation accuracy when ground truth R,t are available

        Loss = chamfer_weight * L_chamfer + transform_weight * L_transform
        where:
            L_chamfer = Chamfer(source, reconstruction_x) + Chamfer(target, reconstruction_y)
            L_transform = ||I - R_est^T * R_gt||^2 + ||t_est - t_gt||^2

        Args:
            data: Tuple of ((source_pc, target_pc), (R_gt, t_gt)) where:
                - source_pc: Source point cloud (batch_size, num_points, 3)
                - target_pc: Target point cloud (batch_size, num_points, 3)
                - R_gt: Ground truth rotation matrix (batch_size, 3, 3)
                - t_gt: Ground truth translation vector (batch_size, 3)

        Returns:
            Dictionary of loss values and metrics:
                - loss: Total weighted loss
                - chamfer_loss: Reconstruction loss (sum of both point clouds)
                - transform_loss: Transformation estimation loss (rotation + translation)
                - Other compiled metrics
        """
        (source_pc, target_pc), (R_gt, t_gt) = data

        # `keras.backend.GradientTape` does not exist in Keras 3; the tape comes
        # from the backend directly. This module already depends on raw TF for
        # the Procrustes SVD (see the module docstring's accepted exception).
        with tf.GradientTape() as tape:
            y_pred = self((source_pc, target_pc), training=True)

            # Compute Chamfer loss (unsupervised reconstruction)
            loss_chamfer_x = self.chamfer_loss_fn(source_pc, y_pred["reconstruction_x"])
            loss_chamfer_y = self.chamfer_loss_fn(target_pc, y_pred["reconstruction_y"])
            total_chamfer_loss = loss_chamfer_x + loss_chamfer_y

            # Compute transformation loss (supervised)
            R_est, t_est = y_pred["estimated_r"], y_pred["estimated_t"]

            # Rotation loss: ||I - R_est^T * R_gt||_F^2
            # This measures how well R_est aligns with R_gt using the Frobenius norm
            # When R_est = R_gt, we have R_est^T * R_gt = I (identity)
            # The closer to zero, the better the rotation alignment
            loss_r = keras.ops.mean(
                keras.ops.square(
                    keras.ops.eye(3) - keras.ops.matmul(
                        keras.ops.transpose(R_est, (0, 2, 1)),
                        R_gt
                    )
                )
            )

            # Translation loss: ||t_est - t_gt||_2^2 (Mean Squared Error)
            # Direct L2 distance between estimated and ground truth translation vectors
            loss_t = keras.ops.mean(keras.ops.square(t_est - t_gt))

            total_transform_loss = loss_r + loss_t

            # Total weighted loss
            total_loss = (
                    self.chamfer_weight * total_chamfer_loss +
                    self.transform_weight * total_transform_loss
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply(gradients, trainable_vars)

        # `self.compiled_metrics` is a Keras 3 deprecation shim that loops over
        # *every* metric including the loss tracker, so it cannot be handed a
        # structured y/y_pred. `compute_metrics` is the supported entry point and
        # no-ops when `compile(metrics=...)` was not given. y_true mirrors `call`'s
        # output dict; the bare `(R_gt, t_gt)` tuple used before could not even be
        # packed, since (B, 3, 3) and (B, 3) do not stack.
        metric_results = self.compute_metrics(
            x=(source_pc, target_pc),
            y={
                "reconstruction_x": source_pc,
                "reconstruction_y": target_pc,
                "estimated_r": R_gt,
                "estimated_t": t_gt,
            },
            y_pred=y_pred,
        )

        # The explicit loss keys come last: `metric_results` carries the loss
        # tracker's stale value, which would otherwise shadow this step's loss.
        return {
            **metric_results,
            "loss": total_loss,
            "chamfer_loss": total_chamfer_loss,
            "transform_loss": total_transform_loss,
        }

    def test_step(
            self,
            data: Tuple[Tuple[keras.KerasTensor, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]]
    ) -> Dict[str, keras.KerasTensor]:
        """Custom test step with semi-supervised loss evaluation.

        Evaluates the same loss components as training but without gradient computation.
        Useful for validation and testing with ground truth transformations.

        Args:
            data: Tuple of ((source_pc, target_pc), (R_gt, t_gt)) where:
                - source_pc: Source point cloud (batch_size, num_points, 3)
                - target_pc: Target point cloud (batch_size, num_points, 3)
                - R_gt: Ground truth rotation matrix (batch_size, 3, 3)
                - t_gt: Ground truth translation vector (batch_size, 3)

        Returns:
            Dictionary of loss values and metrics:
                - loss: Total weighted loss
                - chamfer_loss: Reconstruction loss (sum of both point clouds)
                - transform_loss: Transformation estimation loss (rotation + translation)
                - Other compiled metrics
        """
        (source_pc, target_pc), (R_gt, t_gt) = data

        y_pred = self((source_pc, target_pc), training=False)

        # Compute Chamfer loss (unsupervised reconstruction)
        loss_chamfer_x = self.chamfer_loss_fn(source_pc, y_pred["reconstruction_x"])
        loss_chamfer_y = self.chamfer_loss_fn(target_pc, y_pred["reconstruction_y"])
        total_chamfer_loss = loss_chamfer_x + loss_chamfer_y

        # Compute transformation loss (supervised)
        R_est, t_est = y_pred["estimated_r"], y_pred["estimated_t"]

        # Rotation loss: ||I - R_est^T * R_gt||_F^2
        # Measures alignment quality between estimated and ground truth rotations
        loss_r = keras.ops.mean(
            keras.ops.square(
                keras.ops.eye(3) - keras.ops.matmul(
                    keras.ops.transpose(R_est, (0, 2, 1)),
                    R_gt
                )
            )
        )

        # Translation loss: ||t_est - t_gt||_2^2
        loss_t = keras.ops.mean(keras.ops.square(t_est - t_gt))

        total_transform_loss = loss_r + loss_t

        # Total weighted loss
        total_loss = (
                self.chamfer_weight * total_chamfer_loss +
                self.transform_weight * total_transform_loss
        )

        # `self.compiled_metrics` is a Keras 3 deprecation shim that loops over
        # *every* metric including the loss tracker, so it cannot be handed a
        # structured y/y_pred. `compute_metrics` is the supported entry point and
        # no-ops when `compile(metrics=...)` was not given. y_true mirrors `call`'s
        # output dict; the bare `(R_gt, t_gt)` tuple used before could not even be
        # packed, since (B, 3, 3) and (B, 3) do not stack.
        metric_results = self.compute_metrics(
            x=(source_pc, target_pc),
            y={
                "reconstruction_x": source_pc,
                "reconstruction_y": target_pc,
                "estimated_r": R_gt,
                "estimated_t": t_gt,
            },
            y_pred=y_pred,
        )

        # The explicit loss keys come last: `metric_results` carries the loss
        # tracker's stale value, which would otherwise shadow this step's loss.
        return {
            **metric_results,
            "loss": total_loss,
            "chamfer_loss": total_chamfer_loss,
            "transform_loss": total_transform_loss,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        Returns:
            Dictionary containing all constructor parameters needed to
            recreate this model instance.
        """
        config = super().get_config()
        config.update({
            'num_gaussians': self.num_gaussians,
            'k_neighbors': self.k_neighbors,
            'chamfer_weight': self.chamfer_weight,
            'transform_weight': self.transform_weight
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LatentGMMRegistration":
        """Create model from configuration.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            New model instance reconstructed from configuration.
        """
        return cls(**config)


def compute_gmm_params(
        points: keras.KerasTensor,
        gamma: keras.KerasTensor
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Compute GMM parameters from soft point-to-component assignments.

    Given a point cloud and soft assignments (responsibilities) from an E-step,
    computes the M-step GMM parameters: mixing coefficients and component means.

    Algorithm:
        pi_k = (1/N) * sum_i gamma_ik  (average responsibility per component)
        mu_k = sum_i (gamma_ik * x_i) / sum_i gamma_ik  (weighted mean)

    Args:
        points: Point cloud of shape (batch_size, num_points, 3).
            The 3D coordinates of each point.
        gamma: Soft assignments of shape (batch_size, num_points, num_gaussians).
            gamma[b,i,k] = responsibility of component k for point i in batch b.
            Each row (over k) should sum to 1 (probability distribution).

    Returns:
        Tuple of (pi, mu) where:
            - pi: Mixing coefficients of shape (batch_size, num_gaussians).
                  Represents the weight/importance of each Gaussian component.
            - mu: Component means of shape (batch_size, num_gaussians, 3).
                  The 3D centroid of each Gaussian component.
    """
    # Mixing coefficients: pi_k = (1/N) * sum_i gamma_ik
    # Average the soft assignments across all points to get component weights
    pi = keras.ops.mean(gamma, axis=1)  # Shape: (batch_size, num_gaussians)

    # Component means: mu_k = sum_i (gamma_ik * x_i) / sum_i gamma_ik
    # Weighted average of points, where weights are the soft assignments

    # Expand dimensions for element-wise multiplication and broadcasting
    gamma_expanded = keras.ops.expand_dims(gamma, axis=-1)  # (B, N, K, 1)
    points_expanded = keras.ops.expand_dims(points, axis=2)  # (B, N, 1, 3)

    # Element-wise multiplication: gamma_ik * x_i for all i,k
    # Then sum over all points (axis=1) to get weighted sum per component
    weighted_sum = keras.ops.sum(
        gamma_expanded * points_expanded,  # (B, N, K, 3)
        axis=1
    )  # Shape: (B, K, 3)

    # Normalize by the TOTAL responsibility sum_i gamma_ik -- not by `pi`, which is
    # that sum divided by N. Dividing by the mean instead of the sum inflated every
    # component mean by exactly N (and, through compute_rigid_transform's centroids,
    # the estimated translation with it).
    # Add epsilon to avoid division by zero for components with negligible weight
    gamma_sum = keras.ops.sum(gamma, axis=1, keepdims=False)  # (B, K)
    gamma_sum_expanded = keras.ops.expand_dims(gamma_sum, axis=-1) + 1e-8  # (B, K, 1)
    mu = weighted_sum / gamma_sum_expanded  # Shape: (B, K, 3)

    return pi, mu


def compute_rigid_transform(
        mu_source: keras.KerasTensor,
        pi_source: keras.KerasTensor,
        mu_target: keras.KerasTensor,
        pi_target: keras.KerasTensor
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Compute optimal rigid transformation between GMM means.

    Uses weighted Procrustes analysis to find the optimal rotation R and translation t
    that minimizes the weighted sum of squared distances between corresponding GMM means:

        min_{R,t} sum_k w_k * ||R * mu_source_k + t - mu_target_k||^2

    where w_k = pi_source_k * pi_target_k (component importance weighting).

    Algorithm:
        1. Compute weighted centroids of both GMMs
        2. Center both sets of means around their centroids
        3. Compute weighted covariance matrix H
        4. Perform SVD: H = U * S * V^T
        5. Compute rotation: R = V * U^T (with reflection correction)
        6. Compute translation: t = centroid_target - R * centroid_source

    Args:
        mu_source: Source GMM means of shape (batch_size, num_gaussians, 3).
            The 3D positions of source Gaussian components.
        pi_source: Source mixing coefficients of shape (batch_size, num_gaussians).
            Weights indicating importance of each source component.
        mu_target: Target GMM means of shape (batch_size, num_gaussians, 3).
            The 3D positions of target Gaussian components.
        pi_target: Target mixing coefficients of shape (batch_size, num_gaussians).
            Weights indicating importance of each target component.

    Returns:
        Tuple of (R, t) where:
            - R: Rotation matrix of shape (batch_size, 3, 3).
                 Orthogonal matrix with det(R) = +1 (proper rotation).
            - t: Translation vector of shape (batch_size, 3).
                 The displacement to align centroids after rotation.
    """
    # w_k = pi_source_k * pi_target_k: joint importance of corresponding components.
    weights = keras.ops.expand_dims(
        pi_source * pi_target,
        axis=-1
    )  # Shape: (B, K, 1)

    # centroid = sum_k (w_k * mu_k) / sum_k w_k
    weight_sum = keras.ops.sum(weights, axis=1) + 1e-8  # (B, 1) with stability epsilon

    centroid_source = keras.ops.sum(weights * mu_source, axis=1) / weight_sum  # (B, 3)
    centroid_target = keras.ops.sum(weights * mu_target, axis=1) / weight_sum  # (B, 3)

    # Centering removes translation, leaving only rotation to solve.
    mu_source_centered = mu_source - keras.ops.expand_dims(centroid_source, axis=1)
    mu_target_centered = mu_target - keras.ops.expand_dims(centroid_target, axis=1)

    # H = sum_k w_k * mu_source_k^T * mu_target_k -- the 3x3 matrix that encodes
    # the optimal rotation.
    # DECISION plan_2026-06-15_00924f53/D-001: pre-existing forward blocker exposed once the
    # graph-feature fix let the encoder run. `weights` is (B,K,1); to scale the transposed
    # source (B,3,K) per component, broadcast (B,1,K) via transpose -- NOT expand_dims(axis=1)
    # which yields a rank-4 (B,1,K,1) and crashes the Mul. Minimal in-scope F-LGM-2 fix.
    H = keras.ops.matmul(
        # Transpose source: (B, 3, K); per-component weights broadcast as (B, 1, K)
        keras.ops.transpose(mu_source_centered, (0, 2, 1)) * keras.ops.transpose(weights, (0, 2, 1)),
        mu_target_centered  # (B, K, 3)
    )  # Result: (B, 3, 3)

    # H = U * S * V^T; the optimal rotation is R = V * U^T when det(V*U^T) = +1.
    # Raw tf.linalg is used because keras.ops has no SVD (module docstring §L2-5).
    # DECISION plan_2026-06-15_00924f53/D-001: tf.linalg.svd returns (s, u, v) with
    # H = u @ diag(s) @ v^T (v is NOT pre-transposed). The original unpack `U,_,Vt`
    # mis-bound s->U and v->Vt, crashing transpose on the rank-2 singular values.
    # Bind u, v correctly; R = V @ U^T. Minimal in-scope F-LGM-2 fix.
    # DECISION plan-2026-08-19T163559-499b6f0e/D-011
    # Do NOT try to make this model `mixed_float16`-capable by wrapping this
    # call in a float32 cast island. TensorFlow has NO `Svd` kernel for
    # `DT_HALF` on either CPU or GPU (measured: the raise lists
    # `CPU: {float,double,complex64,complex128}`, `GPU: {double,float}`), and
    # orthogonalizing a 3x3 cross-covariance whose `det` sign selects the
    # reflection correction is not sound arithmetic in half precision anyway.
    # This model is float32-only BY DESIGN -- see the module docstring.
    _s, U, V = tf.linalg.svd(H)

    R = keras.ops.matmul(V, keras.ops.transpose(U, (0, 2, 1)))  # V * U^T

    # det(R) = -1 means a reflection, not a rotation; correct it by flipping the
    # sign of the smallest singular direction.
    det = tf.linalg.det(R)  # Shape: (B,)

    # Create correction matrix: diag([1, 1, det(R)])
    # When det(R) = +1, this is identity (no change)
    # When det(R) = -1, this flips the sign of the third component
    correction = keras.ops.stack([
        keras.ops.ones_like(det),
        keras.ops.ones_like(det),
        det
    ], axis=-1)  # Shape: (B, 3)
    correction_matrix = tf.linalg.diag(correction)  # Shape: (B, 3, 3)

    # Apply correction: R = V * correction * U^T
    R = keras.ops.matmul(
        keras.ops.matmul(V, correction_matrix),
        keras.ops.transpose(U, (0, 2, 1))
    )

    # After rotation, translation aligns the centroids: t = c_target - R * c_source
    t = centroid_target - keras.ops.squeeze(
        keras.ops.matmul(R, keras.ops.expand_dims(centroid_source, axis=-1)),
        axis=-1
    )  # Shape: (B, 3)

    return R, t

# ---------------------------------------------------------------------


def create_latent_gmm_registration(
        num_gaussians: int = 32,
        k_neighbors: int = 16,
        chamfer_weight: float = 1.0,
        transform_weight: float = 1.0,
        **kwargs: Any
) -> LatentGMMRegistration:
    """Create a LatentGMMRegistration model.

    There is no ``MODEL_VARIANTS`` table for this architecture: the paper
    defines a single network and only ``num_gaussians`` / ``k_neighbors``
    scale it, so this factory constructs the class with the paper defaults
    rather than delegating to ``from_variant``.

    Args:
        num_gaussians: Number of latent GMM components. Must be positive.
        k_neighbors: Number of neighbors used for local feature extraction.
            Must be positive and smaller than the number of input points.
        chamfer_weight: Weight of the Chamfer reconstruction loss.
        transform_weight: Weight of the supervised transformation loss.
        **kwargs: Additional arguments forwarded to the model constructor.

    Returns:
        A configured LatentGMMRegistration instance.

    Raises:
        ValueError: If any of the numeric arguments is out of range.

    Examples:
        >>> model = create_latent_gmm_registration(num_gaussians=8, k_neighbors=8)
        >>> out = model((keras.random.normal((2, 64, 3)),
        ...              keras.random.normal((2, 64, 3))))
        >>> tuple(out["estimated_r"].shape)
        (2, 3, 3)
    """
    return LatentGMMRegistration(
        num_gaussians=num_gaussians,
        k_neighbors=k_neighbors,
        chamfer_weight=chamfer_weight,
        transform_weight=transform_weight,
        **kwargs
    )

# ---------------------------------------------------------------------
