import keras
from typing import Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.losses.chamfer_loss import ChamferLoss
from dl_techniques.layers.geometric.point_cloud_autoencoder import PointCloudAutoencoder, CorrespondenceNetwork

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

        # Validate inputs
        if num_gaussians <= 0:
            raise ValueError(f"num_gaussians must be positive, got {num_gaussians}")
        if k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
        if chamfer_weight < 0:
            raise ValueError(f"chamfer_weight must be non-negative, got {chamfer_weight}")
        if transform_weight < 0:
            raise ValueError(f"transform_weight must be non-negative, got {transform_weight}")

        # Store configuration
        self.num_gaussians = num_gaussians
        self.k_neighbors = k_neighbors
        self.chamfer_weight = chamfer_weight
        self.transform_weight = transform_weight

        # Create all sub-layers in __init__
        self.autoencoder = PointCloudAutoencoder(
            k_neighbors=k_neighbors,
            name="autoencoder"
        )
        self.correspondence_net = CorrespondenceNetwork(
            num_gaussians=num_gaussians,
            name="correspondence_net"
        )

        # Loss function
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

        # Step 1: Feature Extraction and Reconstruction
        # The autoencoder processes both point clouds simultaneously to extract:
        # - Reconstructions (x_rec, y_rec): Decoded point clouds for Chamfer loss
        # - Local features: Per-point features capturing neighborhood geometry
        # - Global features: Point cloud-level features capturing overall structure
        (x_rec, y_rec), (local_x, local_y), (global_x, global_y) = self.autoencoder(
            (source_pc, target_pc),
            training=training
        )

        # Step 2: Correspondence Estimation
        # Compute soft assignments (gamma) of each point to K Gaussian components
        # gamma[i,j,k] represents the probability that point j belongs to component k
        # The correspondence network is shared between source and target for consistency
        gamma_x = self.correspondence_net((local_x, global_x), training=training)
        gamma_y = self.correspondence_net((local_y, global_y), training=training)

        # Step 3: GMM Parameter Estimation
        # From soft assignments and point positions, compute GMM statistics:
        # - pi: mixing coefficients (component weights)
        # - mu: component means (centroids of each Gaussian)
        # These operations are differentiable but contain no trainable parameters
        pi_x, mu_x = compute_gmm_params(source_pc, gamma_x)
        pi_y, mu_y = compute_gmm_params(target_pc, gamma_y)

        # Step 4: Rigid Transform Estimation
        # Solve weighted Procrustes problem to find optimal rotation R and translation t
        # that aligns source GMM (mu_x, pi_x) to target GMM (mu_y, pi_y)
        # This gives us the estimated transformation between point clouds
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

        with keras.backend.GradientTape() as tape:
            # Get model predictions
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

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply(gradients, trainable_vars)

        # Update metrics
        self.compiled_metrics.update_state(
            (R_gt, t_gt),
            (y_pred["estimated_r"], y_pred["estimated_t"])
        )

        return {
            "loss": total_loss,
            "chamfer_loss": total_chamfer_loss,
            "transform_loss": total_transform_loss,
            **{m.name: m.result() for m in self.metrics},
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

        # Get model predictions (no gradient tape needed for testing)
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

        # Update metrics
        self.compiled_metrics.update_state(
            (R_gt, t_gt),
            (y_pred["estimated_r"], y_pred["estimated_t"])
        )

        return {
            "loss": total_loss,
            "chamfer_loss": total_chamfer_loss,
            "transform_loss": total_transform_loss,
            **{m.name: m.result() for m in self.metrics},
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

    # Normalize by total responsibility (pi) to get component means
    # Add epsilon to avoid division by zero for components with negligible weight
    pi_expanded = keras.ops.expand_dims(pi, axis=-1) + 1e-8  # (B, K, 1)
    mu = weighted_sum / pi_expanded  # Shape: (B, K, 3)

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
    # Step 1: Compute component correspondence weights
    # w_k = pi_source_k * pi_target_k represents joint importance of corresponding components
    weights = keras.ops.expand_dims(
        pi_source * pi_target,
        axis=-1
    )  # Shape: (B, K, 1)

    # Step 2: Compute weighted centroids
    # centroid = sum_k (w_k * mu_k) / sum_k w_k
    weight_sum = keras.ops.sum(weights, axis=1) + 1e-8  # (B, 1) with stability epsilon

    centroid_source = keras.ops.sum(weights * mu_source, axis=1) / weight_sum  # (B, 3)
    centroid_target = keras.ops.sum(weights * mu_target, axis=1) / weight_sum  # (B, 3)

    # Step 3: Center both GMM means around their respective centroids
    # This removes translation, leaving only rotation to solve
    mu_source_centered = mu_source - keras.ops.expand_dims(centroid_source, axis=1)
    mu_target_centered = mu_target - keras.ops.expand_dims(centroid_target, axis=1)

    # Step 4: Compute weighted cross-covariance matrix
    # H = sum_k w_k * mu_source_k^T * mu_target_k
    # This 3x3 matrix encodes the optimal rotation information
    H = keras.ops.matmul(
        # Transpose source: (B, 3, K)
        keras.ops.transpose(mu_source_centered, (0, 2, 1)) * keras.ops.expand_dims(weights, axis=1),
        mu_target_centered  # (B, K, 3)
    )  # Result: (B, 3, 3)

    # Step 5: SVD decomposition H = U * S * V^T
    # The optimal rotation is given by R = V * U^T (when det(V*U^T) = +1)
    # Note: Using TensorFlow backend for SVD as keras.ops doesn't have native SVD
    import tensorflow as tf
    U, _, Vt = tf.linalg.svd(H)  # Vt is already transposed

    # Compute initial rotation matrix
    R = keras.ops.matmul(Vt, keras.ops.transpose(U, (0, 2, 1)))  # V * U^T

    # Step 6: Ensure proper rotation (det(R) = +1, not -1 for reflection)
    # If det(R) = -1, we have a reflection instead of rotation
    # Fix by negating the last column of V (equivalent to flipping sign of smallest singular value)
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
        keras.ops.matmul(Vt, correction_matrix),
        keras.ops.transpose(U, (0, 2, 1))
    )

    # Step 7: Compute translation
    # After rotation, translation aligns the centroids: t = c_target - R * c_source
    t = centroid_target - keras.ops.squeeze(
        keras.ops.matmul(R, keras.ops.expand_dims(centroid_source, axis=-1)),
        axis=-1
    )  # Shape: (B, 3)

    return R, t

# ---------------------------------------------------------------------
