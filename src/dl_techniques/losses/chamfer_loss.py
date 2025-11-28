import keras

# ---------------------------------------------------------------------

class ChamferLoss(keras.losses.Loss):
    """Computes the Chamfer distance between two point clouds.

    The Chamfer distance is a symmetric distance metric between two point clouds,
    computed as the sum of the mean minimum distances from each point in the first
    cloud to the second cloud and vice versa.

    Args:
        reduction: Type of reduction to apply to the loss. Default is 'sum_over_batch_size'.
        name: Name of the loss instance.

    Returns:
        Scalar loss value representing the Chamfer distance.

    Examples:
        >>> loss_fn = ChamferLoss()
        >>> pc1 = keras.random.normal((32, 1024, 3))  # batch_size=32, 1024 points, 3D
        >>> pc2 = keras.random.normal((32, 1024, 3))
        >>> loss = loss_fn(pc1, pc2)
    """

    def __init__(
            self,
            reduction: str = "sum_over_batch_size",
            name: str = "chamfer_loss"
    ):
        """Initialize the ChamferLoss.

        Args:
            reduction: Type of reduction to apply to the loss.
            name: Name of the loss instance.
        """
        super().__init__(reduction=reduction, name=name)

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Computes the Chamfer distance between two point clouds.

        Args:
            y_true: Ground truth point cloud of shape (batch_size, num_points, num_dims).
            y_pred: Predicted point cloud of shape (batch_size, num_points, num_dims).

        Returns:
            Chamfer distance loss of shape (batch_size,).
        """
        # Pairwise distance matrix: (batch_size, num_points1, num_points2)
        pc1_expand = keras.ops.expand_dims(y_true, axis=2)
        pc2_expand = keras.ops.expand_dims(y_pred, axis=1)
        dist_mat = keras.ops.sum(keras.ops.square(pc1_expand - pc2_expand), axis=-1)

        # Minimum distances
        dist_1 = keras.ops.min(dist_mat, axis=2)
        dist_2 = keras.ops.min(dist_mat, axis=1)

        # Mean over points for each sample in batch
        chamfer_dist = keras.ops.mean(dist_1, axis=-1) + keras.ops.mean(dist_2, axis=-1)

        return chamfer_dist

# ---------------------------------------------------------------------

