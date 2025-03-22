import tensorflow as tf
from typing import Optional

# ---------------------------------------------------------------------


DEFAULT_EPSILON = 1e-6

# ---------------------------------------------------------------------


def reshape_to_2d(
        weights: tf.Tensor,
        name: Optional[str] = None) -> tf.Tensor:
    """Reshape weight tensor to 2D matrix for regularization computations.

    Handles standard neural network weight tensor formats:
    - Dense: (in_features, out_features)
    - Conv2D: (h, w, in_c, out_c)
    - Conv3D: (d, h, w, in_c, out_c)
    - Conv1D: (w, in_c, out_c)

    Args:
        weights: Input weight tensor
        name: Optional name for the operation

    Returns:
        tf.Tensor: 2D tensor where first dimension is output features/channels

    Raises:
        tf.errors.InvalidArgumentError: If tensor rank is not 2, 3, 4, or 5
    """
    with tf.name_scope(name or "reshape_to_2d"):
        # Get tensor shape and rank using TF ops
        weights_shape = tf.shape(weights)
        ndims = tf.rank(weights)

        # Assert supported number of dimensions
        tf.debugging.assert_equal(
            tf.reduce_any(tf.equal(ndims, [2, 3, 4, 5])),
            True,
            message=(
                "Tensor rank must be one of:\n"
                "2 (Dense: in_features, out_features)\n"
                "3 (Conv1D: width, in_channels, out_channels)\n"
                "4 (Conv2D: height, width, in_channels, out_channels)\n"
                "5 (Conv3D: depth, height, width, in_channels, out_channels)"
            )
        )

        # For any conv layer (1D/2D/3D), last dimension is always out_channels
        # Everything else gets flattened into the second dimension
        out_channels = weights_shape[-1]

        # Create permutation indices tensor dynamically based on rank
        # Move out_channels to first dimension
        perm = tf.concat([
            [ndims - 1],  # Last dim (out_channels) goes first
            tf.range(ndims - 1)  # Other dims maintain relative order
        ], axis=0)

        # Transpose the weights according to the permutation
        w_t = tf.transpose(weights, perm)

        # Reshape to 2D: [out_channels, everything_else_flattened]
        return tf.reshape(w_t, [out_channels, -1])

# ---------------------------------------------------------------------


def gram_matrix(weights: tf.Tensor) -> tf.Tensor:
    """Compute the Gram matrix (W * W^T) with improved numerical stability.

    This function calculates the Gram matrix of the input weights tensor,
    which is useful for orthogonality constraints and regularization.

    Args:
        weights: Input weight tensor of any shape

    Returns:
        tf.Tensor: The Gram matrix resulting from W * W^T

    Note:
        The input tensor is first reshaped to 2D using reshape_to_2d
        before computing the Gram matrix for better numerical stability.
    """
    wt = reshape_to_2d(weights)
    return tf.matmul(wt, tf.transpose(wt))

# ---------------------------------------------------------------------


def wt_x_w_normalize(weights: tf.Tensor) -> tf.Tensor:
    """Compute normalized Gram matrix (W^T * W) with improved numerical stability.

    This function calculates a normalized version of the Gram matrix by first
    normalizing each row of the reshaped weight tensor before multiplication.
    This approach improves numerical stability and conditioning of the result.

    Args:
        weights: Input weight tensor of any shape

    Returns:
        tf.Tensor: The normalized Gram matrix resulting from W_norm^T * W_norm

    Note:
        The normalization process divides each row by its L2 norm with a small
        epsilon (1e-5) to prevent division by zero, helping with gradient flow
        during training.
    """
    # Reshape weights to 2D for consistent processing
    wt = reshape_to_2d(weights)

    # Normalize the weights before multiplication for better conditioning
    # Add small epsilon (1e-5) to prevent division by zero
    norm = tf.maximum(tf.norm(wt, axis=1, keepdims=True), DEFAULT_EPSILON)
    wt_normalized = wt / norm

    # Compute the normalized Gram matrix
    return tf.matmul(wt_normalized, tf.transpose(wt_normalized))


# ---------------------------------------------------------------------

def power_iteration(
        matrix: tf.Tensor,
        iterations: int = 10,
        epsilon: float = DEFAULT_EPSILON
) -> tf.Tensor:
    """
    Compute spectral norm using power iteration.

    Args:
        matrix: Input matrix
        iterations: Number of power iterations
        epsilon: Small number for numerical stability

    Returns:
        tf.Tensor: Spectral norm (largest singular value)
    """
    if len(matrix.shape) != 2:
        raise ValueError("Input matrix must be 2-dimensional")

    # Initialize random vector
    matrix_shape = tf.shape(matrix)
    vector = tf.random.normal([matrix_shape[1], 1])
    vector = vector / (tf.norm(vector) + epsilon)

    # Multiple iterations for convergence
    for _ in range(iterations):
        # Compute matrix-vector product
        product = tf.matmul(matrix, vector)
        vector = product / (tf.norm(product) + epsilon)

        # Compute transpose multiplication
        product = tf.matmul(matrix, vector, transpose_a=True)
        vector = product / (tf.norm(product) + epsilon)

    # Final power iteration step
    product = tf.matmul(matrix, vector)

    # Compute spectral norm using the ratio of norms
    return tf.norm(product) / (tf.norm(vector) + epsilon)


# ---------------------------------------------------------------------

def create_causal_mask(size: int) -> tf.Tensor:
    """Create causal mask for attention.

    Args:
        size: Sequence length

    Returns:
        Causal mask tensor
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return tf.cast(mask == 0, tf.float32)


# ---------------------------------------------------------------------

def safe_divide(
        x: tf.Tensor,
        y: tf.Tensor,
        eps: float = DEFAULT_EPSILON) -> tf.Tensor:
    """Safe division with epsilon to prevent div by zero.

    Args:
        x: Numerator tensor
        y: Denominator tensor
        eps: Small constant for numerical stability

    Returns:
        Result of safe division
    """
    return x / (y + tf.constant(eps))

# ---------------------------------------------------------------------
