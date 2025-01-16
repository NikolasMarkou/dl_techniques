import tensorflow as tf


# ---------------------------------------------------------------------

@tf.function
def reshape_to_2d(weights: tf.Tensor) -> tf.Tensor:
    """Reshape N-dimensional tensor to 2D matrix for regularization computations.

    This function takes a tensor of any dimension and reshapes it into a 2D matrix
    where the last dimension becomes the first dimension (F) and all other dimensions
    are flattened into the second dimension.

    Args:
        weights: Input tensor of shape (..., F)

    Returns:
        2D tensor of shape (F, prod(...))
    """
    # Get shape information
    weights_shape = tf.shape(weights)
    ndims = len(weights.shape)

    # Ensure tensor has at least 2 dimensions
    tf.debugging.assert_greater_equal(
        ndims, 2,
        message=f"Input tensor must have at least 2 dimensions, got {ndims}"
    )

    # No reshape needed for 2D
    if ndims == 2:
        return tf.transpose(weights)

    # For N-dimensional tensors:
    # 1. Move last dimension to front
    # 2. Flatten all other dimensions
    F = weights_shape[-1]  # Last dimension size
    perm = tf.concat([
        [ndims - 1],  # Last dim goes first
        tf.range(ndims - 1)  # Other dims follow
    ], axis=0)

    # Reshape to (F, -1) where -1 is product of all other dimensions
    w_t = tf.transpose(weights, perm)
    spatial_dims = tf.reduce_prod(weights_shape[:-1])  # All dims except last

    return tf.reshape(w_t, [F, spatial_dims])


# ---------------------------------------------------------------------

def power_iteration(
        matrix: tf.Tensor,
        iterations: int = 10,
        epsilon: float = 1e-6
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

def safe_divide(x: tf.Tensor, y: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    """Safe division with epsilon to prevent div by zero.

    Args:
        x: Numerator tensor
        y: Denominator tensor
        eps: Small constant for numerical stability

    Returns:
        Result of safe division
    """
    return x / (y + eps)

# ---------------------------------------------------------------------
