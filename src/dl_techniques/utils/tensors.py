import keras
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Any, List, Union

# ---------------------------------------------------------------------


DEFAULT_EPSILON = 1e-6

# ---------------------------------------------------------------------


# DECISION plan-2026-08-31T175140-a4e0c303/D-014
# Keep the explicit `n >= 1` clause below. Do NOT collapse `is_power_of_two`
# to the bare bit trick: it accepts zero, which admits a zero-width dimension
# into both binary-tree consumers.
def is_power_of_two(n: int) -> bool:
    """Report whether ``n`` is a power of two.

    Zero and negative values are ``False``, so this is safe to call on any int
    -- including an unresolved ``-1`` dimension. It is the guard both binary-tree
    layers use to reject a last dimension they cannot pair:
    ``layers/structured_linear/orthogonal_butterfly.py`` and
    ``layers/norms/polar_weight_norm.py``,
    which held byte-identical private copies until
    ``plan-2026-08-31-a4e0c303/iter-1/step-4``. This module imports nothing from
    either of them; the dependency is one-way and stays that way.

    .. warning::
       Do NOT "simplify" this to ``return n & (n - 1) == 0``. Dropping the
       ``n >= 1`` clause makes the predicate accept ``0`` (``0 & -1 == 0``),
       which would admit a zero-width dimension into both trees. Pinned by
       ``tests/test_utils/test_tensors.py::TestIsPowerOfTwo``.

    :param n: The integer to test.
    :type n: int
    :return: ``True`` when ``n >= 1`` and ``n`` has exactly one set bit.
    :rtype: bool
    """
    return n >= 1 and (n & (n - 1)) == 0


def make_divisible(value: float, divisor: int = 8) -> int:
    """Round a channel width to the nearest multiple of ``divisor``, MobileNet style.

    The reference rule (torchvision ``models/_utils._make_divisible``, Keras
    Applications ``mobilenet_v3._depth``, both with ``min_value=divisor``):
    nearest multiple of ``divisor``, at least ``divisor``, and one more step
    when that rounding fell more than 10% below ``value``.

    Interface contract (callers: ``MobileNetV2``, ``MobileNetV3`` channel widths,
    ``SqueezeExcitation(divisor > 1)`` bottleneck width): pure, deterministic,
    no keras op; returns a Python ``int`` >= ``divisor``. Does not validate
    ``divisor`` (callers that take it from a user check it themselves).

    :param value: The unrounded width (any non-negative real).
    :type value: float
    :param divisor: The multiple to round to. Defaults to 8.
    :type divisor: int
    :return: The rounded width.
    :rtype: int
    """
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value

# ---------------------------------------------------------------------


def canonical_binary_input_shape(
        input_shape: Union[
            Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]
        ]
) -> Tuple[Optional[int], ...]:
    """
    Return the one shape a binary elementwise layer works from, validating the pair.

    ``build``, ``call`` and ``compute_output_shape`` all have to answer the
    same three questions -- is this one shape or a list of shapes, are
    there at most two, and do the two agree -- so they ask them here. A
    single shape can itself arrive as a list, for example after
    deserialization, so the list-of-shapes test looks at the first element:
    a list OF shapes has non-dimension elements.

    ``layers/logic/logic_operators.py`` and
    ``layers/logic/arithmetic_operators.py`` held byte-identical private
    copies of this, named ``_canonical_input_shape``, until
    ``plan-2026-08-31-a4e0c303/iter-1/step-5``; both now import this one and
    call it from three sites each. This module imports nothing from either of
    them; the dependency is one-way and stays that way.

    .. warning::
       Do NOT reduce the body to ``return tuple(input_shape[0])``. The
       single-shape branch is load-bearing -- a plain shape tuple would be
       truncated to its first dimension -- and the two raise paths are the
       whole point of the helper. Pinned tree-wide by
       ``tests/test_layers/test_logic/test_the_nine_source_fixes_stay_fixed.py``
       ``::TestB5TheShapeCheckLivesInOnePureHelper`` and by
       ``tests/test_utils/test_tensors.py::TestCanonicalBinaryInputShape``.

    :param input_shape: One shape, or a list of one or two shapes.
    :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    :return: The single shape to build from, which is the first one.
    :rtype: Tuple[Optional[int], ...]
    :raises ValueError: If more than two shapes are given, or two shapes
        are given and their dimensions differ.
    """
    is_list_of_shapes = (
        isinstance(input_shape, list)
        and input_shape
        and not isinstance(input_shape[0], (int, type(None)))
    )
    if not is_list_of_shapes:
        return tuple(input_shape) if isinstance(input_shape, list) else input_shape
    if len(input_shape) > 2:
        raise ValueError(
            f"Expected 1 or 2 inputs, got {len(input_shape)}"
        )
    if len(input_shape) == 2 and list(input_shape[0]) != list(input_shape[1]):
        raise ValueError(
            f"Input tensors must have the same shape for binary operations. "
            f"Got shapes: {input_shape[0]} and {input_shape[1]}"
        )
    return tuple(input_shape[0])

# ---------------------------------------------------------------------


def is_single_shape(shape: Any) -> bool:
    """Is this ONE shape, rather than a collection of shapes?

    Interface contract, shared by every layer that accepts "one shape or a
    list of shapes" (``layers/heads/vlm/factory.py`` checks a fusion output
    with it; ``layers/heads/vision/factory.py``'s ``SegmentationHead`` checks
    its ``build`` and ``compute_output_shape`` input with it):

    * Returns ``True`` for one shape: a list, tuple or other sized iterable
      (``tf.TensorShape``) whose entries are ints or ``None``.
    * Returns ``False`` for a collection of shapes (entries that are
      themselves sequences), for a string, and for anything that is not a
      sized iterable.
    * Never raises.

    A shape and a list of shapes are both sequences, so
    ``isinstance(shape, (list, tuple))`` cannot tell them apart. The
    discriminator is the ELEMENT type. This matters most after
    deserialization: Keras restores a layer's ``build_config`` input shape
    from JSON as a LIST of ints, which a list-means-several-shapes test reads
    as several one-int "shapes".

    Moved here from ``layers/heads/vlm/factory.py`` (``_is_single_shape``),
    where ``BaseVLMHead._build_fusion_stack`` used it to reject a per-modality
    fusion output before Keras surfaced it as ``ValueError: Invalid dtype:
    tuple``, when ``SegmentationHead`` needed the same rule.

    :param shape: A shape, or a list/tuple of shapes.
    :type shape: Any
    :return: ``True`` for a single shape, ``False`` otherwise.
    :rtype: bool
    """
    # `TensorShape` is a sequence of ints but not a list/tuple, so an
    # isinstance check against (list, tuple) alone MISCLASSIFIES it as "not a
    # single shape" -- and a list OF TensorShapes as "single".
    # A string is iterable but is never a shape.
    if isinstance(shape, (str, bytes)):
        return False
    if isinstance(shape, (list, tuple)):
        entries = shape
    # Any other sized iterable, such as a `tf.TensorShape`.
    elif hasattr(shape, "__iter__") and hasattr(shape, "__len__"):
        entries = list(shape)
    else:
        return False
    return not any(
        isinstance(entry, (list, tuple))
        or (hasattr(entry, "__len__") and not isinstance(entry, (str, bytes, int)))
        for entry in entries
    )

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
        epsilon: float = DEFAULT_EPSILON,
        seed: Optional[int] = None
) -> tf.Tensor:
    """
    Compute spectral norm using power iteration.

    Args:
        matrix: Input matrix
        iterations: Number of power iterations
        epsilon: Small number for numerical stability
        seed: Optional integer seed for the random start vector. ``None`` (the
            default) keeps the historical UNSEEDED draw, so repeated calls
            return slightly different estimates at low ``iterations``. An
            integer makes the start vector -- and therefore the returned
            estimate -- fully deterministic: same seed and same matrix give
            bit-identical results across calls, across processes, and
            regardless of any ``tf.random.set_seed`` the caller has made
            (see the DECISION note below for why that last property is not
            achievable with an op-level seed).

    Returns:
        tf.Tensor: Spectral norm (largest singular value)
    """
    if len(matrix.shape) != 2:
        raise ValueError("Input matrix must be 2-dimensional")

    # Initialize random vector
    #
    # DECISION plan-2026-07-30T140922-8af1028f/D-025
    # An UNSEEDED start vector (`seed=None`, the historical and still-default
    # path) makes this function non-deterministic. At low `iterations` the
    # estimate has not converged, so whether it lands inside a given tolerance
    # depends on the draw: `TestPowerIteration::test_convergence_iterations`
    # (`iterations=2` vs `iterations=20`, `rtol=1e-1`) failed at a measured
    # 6.25% +/- 1.06% (95% CI, N=2000 executions of the real test body,
    # 2026-07-30). That rate is measured HERE and nowhere else -- do not restate
    # it elsewhere; cite this comment.
    #
    # Do NOT "fix" that flake by widening the test's tolerance: the bound is
    # attainable, and the error tail at `iterations=2` reaches the
    # wrong-eigenvector case (~50% relative error), so a safe rtol would have to
    # be >=0.5, which makes the test vacuous. Do NOT raise an iteration floor
    # either -- the empirical "8 is enough" floor was derived from ONE matrix's
    # 2:1 eigenvalue gap and does not generalise, and the test exists precisely
    # to probe a LOW iteration count. The fix is the optional `seed` below.
    #
    # Seeding convention follows the sibling implementation
    # `SRIPRegularizer._power_iteration` (`regularizers/srip.py:236-244`), which
    # draws with `keras.random.normal(..., seed=int)`. `stateless_normal` with
    # `seed=[seed, 0]` is BIT-IDENTICAL to that call (verified) while keeping
    # this module raw-`tf` like the rest of it.
    #
    # `tf.random.normal(..., seed=seed)` is NOT a substitute and must not be
    # substituted here: an op-level seed is combined with the process-global
    # seed and advances per call, so two consecutive same-seed calls return
    # DIFFERENT vectors (measured) -- it would leave the flake in place while
    # looking seeded. Only `seed=None` may reach `tf.random.normal`.
    matrix_shape = tf.shape(matrix)
    if seed is None:
        vector = tf.random.normal([matrix_shape[1], 1])
    else:
        vector = tf.random.stateless_normal([matrix_shape[1], 1], seed=[seed, 0])
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

def l2_normalize(
        x: keras.KerasTensor,
        axis: int = -1,
        eps: float = 1e-7) -> keras.KerasTensor:
    """L2-normalize along ``axis`` without losing the result to overflow.

    The obvious spelling fails at BOTH ends of the input scale, because it
    squares before it rescales::

        x / sqrt(sum(x**2) + eps)

    Measured in float32 against this function, over 256x8 inputs, error taken
    against a float64 oracle:

    ==================  =========================  ==========================
    input scale         naive error                this function's error
    ==================  =========================  ==========================
    ``1e-08``           ``9.5e-01``                ``9.1e-08``
    ``1e-06``           ``9.5e-01``                ``8.3e-08``
    ``1e-03``           ``2.2e-02``                ``1.1e-07``
    ``1e+00``           ``8.3e-08``                ``9.7e-08``
    ==================  =========================  ==========================

    Two distinct failures live in that table, and they are worth keeping apart:

    * **Underflow at small scale.** ``x**2`` of a ``1e-8`` input is ``1e-16``,
      below float32's normal range, so the summed norm loses most of its
      significant digits. Worse, ``eps`` is then added to a number of the same
      order, so the epsilon dominates the norm entirely and the result is not a
      unit vector at all -- it is off by nearly the full magnitude. The output
      stays finite and plausible, so nothing reports the problem.
    * **Overflow at large scale.** Above ``sqrt(3.4e38) ~ 1.84e19`` the squared
      sum becomes ``inf``, so ``x / inf`` is exactly ``0``. The result is again
      finite -- measured ``absmax = 0.0`` at inputs of ``1e20``, ``1e30`` and
      ``1e38`` -- and silently carries no directional information. This is NOT
      a NaN: ``x / inf == 0``. A NaN arises only when the INPUT is already
      ``inf`` or ``NaN``, which this function does not and cannot repair.

    The fix is to rescale before squaring. Dividing by the maximum absolute
    coordinate first bounds the vector elementwise by 1, so the squared sum can
    neither overflow nor sit near the denormal range. The result is
    algebraically identical -- ``x/||x|| == (x/max|x|)/||x/max|x||`` -- so this
    is the same quantity computed stably, not a different estimator. It costs
    one extra reduction over the axis already being reduced.

    The one regime this does NOT rescue is input whose magnitude is so small
    that float32 has no precision left in it. Sweeping downward against the
    oracle, error stays at the 8.5e-08 to 1.0e-07 floor from ``1e-33`` up to
    ``1e+38``; at ``1e-34`` it degrades to ``1.3e-04``, and in the subnormal
    range (``1e-44``) to ``0.88`` -- identical there for the naive spelling, so
    that limit is a property of the dtype rather than of the method. The
    practical floor is therefore ``~1e-33``, which is a measured bound rather
    than a guess at min-normal (``1.175e-38``, about four decades further down,
    where nothing changes). An activation that small has already lost its data
    upstream.

    An all-zero vector has ``max|x| = 0``, which would divide into ``0/0``. That
    is handled by substituting ``1`` for a zero divisor rather than by flooring
    the divisor with a constant. Flooring does not work here: legitimate float32
    data reaches down to ``1.175e-38``, so any floor above that clamps real
    input -- a floor of ``1e-30`` was measured leaving a genuine ``1e-30``
    vector with an error of ``5.8e-07`` -- while any floor below that is
    subnormal, and ``0 / subnormal`` rounds to ``nan`` in float32. The
    substitution has no threshold to get wrong.

    Note there is deliberately NO second guard on the RESULT. ``peak == 0``
    implies ``x`` is all zeros, so ``scaled`` is already ``0``, the norm is
    ``sqrt(eps)``, and ``0 / sqrt(eps)`` is ``0`` -- the correct answer falls out
    of the divisor substitution alone. A trailing ``ops.where`` was present and
    has been removed: mutation testing found that deleting it changed nothing
    at all (bit-identical on zero, ordinary, ``1e-30``, ``1e30`` and mixed
    zero/huge rows), which makes it dead code that only LOOKS like a safety
    net.

    Args:
        x: Tensor to normalize.
        axis: Axis reduced over. Defaults to the last.
        eps: Added to the norm of the *pre-scaled* vector, so a degenerate one
            yields zeros rather than a division by zero.
    Returns:
        Tensor of the same shape, unit norm along ``axis`` wherever the input had
        non-zero norm, and zeros wherever it did not.

    References:
        - Higham, 2002. Accuracy and Stability of Numerical Algorithms, 2nd ed.
          (https://epubs.siam.org/doi/10.1137/1.9780898718027) -- scaling
          before squaring when computing a norm in finite precision.
    """
    # Rescale first: |x| <= 1 elementwise from here on, so the squared sum is
    # bounded by the vector length regardless of the input magnitude. Dividing
    # by a strictly positive peak is safe at ANY positive float32 -- including a
    # subnormal one -- because the quotient is bounded by 1 by construction.
    peak = keras.ops.max(keras.ops.abs(x), axis=axis, keepdims=True)

    scaled = x / keras.ops.where(peak > 0.0, peak, keras.ops.ones_like(peak))
    norms = keras.ops.sqrt(keras.ops.sum(keras.ops.square(scaled), axis=axis, keepdims=True) + eps)

    return scaled / norms


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

# ---------------------------------------------------------------------

def gaussian_kernel(
        kernel_size: Tuple[int, int],
        nsig: Tuple[float, float]
) -> np.ndarray:
    """
    Build a 2D Gaussian kernel array.

    Args:
        kernel_size (Tuple[int, int]): Size of the grid (height, width).
        nsig (Tuple[float, float]): Half-extent of the grid along (height,
            width), in units of the Gaussian's standard deviation: the kernel
            samples ``exp(-t^2 / 2)`` at ``linspace(-nsig, nsig, size)`` per
            axis, so the standard deviation in pixels is
            about ``(size - 1) / (2 * nsig)`` (the untruncated limit, accurate
            for ``nsig >= 3``; a truncated kernel is narrower) and a larger
            ``nsig`` blurs less.

    Returns:
        np.ndarray: 2D Gaussian kernel of shape ``kernel_size``.
    """
    if len(nsig) != 2 or len(kernel_size) != 2:
        raise ValueError("Both kernel_size and nsig must be tuples of length 2.")

    x = np.linspace(-nsig[0], nsig[0], kernel_size[0])
    y = np.linspace(-nsig[1], nsig[1], kernel_size[1])
    # ``ij`` indexing keeps axis 0 = height / ``nsig[0]``. The default ``xy``
    # returns shape ``(width, height)`` with the two sigmas swapped, which made
    # a non-square kernel_size fail to broadcast into ``(height, width)``.
    x, y = np.meshgrid(x, y, indexing="ij")

    kernel = np.exp(-(x ** 2 + y ** 2) / 2)
    return kernel / np.sum(kernel)


# ---------------------------------------------------------------------


def depthwise_gaussian_kernel(
        channels: int = 3,
        kernel_size: Tuple[int, int] = (5, 5),
        nsig: Tuple[float, float] = (2.0, 2.0),
        dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """
    Create a depthwise Gaussian kernel.

    Args:
        channels (int): Number of input channels.
        kernel_size (Tuple[int, int]): Size of the kernel (height, width).
        nsig (Tuple[float, float]): Half-extent of the grid along (height,
            width) in standard deviations; see ``gaussian_kernel``.
        dtype (Optional[np.dtype]): Data type of the output kernel.

    Returns:
        np.ndarray: Depthwise Gaussian kernel of shape (kernel_height, kernel_width, in_channels, 1).
    """
    # Generate the 2D Gaussian kernel
    kernel_2d = gaussian_kernel(kernel_size, nsig)

    # Create the depthwise kernel
    kernel = np.zeros((*kernel_size, channels, 1))
    for i in range(channels):
        kernel[:, :, i, 0] = kernel_2d

    # Set the data type
    if dtype is not None:
        kernel = kernel.astype(dtype)

    return kernel

# ---------------------------------------------------------------------


def compute_prediction_entropy(
        y_pred: keras.KerasTensor,
        temperature: float = 1.0,
        epsilon: float = 1e-8
) -> keras.KerasTensor:
    """
    Compute the entropy of predictions for calibration analysis.

    This utility helps analyze how well-calibrated the model's predictions are,
    which is crucial for understanding Goodhart's Law effects.

    Args:
        y_pred: Predicted logits, shape (batch_size, num_classes)
        temperature: Temperature scaling factor. Default: 1.0
        epsilon: Small constant for numerical stability. Default: 1e-8
    Returns:
        Entropy values for each prediction, shape (batch_size,)

    Example:
        >>> entropies = compute_prediction_entropy(y_pred, temperature=2.0)
        >>> mean_entropy = keras.ops.mean(entropies)
        >>> print(f"Average prediction entropy: {mean_entropy:.4f}")
    """
    scaled_logits = y_pred / temperature
    probs = keras.ops.softmax(scaled_logits, axis=-1)
    probs = keras.ops.clip(probs, x_min=epsilon, x_max=1.0 - epsilon)
    entropy = -keras.ops.sum(probs * keras.ops.log(probs), axis=-1)
    return entropy

# ---------------------------------------------------------------------


def validate_orthonormality(
    vectors: Any,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Validates that a set of vectors is orthonormal using the Keras backend.

    A set of vectors is orthonormal if each vector is a unit vector and all
    vectors in the set are mutually orthogonal. This is verified by checking
    if the Gram matrix (vectors @ vectors.T) is close to the identity matrix.

    Parameters
    ----------
    vectors : tensor-like
        A 2D tensor-like object (e.g., Keras tensor, NumPy array, TensorFlow
        tensor) where each row represents a vector.
    rtol : float, optional
        Relative tolerance for the `allclose` comparison. Defaults to 1e-5.
    atol : float, optional
        Absolute tolerance for the `allclose` comparison. Defaults to 1e-8.

    Returns
    -------
    bool
        `True` if the vectors are orthonormal within the specified tolerance,
        `False` otherwise.

    Raises
    ------
    TypeError
        If the input `vectors` do not have a floating-point dtype.
    ValueError
        If the input `vectors` is not a 2D matrix.

    Example
    -------
    >>> import numpy as np
    >>> # A perfect orthonormal basis
    >>> perfect_vectors = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    >>> validate_orthonormality(perfect_vectors)
    True

    >>> # A non-orthogonal set
    >>> non_ortho = np.array([[1., 0.5], [0., 1.]])
    >>> validate_orthonormality(non_ortho)
    False

    >>> # A non-normalized set
    >>> non_normalized = np.array([[2., 0.], [0., 1.]])
    >>> validate_orthonormality(non_normalized)
    False
    """
    # 1. Convert input to a tensor and perform validation
    try:
        vectors = keras.ops.convert_to_tensor(vectors)
    except Exception as e:
        raise TypeError(f"Input could not be converted to a Keras tensor. Error: {e}")

    # this is set to the highest precision because errors accumulate for large matrices
    vectors = keras.ops.cast(vectors, dtype='float64')

    if keras.ops.ndim(vectors) != 2:
        raise ValueError(
            "Input must be a 2D matrix where each row is a vector, but got "
            f"shape {keras.ops.shape(vectors)}."
        )

    # 2. Handle the edge case of an empty set of vectors
    n_vectors = keras.ops.shape(vectors)[0]
    if n_vectors == 0:
        return True  # An empty set is vacuously orthonormal

    # 3. Compute the Gram matrix (inner products of all vector pairs)
    gram_matrix = keras.ops.matmul(vectors, keras.ops.transpose(vectors))

    # 4. Create the target identity matrix
    identity = keras.ops.eye(n_vectors, dtype=vectors.dtype)

    # 5. Manually implement the `allclose` check since it is not in keras.ops.
    #    The formula is: absolute(a - b) <= atol + rtol * absolute(b)
    #    This must hold true for all elements in the tensors.
    is_close = keras.ops.absolute(gram_matrix - identity) <= (
        atol + rtol * keras.ops.absolute(identity)
    )
    is_orthonormal = keras.ops.all(is_close)

    # Return a standard Python boolean, as ops.all() returns a scalar tensor
    return bool(is_orthonormal)

# ---------------------------------------------------------------------

def window_partition(x: keras.KerasTensor, window_size: int) -> keras.KerasTensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x: Feature map tensor of shape (B, H, W, C).
        window_size: Window size for partitioning.

    Returns:
        Windows tensor of shape (B*num_windows, window_size, window_size, C).
    """
    B, H, W, C = keras.ops.shape(x)[0], keras.ops.shape(x)[1], keras.ops.shape(x)[2], keras.ops.shape(x)[3]
    x = keras.ops.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = keras.ops.reshape(windows, (-1, window_size, window_size, C))
    return windows

# ---------------------------------------------------------------------

def window_reverse(windows: keras.KerasTensor, window_size: int, H: int, W: int) -> keras.KerasTensor:
    """Reverse window partitioning back to feature map.

    Args:
        windows: Windows tensor of shape (B*num_windows, window_size, window_size, C).
        window_size: Window size that was used for partitioning.
        H: Height of the original feature map.
        W: Width of the original feature map.

    Returns:
        Feature map tensor of shape (B, H, W, C).
    """
    B = keras.ops.shape(windows)[0] // (H * W // window_size // window_size)
    x = keras.ops.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
    x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = keras.ops.reshape(x, (B, H, W, -1))
    return x

# ---------------------------------------------------------------------

def gaussian_probability(y: keras.KerasTensor, mu: keras.KerasTensor, sigma: keras.KerasTensor) -> keras.KerasTensor:
    """Compute Gaussian probability density using Keras operations.

    Parameters
    ----------
    y : keras.KerasTensor
        Target values tensor of shape [batch_size, 1, output_dim] or [batch_size, output_dim]
    mu : keras.KerasTensor
        Mean values tensor of shape [batch_size, num_mixtures, output_dim]
    sigma : keras.KerasTensor
        Standard deviation tensor of shape [batch_size, num_mixtures, output_dim]

    Returns
    -------
    keras.KerasTensor
        Probability densities tensor of shape [batch_size, num_mixtures, output_dim]
    """
    # Ensure numerical stability with a minimum standard deviation
    sigma = keras.ops.maximum(1e-6, sigma)
    sigma = keras.ops.cast(sigma, "float32")

    # Compute normalized squared difference
    norm = keras.ops.sqrt(2.0 * np.pi) * sigma
    y = keras.ops.cast(y, "float32")
    mu = keras.ops.cast(mu, "float32")
    norm = keras.ops.cast(norm, "float32")
    exp_term = -0.5 * keras.ops.square((y - mu) / sigma)

    return keras.ops.exp(exp_term) / norm

# ---------------------------------------------------------------------

# Lanczos approximation coefficients, g = 7, n = 9 (Press et al., Numerical
# Recipes, 3rd ed., §6.1). Accurate to ~1e-15 relative in float64 for x > 0.
_LANCZOS_G = 7.0
_LANCZOS_COEFFS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def log_gamma(x: keras.KerasTensor) -> keras.KerasTensor:
    """Natural log of the gamma function, differentiable, `keras.ops` only.

    Interface contract (shared helper, 2+ intended callers):

    - **Parameters**: ``x`` — a float tensor of any shape, strictly positive.
      The computation is elementwise and shape-preserving.
    - **Returns**: a tensor of the same shape and dtype as ``x`` holding
      ``log(Gamma(x))``.
    - **Failure mode**: it does not raise. For ``x <= 0`` the result is
      mathematically undefined here (the reflection formula is not
      implemented) and will be ``nan`` or ``inf``; callers must clamp. It is
      differentiable everywhere it is finite, which is the whole reason this
      exists.

    `keras.ops` has no ``lgamma`` in Keras 3.8 (verified: ``hasattr(keras.ops,
    'lgamma')`` is ``False``), and the only in-repo alternative was Python's
    ``math.lgamma`` on scalar constants, which is not differentiable and does
    not accept tensors. This is a Lanczos approximation rather than a call to
    ``tf.math.lgamma`` so the forward path stays backend-agnostic, matching the
    repository's standing preference for `keras.ops` over raw TensorFlow.

    Accuracy is limited by the input dtype, not by the series: pass float64
    tensors when a high-accuracy value is wanted.

    Args:
        x: Strictly positive float tensor.

    Returns:
        ``log(Gamma(x))``, elementwise.

    References:
        - Lanczos, 1964. A Precision Approximation of the Gamma Function.
          J. SIAM Numer. Anal. 1(1): 86-96.
    """
    dtype = getattr(x, "dtype", None) or keras.config.floatx()
    z = keras.ops.cast(x, dtype) - 1.0

    series = keras.ops.full_like(z, _LANCZOS_COEFFS[0])
    for i, coeff in enumerate(_LANCZOS_COEFFS[1:], start=1):
        series = series + coeff / (z + float(i))

    t = z + _LANCZOS_G + 0.5
    half_log_two_pi = 0.5 * float(np.log(2.0 * np.pi))
    return half_log_two_pi + (z + 0.5) * keras.ops.log(t) - t + keras.ops.log(series)

# ---------------------------------------------------------------------

def length(vectors) -> Any:
    """Compute length of capsule vectors.

    Args:
        vectors: Capsule vectors of shape [..., dim_capsule]

    Returns:
        Length of vectors with shape [...]
    """
    return keras.ops.sqrt(keras.ops.sum(keras.ops.square(vectors), axis=-1) + keras.config.epsilon())

# ---------------------------------------------------------------------

def pairwise_squared_distance(
    points: keras.KerasTensor,
    centers: keras.KerasTensor,
) -> keras.KerasTensor:
    """Squared Euclidean distance from each point to each center.

    Computes ``||points_i - centers_j||^2`` by direct subtraction (NOT the
    ``|a|^2 - 2 a·b + |b|^2`` expansion, which can go slightly negative before a
    sqrt due to floating-point cancellation). The result is structurally
    non-negative.

    # DECISION plan-2026-07-21T083606-47dc4421/D-002: shared home for the pairwise
    # squared-distance broadcast previously reimplemented inline in KMeansLayer
    # (_compute_distances), RBFLayer (kernel distance), and RBFLayer
    # (_compute_repulsion_loss). RBFLayer is deliberately outside the
    # BaseMixtureLayer hierarchy, so this lives here as a free function rather than
    # a mixin method. Do NOT reroute KMeansLayer._compute_repulsion_forces through
    # this — that path needs the difference VECTORS (for the repulsion direction),
    # so calling this would re-duplicate the subtraction.

    :param points: Tensor of shape ``(..., N, D)`` (a single leading ``N`` axis,
        optionally preceded by batch/time axes).
    :type points: keras.KerasTensor
    :param centers: Tensor of shape ``(M, D)``.
    :type centers: keras.KerasTensor
    :return: Squared distances of shape ``(..., N, M)``.
    :rtype: keras.KerasTensor
    """
    # (..., N, 1, D) - (M, D) -> (..., N, M, D), summed over D -> (..., N, M).
    diff = keras.ops.expand_dims(points, axis=-2) - centers
    return keras.ops.sum(keras.ops.square(diff), axis=-1)

# ---------------------------------------------------------------------

def resolve_training_factor(training: Any, dtype: Any = "float32") -> Any:
    """Resolve a Keras ``training`` flag into a graph-safe scalar factor.

    Layers that run a training-only side effect (an EMA state update, an
    ``add_loss`` term, ...) must gate it on ``training`` without coercing a
    symbolic tensor to a Python ``bool`` (which raises
    ``OperatorNotAllowedInGraphError`` under ``tf.function``). This helper
    returns one of three things so callers can both skip and mask correctly:

    - ``None`` — ``training`` is ``None`` or ``False``: the caller should SKIP
      the side effect entirely (no state writes at inference).
    - ``1.0`` — ``training`` is Python ``True``: run the side effect at full
      strength (multiplying by the Python float ``1.0`` is exact, so callers may
      take an unmasked fast path on ``isinstance(factor, float)``).
    - a 0/1 scalar tensor — ``training`` is a symbolic ``tf.Tensor`` (custom
      ``@tf.function`` train loop): the caller should MULTIPLY the side-effect
      delta / loss by this factor so a runtime-``False`` flag becomes a true
      no-op, all without any tensor-to-bool branching.

    :param training: The Keras ``call`` training flag (``None``, ``bool``, or a
        symbolic boolean tensor).
    :type training: Any
    :param dtype: Target dtype for the cast in the symbolic-tensor case.
    :type dtype: Any
    :return: ``None``, the float ``1.0``, or a scalar tensor.
    :rtype: Any
    """
    if training is None or training is False:
        return None
    if training is True:
        return 1.0
    return keras.ops.cast(training, dtype)

# ---------------------------------------------------------------------
