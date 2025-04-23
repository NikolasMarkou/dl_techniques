import tensorflow as tf
from typing import Union, Tuple, Callable

def range_from_bits(bits: Union[float, int]) -> Tuple[int, int]:
    """Compute the range of values based on the number of bits.

    For example, 2 bits would give a range of (-2, 1), while 1.58 bits would
    give a specialized range for ternary quantization.

    Args:
        bits: Number of bits (can be fractional)

    Returns:
        tuple: Min and max values (min_value, max_value)

    Raises:
        ValueError: If bits is not positive

    Examples:
        >>> range_from_bits(2)
        (-2, 1)
        >>> range_from_bits(1.58)
        (-1, 1)
    """
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")

    return (int(tf.math.ceil(-2**(bits-1))), int(tf.math.ceil(2**(bits-1)-1)))


def round_clamp(inputs: tf.Tensor,
                value_range: Tuple[int, int],
                lambda_: float = 1.0) -> tf.Tensor:
    """Round the tensor and clamp it to the specified range using straight-through estimator.

    This function performs quantization by rounding values and then clamping them to the
    specified range. It uses a straight-through estimator to allow gradients to flow
    through the quantization operation during backpropagation.

    Args:
        inputs: Input tensor to be quantized
        value_range: Tuple containing (min_value, max_value)
        lambda_: Scaling factor for the straight-through estimator

    Returns:
        tf.Tensor: Rounded and clamped tensor with gradient passing through

    Examples:
        >>> x = tf.constant([0.1, 0.9, 1.6, -0.5, -1.2])
        >>> round_clamp(x, (-1, 1))
        <tf.Tensor: shape=(5,), dtype=float32, numpy=array([ 0.,  1.,  1., -1., -1.], dtype=float32)>
    """
    # Validate inputs
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be positive, got {lambda_}")
    if value_range[0] > value_range[1]:
        raise ValueError(f"Invalid value_range: {value_range}, min should be <= max")

    # Round and clamp values
    rounded = tf.round(inputs)
    clamped = tf.clip_by_value(rounded, value_range[0], value_range[1])

    # Straight-through estimator: pass through gradients with scaling
    return inputs + lambda_ * tf.stop_gradient(clamped - inputs)


def sample(inputs: tf.Tensor,
           value_range: Tuple[int, int],
           lambda_: float = 1.0) -> tf.Tensor:
    """Sample a discrete tensor from the input tensor using straight-through estimator.

    For ternary values (-1, 0, 1), this implements a stochastic rounding approach.
    For other ranges, it falls back to round_clamp.

    Args:
        inputs: Input tensor to be quantized
        value_range: Tuple containing (min_value, max_value)
        lambda_: Scaling factor for the straight-through estimator

    Returns:
        tf.Tensor: Sampled tensor with gradient passing through

    Examples:
        >>> tf.random.set_seed(42)
        >>> x = tf.constant([0.1, 0.9, 1.6, -0.5, -1.2])
        >>> sample(x, (-1, 1))
        <tf.Tensor: shape=(5,), dtype=float32, numpy=array([ 0.,  1.,  1., -1., -1.], dtype=float32)>
    """
    # Validate inputs
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be positive, got {lambda_}")
    if value_range[0] > value_range[1]:
        raise ValueError(f"Invalid value_range: {value_range}, min should be <= max")

    # For non-ternary values, use round_clamp
    if value_range != (-1, 1):
        return round_clamp(inputs, value_range, lambda_)

    # For ternary values (-1, 0, 1), use stochastic sampling
    rand = tf.random.uniform(tf.shape(inputs), dtype=inputs.dtype)
    abs_inputs = tf.abs(inputs)
    sign = tf.sign(inputs)

    # Probabilistic quantization
    result = sign * tf.cast(abs_inputs >= rand, inputs.dtype)

    # Straight-through estimator
    return inputs + lambda_ * tf.stop_gradient(result - inputs)


def abs_max(inputs: tf.Tensor, keepdim: bool = False) -> tf.Tensor:
    """Compute the absolute maximum value of the input tensor.

    Args:
        inputs: Input tensor
        keepdim: Whether to keep the dimensions of the input tensor

    Returns:
        tf.Tensor: Absolute maximum value(s) of the input tensor

    Examples:
        >>> x = tf.constant([[-1.5, 2.0], [0.5, -3.0]])
        >>> abs_max(x)
        <tf.Tensor: shape=(), dtype=float32, numpy=3.0>
        >>> abs_max(x, keepdim=True)
        <tf.Tensor: shape=(2, 1), dtype=float32, numpy=array([[2.], [3.]], dtype=float32)>
    """
    if keepdim:
        return tf.reduce_max(tf.abs(inputs), axis=-1, keepdims=True)
    else:
        return tf.reduce_max(tf.abs(inputs))


def abs_mean(inputs: tf.Tensor, keepdim: bool = False) -> tf.Tensor:
    """Compute the absolute mean value of the input tensor.

    Args:
        inputs: Input tensor
        keepdim: Whether to keep the dimensions of the input tensor

    Returns:
        tf.Tensor: Absolute mean value(s) of the input tensor

    Examples:
        >>> x = tf.constant([[-1.5, 2.0], [0.5, -3.0]])
        >>> abs_mean(x)
        <tf.Tensor: shape=(), dtype=float32, numpy=1.75>
        >>> abs_mean(x, keepdim=True)
        <tf.Tensor: shape=(2, 1), dtype=float32, numpy=array([[1.75], [1.75]], dtype=float32)>
    """
    if keepdim:
        return tf.reduce_mean(tf.abs(inputs), axis=-1, keepdims=True)
    else:
        return tf.reduce_mean(tf.abs(inputs))


def _median(tensor: tf.Tensor, axis: int = -1, keepdims: bool = False) -> tf.Tensor:
    """Compute the median of a tensor using TensorFlow operations.

    Args:
        tensor: Input tensor
        axis: Axis along which to compute the median
        keepdims: Whether to keep the dimensions of the input tensor

    Returns:
        tf.Tensor: Median value(s) of the input tensor
    """
    # Sort the tensor along the specified axis
    tensor_sorted = tf.sort(tensor, axis=axis)
    size = tf.shape(tensor)[axis]
    mid = size // 2

    if size % 2 == 0:
        # Even number of elements, average the middle two
        idx1 = mid - 1
        idx2 = mid

        # Handle tensor slicing based on the axis
        slices1 = [slice(None)] * tensor.shape.rank
        slices1[axis] = idx1
        slices2 = [slice(None)] * tensor.shape.rank
        slices2[axis] = idx2

        val1 = tensor_sorted[tuple(slices1)]
        val2 = tensor_sorted[tuple(slices2)]
        result = (val1 + val2) / 2.0
    else:
        # Odd number of elements, take the middle one
        slices = [slice(None)] * tensor.shape.rank
        slices[axis] = mid
        result = tensor_sorted[tuple(slices)]

    if keepdims:
        # Reshape to keep dimensions
        if axis < 0:
            axis = tensor.shape.rank + axis
        shape = list(tensor.shape)
        shape[axis] = 1
        result = tf.reshape(result, shape)

    return result


def abs_median(inputs: tf.Tensor, keepdim: bool = False) -> tf.Tensor:
    """Compute the absolute median value of the input tensor.

    Args:
        inputs: Input tensor
        keepdim: Whether to keep the dimensions of the input tensor

    Returns:
        tf.Tensor: Absolute median value(s) of the input tensor

    Examples:
        >>> x = tf.constant([[-1.5, 2.0, 0.1], [0.5, -3.0, 0.2]])
        >>> abs_median(x)
        <tf.Tensor: shape=(), dtype=float32, numpy=1.05>
    """
    abs_inputs = tf.abs(inputs)
    if keepdim:
        return _median(abs_inputs, axis=-1, keepdims=True)
    else:
        return _median(abs_inputs)


def scale(inputs: tf.Tensor,
          value_range: Tuple[int, int],
          measure_fn: Callable[[tf.Tensor, bool], tf.Tensor],
          keepdim: bool,
          eps: float) -> tf.Tensor:
    """Scale the input tensor based on the measure and range.

    This function computes a scaling factor to map the input tensor
    to the specified value range.

    Args:
        inputs: Input tensor
        value_range: Tuple containing (min_value, max_value)
        measure_fn: Function that computes a representative value of the tensor
        keepdim: Whether to keep dimensions in the measure
        eps: Small value to avoid division by zero

    Returns:
        tf.Tensor: Scaling factor for the input tensor

    Examples:
        >>> x = tf.constant([[-1.5, 2.0], [0.5, -3.0]])
        >>> scale(x, (-1, 1), abs_max, False, 1e-5)
        <tf.Tensor: shape=(), dtype=float32, numpy=0.33333334>
    """
    # Validate inputs
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if value_range[0] > value_range[1]:
        raise ValueError(f"Invalid value_range: {value_range}, min should be <= max")

    # Compute the maximum value in the range
    max_range = max(abs(value_range[0]), abs(value_range[1]))

    # Compute the measure value, stopping gradient to avoid affecting the original computation
    measure_val = tf.clip_by_value(
        measure_fn(tf.stop_gradient(inputs), keepdim),
        eps,
        tf.float32.max
    )

    # Return the scaling factor
    return max_range / measure_val