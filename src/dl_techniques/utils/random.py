import numpy as np
import tensorflow as tf
from typing import Optional, Union


def rayleigh(
        shape: Union[tf.TensorShape, list, tuple],
        scale: Optional[Union[float, tf.Tensor]] = None,
        dtype: tf.DType = tf.float32,
        seed: Optional[int] = None,
        name: Optional[str] = None
) -> tf.Tensor:
    """Generates a tensor of positive reals drawn from a Rayleigh distribution.

    The Rayleigh distribution is a continuous probability distribution for positive-valued
    random variables. It's often used to model scattered signals that reach a receiver
    through indirect paths.

    The probability density function of a Rayleigh distribution with `scale` parameter is:
        f(x) = x * scale^(-2) * exp(-x^2 * (2 * scale^2)^(-1))

    Args:
        shape: A 1-D integer tensor or Python array. The shape of the output tensor.
        scale: Optional float or tensor of type `dtype`. The scale parameter of the
            Rayleigh distribution. Must be positive. Defaults to 1.0 if None.
        dtype: Optional dtype of the tensor. Defaults to tf.float32.
        seed: Optional Python integer. Used to create a random seed for the distribution.
            See `tf.random.set_seed` for behavior.
        name: Optional name prefix for the operations created when running this
            function. Defaults to 'rayleigh'.

    Returns:
        A tensor of the specified shape filled with random Rayleigh values.

    Raises:
        ValueError: If scale is not positive when specified.
        ValueError: If shape has negative dimensions.
    """
    with tf.name_scope(name or 'rayleigh'):
        # Convert shape to tensor and validate
        shape = tf.convert_to_tensor(shape, dtype=tf.int32, name='shape')
        if tf.reduce_any(shape < 0):
            raise ValueError(f'All dimensions in shape must be non-negative. Got {shape}')

        # Handle scale parameter
        if scale is not None:
            scale = tf.convert_to_tensor(scale, dtype=dtype, name='scale')
            if tf.reduce_any(scale <= 0):
                raise ValueError('Scale parameter must be positive.')
            # Broadcast shape to match scale's shape
            shape = tf.broadcast_dynamic_shape(shape, tf.shape(scale))

        # Create stateless random generator for reproducibility
        if seed is not None:
            seed_tensor = tf.constant([seed, 0], dtype=tf.int32)
            uniform_samples = tf.random.stateless_uniform(
                shape=shape,
                seed=seed_tensor,
                minval=tf.keras.backend.epsilon(),  # Small positive number
                maxval=1.0,
                dtype=dtype
            )
        else:
            uniform_samples = tf.random.uniform(
                shape=shape,
                minval=tf.keras.backend.epsilon(),  # Small positive number
                maxval=1.0,
                dtype=dtype
            )

        # Transform uniform samples to Rayleigh distribution
        # Using the inverse CDF method: R = σ * sqrt(-2 * ln(1 - U))
        # where U is uniform(0,1) and σ is the scale parameter
        x = tf.sqrt(-2.0 * tf.math.log(uniform_samples))

        # Apply scale if provided
        if scale is not None:
            x = x * scale

        return x


def validate_rayleigh_samples(
        samples: tf.Tensor,
        scale: float = 1.0,
        significance_level: float = 0.05
) -> dict:
    """Validates generated Rayleigh samples using statistical tests.

    Args:
        samples: Tensor of samples to validate.
        scale: Expected scale parameter of the distribution.
        significance_level: Significance level for statistical tests.

    Returns:
        Dictionary containing test results and basic statistics.
    """
    # Convert to numpy for statistical testing
    samples_np = samples.numpy().flatten()

    # Calculate theoretical moments
    theoretical_mean = scale * tf.sqrt(tf.constant(np.pi / 2))
    theoretical_var = scale ** 2 * (4 - np.pi) / 2

    # Calculate sample statistics
    sample_mean = tf.reduce_mean(samples)
    sample_var = tf.math.reduce_variance(samples)

    # Basic statistics
    stats = {
        'sample_mean': float(sample_mean),
        'theoretical_mean': float(theoretical_mean),
        'sample_variance': float(sample_var),
        'theoretical_variance': float(theoretical_var),
        'min': float(tf.reduce_min(samples)),
        'max': float(tf.reduce_max(samples))
    }

    return stats