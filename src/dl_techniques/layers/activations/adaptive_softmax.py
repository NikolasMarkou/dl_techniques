"""
Softmax with an entropy-driven temperature.

A softmax spreads its probability mass out as the number of classes grows.
With many classes the model struggles to express high confidence, which is
worst exactly where you want confidence: an out-of-distribution input with
more classes than the model saw in training. This layer sharpens the
distribution by dividing the logits by a temperature ``T < 1`` before the
final softmax, and it picks ``T`` per sample from the Shannon entropy of an
initial ``T = 1`` softmax.

The pipeline is: softmax at ``T = 1`` -> Shannon entropy ``H`` -> ``T = f(H)``
-> softmax at ``T``. ``f`` is a fixed degree-4 polynomial, clipped to
``[0, 1]`` and scaled into ``[min_temp, max_temp]``. It is bypassed entirely
(``T = 1.0``) whenever ``H <= entropy_threshold``.

Read the ``AdaptiveTemperatureSoftmax`` class docstring before using this.
The default polynomial is not monotonic in ``H``, so the layer does not
simply sharpen more as entropy rises, and with fewer than 10 classes it
cannot reach ``min_temp`` at all. The measured entropy-to-temperature table
is in that docstring.

References:
    - I. Drozdov et al., "Softmax is not enough (for sharp
      out-of-distribution)," 2024.

"""

import keras
from typing import Optional, Tuple, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AdaptiveTemperatureSoftmax(keras.layers.Layer):
    """Softmax whose temperature is chosen per sample from its entropy.

    Runs a plain softmax on the logits, measures the Shannon entropy ``H`` of
    the result, maps ``H`` to a temperature ``T``, then re-runs the softmax on
    ``logits / T``. A ``T`` below 1 sharpens the distribution. Output shape
    equals input shape and every row sums to 1.

    The layer owns no weights. ``min_temp``, ``max_temp`` and
    ``entropy_threshold`` are validated in ``__init__``; ``build`` only checks
    the input rank and that the class axis is static.

    **Architecture Overview:**

    .. code-block:: text

                   x  logits  [B, ..., C]
                      │
                      ├───────────────────────┐
                      ▼                       │
        ┌───────────────────────────┐         │
        │ p = softmax(x)            │         │
        └─────────────┬─────────────┘         │
                      ▼                       │
        ┌───────────────────────────┐         │
        │ p clipped to [eps, 1-eps] │         │
        │ H = -sum(p * log p)       │         │
        └─────────────┬─────────────┘         │
                      ▼                       │
        ┌───────────────────────────┐         │
        │ poly(H), degree 4, Horner │         │
        │ clip to [0, 1]            │         │
        │ T = min_temp + span*clip  │         │
        └─────────────┬─────────────┘         │
                      │                       │
              ┌───────┴───────┐               │
              │ H > threshold │ else          │
              ▼               ▼               │
        ┌───────────┐   ┌───────────┐         │
        │ T (above) │   │ T = 1.0   │         │
        └─────┬─────┘   └─────┬─────┘         │
              └───────┬───────┘               │
                      ▼                       │
              T = max(T, eps)                 │
                      │                       │
                      └───────────┬───────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │ softmax(x / T)            │
                    └─────────────┬─────────────┘
                                  ▼
                       y  [B, ..., C]

    ``span`` is ``max_temp - min_temp``. The right-hand lane carries the raw
    logits: the final softmax reads ``x``, not ``p``. ``H`` has shape
    ``[B, ..., 1]``, so ``T`` broadcasts back over the class axis.

    **Temperature by entropy, measured at the defaults.** Every number below
    holds only for ``polynomial_coeffs = [-1.791, 4.917, -2.3, 0.481,
    -0.037]``, ``min_temp=0.1``, ``max_temp=1.0`` and
    ``entropy_threshold=0.5``. Change any of them and the whole table moves.
    The three ``H`` values are where the clip engages, found by bisection on
    ``_compute_adaptive_temperature``. Only 2.21940 is a root of the
    polynomial itself; 0.91856 and 2.14702 are the two solutions of
    ``f(H) = max_temp``:

    .. code-block:: text

        H range                    T             effect
        H <= 0.5                   1.0           bypassed, plain softmax
        0.5 < H < 0.91856          0.218 .. 1.0  sharpens, less as H rises
        0.91856 <= H <= 2.14702    1.0           no sharpening
        2.14702 < H < 2.21940      1.0 .. 0.1    sharpens, more as H rises
        H >= 2.21940               0.1           maximum sharpening

    The fourth row is the one to watch. It is narrow, it is at the top of the
    reachable entropy range, and ``T`` falls fast across it. Measured: a
    uniform 9-class input has ``H = log(9) = 2.19722`` and gets
    ``T = 0.39728``, not 1.0.

    Three things follow from that table, and none of them is obvious from the
    formula:

    - The default polynomial is **not** monotonic in ``H``. More entropy does
      not mean more sharpening. Measured: a 2-class input with a logit gap of
      1.0 (``H = 0.58220``) gets ``T = 0.30520``, while the higher-entropy
      uniform 2-class input (``H = 0.69315``) gets ``T = 0.47388``.
    - ``T`` jumps at ``H = entropy_threshold``. Just below it ``T`` is 1.0;
      just above it ``T`` is 0.21807. That is a discontinuity, not a smooth
      handover, and it is a discontinuity in the sample's output.
    - With fewer than 10 classes the ``T = min_temp`` branch is unreachable,
      but sharpening is not. A 9-class distribution tops out at
      ``H = log(9) = 2.19722``, short of the 2.21940 crossing that would give
      ``T = 0.1``, yet already inside the fourth row: measured ``T = 0.39728``
      for a uniform 9-class input. Measured elsewhere: uniform over 10 classes
      (``H = 2.30259``) gives ``T = 0.1``; uniform over 5 classes
      (``H = 1.60944``) gives ``T = 1.0`` and passes through unchanged.

    So the layer sharpens hard at the diffuse extreme, which is the
    out-of-distribution case the paper targets and needs at least 10 classes,
    and is a no-op through the middle of the entropy range. Pass your own
    ``polynomial_coeffs`` if you want a different shape.

    :param min_temp: Lowest temperature the mapping can produce, so the
        sharpest possible output. Must be positive. Defaults to 0.1.
    :type min_temp: float
    :param max_temp: Highest temperature the mapping can produce, so the
        smoothest possible output. Must be positive and ``>= min_temp``.
        Defaults to 1.0.
    :type max_temp: float
    :param entropy_threshold: Below or at this entropy the mapping is
        bypassed and ``T = 1.0``. Must be non-negative. Defaults to 0.5.
    :type entropy_threshold: float
    :param eps: Clamp used twice: probabilities are clipped to
        ``[eps, 1-eps]`` before the log, and ``T`` is floored at ``eps``
        before the division. ``None`` means 1e-7.
    :type eps: Optional[float]
    :param polynomial_coeffs: Coefficients of ``f``, highest degree first.
        Any length works. ``None`` means the 5 default coefficients above,
        and so does ``[]`` -- the constructor uses ``or``, so an empty list
        is falsy and silently becomes the defaults. If you want a constant
        temperature, pass a one-element list.
    :type polynomial_coeffs: Optional[List[float]]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``min_temp <= 0``, ``max_temp <= 0``,
        ``min_temp > max_temp``, or ``entropy_threshold < 0``. Raised from
        ``__init__``. ``build`` raises ``ValueError`` separately for a rank-1
        input or an undefined last dimension.
    """

    def __init__(
        self,
        min_temp: float = 0.1,
        max_temp: float = 1.0,
        entropy_threshold: float = 0.5,
        eps: Optional[float] = None,
        polynomial_coeffs: Optional[List[float]] = None,
        **kwargs: Any
    ) -> None:
        """Validate the temperature bounds and store the configuration.

        :param min_temp: Lowest temperature the mapping can produce. Must be
            positive.
        :type min_temp: float
        :param max_temp: Highest temperature the mapping can produce. Must be
            positive and ``>= min_temp``.
        :type max_temp: float
        :param entropy_threshold: Below or at this entropy the mapping is
            bypassed. Must be non-negative.
        :type entropy_threshold: float
        :param eps: Numerical clamp. ``None`` means 1e-7.
        :type eps: Optional[float]
        :param polynomial_coeffs: Coefficients highest degree first. ``None``
            or ``[]`` means the defaults.
        :type polynomial_coeffs: Optional[List[float]]
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If ``min_temp <= 0``, ``max_temp <= 0``,
            ``min_temp > max_temp``, or ``entropy_threshold < 0``.
        """
        super().__init__(**kwargs)

        if min_temp <= 0.0:
            raise ValueError(f"min_temp must be positive, got {min_temp}")
        if max_temp <= 0.0:
            raise ValueError(f"max_temp must be positive, got {max_temp}")
        if min_temp > max_temp:
            raise ValueError(f"min_temp ({min_temp}) must be <= max_temp ({max_temp})")
        if entropy_threshold < 0.0:
            raise ValueError(f"entropy_threshold must be non-negative, got {entropy_threshold}")

        self.min_temp = min_temp
        self.max_temp = max_temp
        self.entropy_threshold = entropy_threshold
        self.eps = eps if eps is not None else 1e-7

        # Ordered highest degree first: [x^4, x^3, x^2, x^1, x^0]. `or` means
        # an empty list falls back to these defaults too, not to a zero
        # polynomial.
        self.polynomial_coeffs = polynomial_coeffs or [-1.791, 4.917, -2.3, 0.481, -0.037]

        logger.info(
            f"Initialized AdaptiveTemperatureSoftmax: min_temp={min_temp}, "
            f"max_temp={max_temp}, entropy_threshold={entropy_threshold}"
        )

    def build(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> None:
        """Check the input shape. No weights are created.

        The layer needs at least a batch axis and a class axis, and the class
        axis must be static because the entropy sum reduces over it.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input has fewer than 2 dimensions, or if
            its last dimension is ``None``.
        """
        if self.built:
            return

        if len(input_shape) < 2:
            raise ValueError(
                f"AdaptiveTemperatureSoftmax expects at least 2D input, "
                f"got shape {input_shape}"
            )

        if input_shape[-1] is None:
            raise ValueError(
                f"Last dimension (num_classes) must be defined, got shape {input_shape}"
            )

        super().build(input_shape)

    def _evaluate_polynomial(
            self,
            coeffs: List[float], x: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Evaluate a polynomial at ``x`` by Horner's method.

        Horner rewrites ``a_n x^n + ... + a_1 x + a_0`` as
        ``(...((a_n x + a_{n-1}) x + a_{n-2}) x + ... + a_1) x + a_0``, which
        needs ``n`` multiplies and no ``power`` op.

        The empty-``coeffs`` branch is dead when called through ``__init__``:
        ``polynomial_coeffs or [defaults]`` never stores an empty list. It is
        kept because this method takes ``coeffs`` as an argument.

        :param coeffs: Coefficients, highest degree first.
        :type coeffs: List[float]
        :param x: Tensor to evaluate at.
        :type x: keras.KerasTensor
        :return: Tensor of the same shape as ``x``.
        :rtype: keras.KerasTensor
        """
        if not coeffs:
            return keras.ops.zeros_like(x)

        result = keras.ops.full_like(x, coeffs[0])

        for coeff in coeffs[1:]:
            result = result * x + coeff

        return result

    def _compute_entropy(
            self,
            probabilities: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute Shannon entropy over the last axis.

        Computes ``H = -sum(p * log(p))`` on probabilities first clipped to
        ``[eps, 1-eps]``. The clip is what keeps ``log(0)`` out, so this is
        the entropy of the clipped values, not of ``probabilities`` -- they
        differ only where a probability is below ``eps`` (1e-7 by default).

        :param probabilities: Probability tensor, shape ``(..., C)``.
        :type probabilities: keras.KerasTensor
        :return: Entropy tensor, shape ``(..., 1)``. The class axis is kept
            so the result broadcasts back over it.
        :rtype: keras.KerasTensor
        """
        safe_probs = keras.ops.clip(probabilities, self.eps, 1.0 - self.eps)

        log_probs = keras.ops.log(safe_probs)
        entropy = -keras.ops.sum(safe_probs * log_probs, axis=-1, keepdims=True)

        return entropy

    def _compute_adaptive_temperature(
            self,
            entropy: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Map entropy to a temperature.

        Evaluates the polynomial at ``entropy``, clips the result to
        ``[0, 1]``, and rescales it into ``[min_temp, max_temp]``. Samples at
        or below ``entropy_threshold`` get 1.0 instead, chosen elementwise by
        ``where``, so both branches are computed for every sample.

        Note that ``T = 1.0`` is not necessarily inside
        ``[min_temp, max_temp]``. With ``max_temp < 1.0`` the bypass branch
        produces a temperature above the configured maximum.

        The mapping is not monotonic at the default coefficients, and it is
        discontinuous at the threshold. The class docstring has the measured
        table.

        :param entropy: Entropy tensor, shape ``(..., 1)``.
        :type entropy: keras.KerasTensor
        :return: Temperature tensor of the same shape as ``entropy``.
        :rtype: keras.KerasTensor
        """
        needs_adaptation = entropy > self.entropy_threshold

        poly_value = self._evaluate_polynomial(self.polynomial_coeffs, entropy)

        clamped_poly = keras.ops.clip(poly_value, 0.0, 1.0)

        temperature_range = self.max_temp - self.min_temp
        scaled_temp = self.min_temp + temperature_range * clamped_poly

        temperature = keras.ops.where(
            needs_adaptation,
            scaled_temp,
            keras.ops.ones_like(scaled_temp)
        )

        return temperature

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Softmax the logits, pick a temperature from the entropy, softmax
        again.

        The second softmax reads the original ``inputs``, not the first
        softmax's output, so the two are independent normalizations of the
        same logits.

        :param inputs: Logits, shape ``(..., C)``.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused here; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Probabilities of the same shape as ``inputs``, summing to 1
            over the last axis.
        :rtype: keras.KerasTensor
        """
        initial_probs = keras.ops.nn.softmax(inputs, axis=-1)

        entropy = self._compute_entropy(initial_probs)

        temperature = self._compute_adaptive_temperature(entropy)

        # min_temp is already validated positive, so this floor only matters
        # if a caller subclasses and produces a smaller temperature.
        safe_temperature = keras.ops.maximum(temperature, self.eps)
        scaled_logits = inputs / safe_temperature

        output_probs = keras.ops.nn.softmax(scaled_logits, axis=-1)

        return output_probs

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape tuple, unchanged.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(
            self
    ) -> Dict[str, Any]:
        """Return the config needed to rebuild the layer.

        ``eps`` and ``polynomial_coeffs`` are stored as resolved, so a
        reloaded layer keeps whatever the defaults were at save time rather
        than picking up new ones.

        :return: The base Layer config plus ``min_temp``, ``max_temp``,
            ``entropy_threshold``, ``eps`` and ``polynomial_coeffs``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'min_temp': self.min_temp,
            'max_temp': self.max_temp,
            'entropy_threshold': self.entropy_threshold,
            'eps': self.eps,
            'polynomial_coeffs': self.polynomial_coeffs,
        })
        return config

# ---------------------------------------------------------------------
