"""Push a weight tensor toward a target Shannon entropy.

Provides :class:`EntropyRegularizer`, which penalizes the distance between a
weight vector's normalized Shannon entropy and a target, and
:func:`create_entropy_regularizer`, which builds one from a named preset.

Unlike L1/L2, this penalizes the *shape* of the weight distribution rather than
its magnitude. A low target makes a few weights dominate, which is a soft form
of feature selection. A high target spreads importance evenly, which gives more
distributed, fault-tolerant representations.

How the penalty is computed
---------------------------
Shannon entropy of a discrete distribution ``P = {p_1, ..., p_n}`` is::

    H(P) = - sum_i p_i * log(p_i)

Weights are not a probability distribution, so four steps get there:

1. **Probabilistic transformation**: ``p = softmax(|w|)``, which makes every
   ``p_i`` positive and the vector sum to 1. Absolute values make the result
   independent of weight signs.
2. **Entropy**: compute ``H(p)``.
3. **Normalization**: divide by the maximum possible entropy ``log(n)``, which
   maps the result into ``[0, 1]`` and makes the target independent of layer
   size. ``H_norm = H(p) / log(n)``.
4. **Penalty**: ``Loss = (H_norm - H_target)^2``. The quadratic gives a smooth
   landscape that drives the distribution toward the target entropy.

References
----------
-   Shannon, C. E. (1948). "A Mathematical Theory of Communication".
    *Bell System Technical Journal*.
-   Tishby, N., & Zaslavsky, N. (2015). "Deep learning and the information
    bottleneck principle". *IEEE Information Theory Workshop (ITW)*.
-   Yang, G., & Schoenholz, S. (2017). "Mean Field Residual Networks: On the
    Edge of Chaos". *Advances in Neural Information Processing Systems
    (NeurIPS)*.
"""

import keras
from keras import ops
from typing import Dict, Any, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_ENTROPY_STRENGTH: float = 0.01
DEFAULT_TARGET_ENTROPY: float = 0.7
DEFAULT_ENTROPY_AXIS: int = -1
DEFAULT_ENTROPY_EPSILON: float = 1e-10

# String constants for serialization
STR_STRENGTH: str = "strength"
STR_TARGET_ENTROPY: str = "target_entropy"
STR_AXIS: str = "axis"
STR_EPSILON: str = "epsilon"

# Predefined entropy targets
ENTROPY_LOW: float = 0.2
ENTROPY_MEDIUM: float = 0.5
ENTROPY_HIGH: float = 0.8

# Mode constants
MODE_LOW: str = "low"
MODE_MEDIUM: str = "medium"
MODE_HIGH: str = "high"


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.regularizers.entropy_regularizer")
class EntropyRegularizer(keras.regularizers.Regularizer):
    """Penalize the squared distance from a target normalized Shannon entropy.

    Controls how information is spread across a weight tensor: a low target
    concentrates it in a few weights, a high target spreads it out.

    **Penalty pipeline:**

    .. code-block:: text

        weights  [..., n, ...]        n = shape[axis]
             |
             v
        ┌──────────────────────────────┐
        │ p = softmax(|w|, axis)       │  signs discarded, sums to 1
        └──────────────┬───────────────┘
                       v
        ┌──────────────────────────────┐
        │ H = -sum(p * log(max(p,eps))) │  reduced over `axis`
        └──────────────┬───────────────┘
                       v
        ┌──────────────────────────────┐
        │ H_norm = H / log(n)          │  into [0, 1]
        └──────────────┬───────────────┘
                       v
        ┌──────────────────────────────┐
        │ mean((H_norm - target)^2)     │
        └──────────────┬───────────────┘
                       v
                  * strength
                       v
                    scalar

    **Target presets:**

    .. code-block:: text

        mode      target   effect on the weight distribution
        -------   ------   ---------------------------------
        'low'     0.2      concentrated, sparse-like weights
        'medium'  0.5      balanced distribution
        'high'    0.8      broadly distributed weights
        None      0.7      the module default

    :param strength: Scaling factor for the penalty. Larger values enforce the
        target more strongly. Must be non-negative.
    :type strength: float
    :param target_entropy: Normalized target entropy in ``[0, 1]``. Values near
        0 concentrate the weights; values near 1 spread them out.
    :type target_entropy: float
    :param axis: Axis along which entropy is computed.
    :type axis: int
    :param epsilon: Floor applied inside the logarithm for numerical stability.
        Must be positive.
    :type epsilon: float
    :param kwargs: Must be empty. ``keras.regularizers.Regularizer`` defines no
        ``__init__``, so any keyword forwarded here reaches ``object.__init__``
        and raises ``TypeError``.

    :ivar strength: The penalty scaling factor.
    :vartype strength: float
    :ivar target_entropy: The normalized target entropy.
    :vartype target_entropy: float
    :ivar axis: The reduction axis.
    :vartype axis: int
    :ivar epsilon: The logarithm floor.
    :vartype epsilon: float

    :raises ValueError: If ``strength`` is negative, ``target_entropy`` is
        outside ``[0, 1]``, or ``epsilon`` is not positive.

    Example:
        >>> # Medium entropy: balanced information distribution
        >>> regularizer = EntropyRegularizer(strength=0.01, target_entropy=0.5)
        >>> layer = keras.layers.Dense(64, kernel_regularizer=regularizer)

        >>> # Low entropy: more concentrated weights
        >>> regularizer = EntropyRegularizer(strength=0.02, target_entropy=0.2)
        >>> layer = keras.layers.Dense(64, kernel_regularizer=regularizer)
    """

    def __init__(
        self,
        strength: float = DEFAULT_ENTROPY_STRENGTH,
        target_entropy: float = DEFAULT_TARGET_ENTROPY,
        axis: int = DEFAULT_ENTROPY_AXIS,
        epsilon: float = DEFAULT_ENTROPY_EPSILON,
        **kwargs: Any
    ) -> None:
        """Validate and store the entropy target and penalty settings.

        :param strength: Non-negative scaling factor for the penalty.
        :type strength: float
        :param target_entropy: Target normalized entropy, in ``[0, 1]``.
        :type target_entropy: float
        :param axis: Axis along which entropy is computed.
        :type axis: int
        :param epsilon: Positive floor applied inside the logarithm.
        :type epsilon: float
        :param kwargs: Must be empty; see the class docstring.
        :raises ValueError: If any parameter is outside its valid range.
        :raises TypeError: If any keyword argument is supplied.
        """
        super().__init__(**kwargs)

        if strength < 0.0:
            raise ValueError(f"strength must be non-negative, got {strength}")
        if not (0.0 <= target_entropy <= 1.0):
            raise ValueError(f"target_entropy must be in [0, 1], got {target_entropy}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.strength = strength
        self.target_entropy = target_entropy
        self.axis = axis
        self.epsilon = epsilon

        logger.debug(
            f"Initialized EntropyRegularizer with strength={strength}, "
            f"target_entropy={target_entropy}, axis={axis}, epsilon={epsilon}"
        )

    def __call__(self, weights: Union[keras.KerasTensor, Any]) -> Union[keras.KerasTensor, Any]:
        """Compute the entropy penalty for a weight tensor.

        :param weights: Weight tensor to regularize.
        :type weights: tensor
        :return: The scalar penalty.
        :rtype: tensor
        """
        # Softmax over |w| turns the weights into a distribution whose shape
        # does not depend on their signs.
        weights_abs = ops.abs(weights)
        weights_normalized = ops.softmax(weights_abs, axis=self.axis)

        # Shannon entropy, with epsilon flooring the logarithm's argument.
        epsilon_tensor = ops.cast(self.epsilon, dtype=weights.dtype)
        safe_weights = ops.maximum(weights_normalized, epsilon_tensor)

        log_weights = ops.log(safe_weights)
        entropy_terms = ops.multiply(weights_normalized, log_weights)
        entropy = ops.negative(ops.sum(entropy_terms, axis=self.axis))

        # log(n) is the entropy of the uniform distribution, the maximum, so
        # dividing by it puts the result in [0, 1].
        n_weights = ops.cast(ops.shape(weights)[self.axis], dtype=weights.dtype)
        max_entropy = ops.log(n_weights)
        normalized_entropy = ops.divide(entropy, max_entropy)

        # Squared error grows quadratically with distance from the target.
        target_tensor = ops.cast(self.target_entropy, dtype=weights.dtype)
        deviation = ops.subtract(normalized_entropy, target_tensor)
        squared_deviation = ops.square(deviation)
        penalty = ops.mean(squared_deviation)

        strength_tensor = ops.cast(self.strength, dtype=weights.dtype)
        return ops.multiply(strength_tensor, penalty)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: A dict holding ``strength``, ``target_entropy``, ``axis`` and
            ``epsilon``.
        :rtype: dict
        """
        return {
            STR_STRENGTH: self.strength,
            STR_TARGET_ENTROPY: self.target_entropy,
            STR_AXIS: self.axis,
            STR_EPSILON: self.epsilon,
        }


# ---------------------------------------------------------------------

def create_entropy_regularizer(
    strength: float = DEFAULT_ENTROPY_STRENGTH,
    target_entropy: Optional[float] = None,
    mode: Optional[str] = None,
    **kwargs: Any
) -> EntropyRegularizer:
    """Build an :class:`EntropyRegularizer` from an explicit target or a preset.

    ``target_entropy`` wins when both it and ``mode`` are given.

    :param strength: Non-negative penalty scaling factor.
    :type strength: float
    :param target_entropy: Explicit normalized target entropy in ``[0, 1]``.
        ``None`` selects by ``mode`` instead.
    :type target_entropy: float or None
    :param mode: Preset target: ``'low'`` (0.2) for sparse, concentrated
        weights, ``'medium'`` (0.5) for a balanced distribution, ``'high'``
        (0.8) for widely distributed weights, or ``None`` for the module
        default of 0.7.
    :type mode: str or None
    :param kwargs: Forwarded to :class:`EntropyRegularizer`.
    :return: The configured regularizer.
    :rtype: EntropyRegularizer
    :raises ValueError: If ``strength`` is negative, ``target_entropy`` is
        outside ``[0, 1]``, or ``mode`` is not a recognized preset.

    Example:
        >>> # Low-entropy regularizer for concentrated weights
        >>> reg = create_entropy_regularizer(strength=0.01, mode='low')

        >>> # High-entropy regularizer for distributed weights
        >>> reg = create_entropy_regularizer(strength=0.02, mode='high')

        >>> # Explicit target
        >>> reg = create_entropy_regularizer(strength=0.01, target_entropy=0.6)
    """
    if strength < 0.0:
        raise ValueError(f"strength must be non-negative, got {strength}")

    if target_entropy is not None:
        if not (0.0 <= target_entropy <= 1.0):
            raise ValueError(f"target_entropy must be in [0, 1], got {target_entropy}")
        logger.debug(f"Creating EntropyRegularizer with custom target_entropy={target_entropy}")
        return EntropyRegularizer(
            strength=strength,
            target_entropy=target_entropy,
            **kwargs
        )

    mode_targets = {
        MODE_LOW: ENTROPY_LOW,
        MODE_MEDIUM: ENTROPY_MEDIUM,
        MODE_HIGH: ENTROPY_HIGH,
        None: DEFAULT_TARGET_ENTROPY
    }

    if mode not in mode_targets:
        valid_modes = [k for k in mode_targets.keys() if k is not None]
        raise ValueError(f"Invalid mode '{mode}'. Valid modes are: {valid_modes}")

    selected_target = mode_targets[mode]
    logger.debug(f"Creating EntropyRegularizer with mode='{mode}', target_entropy={selected_target}")

    return EntropyRegularizer(
        strength=strength,
        target_entropy=selected_target,
        **kwargs
    )


# ---------------------------------------------------------------------
