"""
Mish, a smooth non-monotonic activation, plus a saturating variant.

Mish is ``f(x) = x * tanh(softplus(x))``, where ``softplus(x) = log(1 + e^x)``.
Like Swish it is a self-gate: the input is multiplied by a smooth function of
itself rather than passed through a hard threshold. ``softplus`` maps every
input to a positive number with no kink at zero, and ``tanh`` squashes that
into ``[0, 1)``, giving a smooth saturating gate.

What that buys you:

- **Unbounded above.** As ``x`` grows, ``tanh(softplus(x)) -> 1`` and
  ``f(x) ~ x``, so positive inputs do not saturate, as with ReLU.
- **Bounded below.** Measured minimum: ``f(-1.19243) = -0.30884``. It never
  goes lower.
- **Smooth everywhere.** Infinitely differentiable, so no kink at zero.
- **Non-monotonic.** It dips negative for small negative inputs before
  climbing back toward its lower bound.

This module also ships :class:`SaturatedMish`, which caps large positive
outputs by blending Mish toward a constant. Read its class docstring before
using it: it overshoots before it settles, and the module-level
:func:`saturated_mish` function has a default that does not match the layer.

References:
    - Misra, D. (2019). "Mish: A Self Regularized Non-Monotonic Neural
      Activation Function."

"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------
# Standalone activation functions
# ---------------------------------------------------------------------


def mish(inputs: keras.KerasTensor) -> keras.KerasTensor:
    """Compute Mish: ``f(x) = x * tanh(softplus(x))``.

    The gate ``tanh(softplus(x))`` lies in ``[0, 1)``. Measured minimum of
    the whole function: ``f(-1.19243) = -0.30884``. At large negative inputs
    ``softplus`` underflows in float32 and the gate hits exactly 0: measured,
    ``mish(-100.0)`` returns ``-0.0``, not ``NaN``.

    :param inputs: Input tensor of any shape.
    :type inputs: keras.KerasTensor
    :return: Tensor of the same shape as ``inputs``.
    :rtype: keras.KerasTensor
    """
    softplus = keras.ops.softplus(inputs)
    tanh_softplus = keras.ops.tanh(softplus)
    return inputs * tanh_softplus


def saturated_mish(
        inputs: keras.KerasTensor,
        alpha: float = 3.0,
        beta: float = 0.5,
        mish_at_alpha: float = 1.0
) -> keras.KerasTensor:
    """Blend Mish toward a constant above a threshold.

    Computes ``mish(x) * (1 - blend) + mish_at_alpha * blend``, where
    ``blend = sigmoid((x - alpha) / beta)``. Below ``alpha`` the blend is near
    0 and the result is nearly plain Mish. Above it the blend goes to 1 and
    the result settles at ``mish_at_alpha``.

    Pass ``mish_at_alpha=mish(alpha)`` if you want the two regions to meet
    smoothly. The default of 1.0 does **not** do that: with the other
    defaults, ``mish(3.0) = 2.986535``, so calling this function with only
    ``inputs`` saturates toward 1.0 while :class:`SaturatedMish` saturates
    toward 2.986535. The two are different functions. The layer computes the
    right value for you; the bare function does not.

    :param inputs: Input tensor of any shape.
    :type inputs: keras.KerasTensor
    :param alpha: Input value at which the blend factor is 0.5.
    :type alpha: float
    :param beta: Transition width. Smaller means a sharper handover.
    :type beta: float
    :param mish_at_alpha: The value the output settles at for large inputs.
        Meant to be ``mish(alpha)``, precomputed so it is not recomputed per
        call.
    :type mish_at_alpha: float
    :return: Tensor of the same shape as ``inputs``.
    :rtype: keras.KerasTensor
    """
    tmp_mish = mish(inputs)

    blend_factor = keras.ops.sigmoid((inputs - alpha) / beta)

    return tmp_mish * (1.0 - blend_factor) + mish_at_alpha * blend_factor


# ---------------------------------------------------------------------
# Keras layer implementations
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Mish(keras.layers.Layer):
    """Mish activation: ``f(x) = x * tanh(softplus(x))``.

    A self-gate. The input is multiplied by ``tanh(softplus(x))``, a smooth
    gate in ``[0, 1)``. The result is smooth everywhere, non-monotonic,
    unbounded above (``f(x) ~ x`` for large positive ``x``) and bounded below
    at a measured minimum of ``-0.30884``, reached at ``x = -1.19243``. Output
    shape equals input shape.

    The layer owns no weights and takes no arguments. It calls the
    module-level :func:`mish` function.

    **Architecture Overview:**

    .. code-block:: text

                             x  [B, ..., F]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │ identity: x           │       │ tanh(softplus(x))     │
        └───────────┬───────────┘       └───────────┬───────────┘
                    │                               │ in [0, 1)
                    └───────────────┬───────────────┘
                                    │  x * gate
                                    ▼
                             y  [B, ..., F]

    The left branch is the tensor itself, not a weight or a sub-layer. The
    gate reaches exactly 0 in float32 once ``softplus`` underflows: measured,
    ``Mish()(-100.0)`` returns ``-0.0``.

    :param kwargs: Additional keyword arguments passed to the Layer base class,
        such as ``name``, ``dtype``, ``trainable``, etc.

    References:
        - Misra, Diganta. "Mish: A self regularized non-monotonic neural
          activation function." arXiv preprint arXiv:1908.08681 (2019).
        - https://github.com/digantamisra98/Mish
    """

    def __init__(self, **kwargs: Any) -> None:
        """Create the layer. There is nothing to configure.

        :param kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply ``x * tanh(softplus(x))`` element-wise.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused here; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        return mish(inputs)

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

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to rebuild the layer.

        :return: The base Layer config. This layer adds nothing to it.
        :rtype: Dict[str, Any]
        """
        return super().get_config()

    def __repr__(self) -> str:
        """Return a short representation showing the layer name.

        :return: A string such as ``Mish(name='mish')``.
        :rtype: str
        """
        return f"Mish(name='{self.name}')"

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SaturatedMish(keras.layers.Layer):
    """Mish that flattens out above a threshold, with a smooth handover.

    Plain Mish grows without bound. This variant blends it toward a constant
    so large positive inputs stop growing. The blend is
    ``sigmoid((x - alpha) / beta)``, so below ``alpha`` the output is close to
    Mish and well above ``alpha`` it settles at ``mish_at_alpha``. At exactly
    ``x = alpha`` the blend is 0.5 and, because ``mish_at_alpha`` is computed
    as ``mish(alpha)``, the two blended terms are the same number, so the
    output matches plain Mish there. Measured at the defaults: both read
    2.986535. Output shape equals input shape.

    The layer owns no weights. ``alpha`` and ``beta`` are validated in
    ``__init__``, and ``mish_at_alpha`` is computed there once in numpy.

    **Architecture Overview:**

    .. code-block:: text

                             x  [B, ..., F]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │ Mish(x)               │       │ blend = sigmoid(      │
        │ = x*tanh(softplus(x)) │       │   (x - alpha) / beta )│
        └───────────┬───────────┘       └───────────┬───────────┘
                    │                               │ in (0, 1)
                    └───────────────┬───────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │ Mish(x) * (1 - blend)                         │
            │     + mish_at_alpha * blend                   │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
                             y  [B, ..., F]

    Both branches read the same ``x``. ``mish_at_alpha`` is a stored constant,
    not a tensor.

    :param alpha: Input value at which the blend factor is 0.5, so roughly
        where saturation starts. Must be greater than 0. Defaults to 3.0.
    :type alpha: float
    :param beta: Transition width. Smaller means a sharper handover from Mish
        to the flat region. Must be greater than 0. Defaults to 0.5.
    :type beta: float
    :param kwargs: Additional keyword arguments passed to the Layer base class,
        such as ``name``, ``dtype``, ``trainable``, etc.

    :raises ValueError: If ``alpha`` or ``beta`` is not greater than 0.

    Note:
        The output overshoots its own plateau before settling. Measured at the
        defaults ``alpha=3.0``, ``beta=0.5``: the plateau is 2.986535, but the
        maximum output is 3.127664 at ``x = 3.6365``, then it decays back to
        2.986535 by ``x = 10``. So this caps activations, but it is not
        monotonic and its bound is 3.127664, not 2.986535. Measured minimum
        over the same settings: -0.308095.
    """

    def __init__(
            self,
            alpha: float = 3.0,
            beta: float = 0.5,
            **kwargs: Any
    ) -> None:
        """Validate ``alpha`` and ``beta``, then precompute ``mish(alpha)``.

        :param alpha: Input value at which the blend factor is 0.5. Must be
            greater than 0.
        :type alpha: float
        :param beta: Transition width. Must be greater than 0.
        :type beta: float
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If ``alpha`` or ``beta`` is not greater than 0.
        """
        super().__init__(**kwargs)

        if alpha <= 0.0:
            raise ValueError(f"alpha must be greater than 0, got {alpha}")
        if beta <= 0.0:
            raise ValueError(f"beta must be greater than 0, got {beta}")

        self.alpha = float(alpha)
        self.beta = float(beta)

        # mish(alpha) is a constant, so compute it once here in numpy rather
        # than rebuilding it in the graph on every call. float32 is used so
        # the stored value matches what the float32 forward path would give.
        alpha_np = np.float32(self.alpha)
        softplus_alpha = np.log(1.0 + np.exp(alpha_np))
        tanh_softplus_alpha = np.tanh(softplus_alpha)
        self.mish_at_alpha = float(alpha_np * tanh_softplus_alpha)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply Mish blended toward ``mish_at_alpha`` above ``alpha``.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused here; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        return saturated_mish(
            inputs,
            alpha=self.alpha,
            beta=self.beta,
            mish_at_alpha=self.mish_at_alpha
        )

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

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to rebuild the layer.

        ``mish_at_alpha`` is not stored: ``__init__`` recomputes it from
        ``alpha``, so saving it would be a second copy of the same fact.

        :return: The base Layer config plus ``alpha`` and ``beta``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta
        })
        return config

    def __repr__(self) -> str:
        """Return a short representation showing the config and layer name.

        :return: A string such as
            ``SaturatedMish(alpha=3.0, beta=0.5, name='saturated_mish')``.
        :rtype: str
        """
        return f"SaturatedMish(alpha={self.alpha}, beta={self.beta}, name='{self.name}')"

# ---------------------------------------------------------------------
