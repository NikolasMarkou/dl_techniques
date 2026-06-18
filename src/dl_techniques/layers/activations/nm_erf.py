"""
NMErf: a localized, tuning-curve-like activation function.

This module implements **NMErf**, a novel element-wise activation that pairs a
``tanh`` sign carrier with a Gaussian "bump" gate, producing a sharply
localized response reminiscent of a biological neuron's tuning curve.

Foundational Mathematics and Intuition
--------------------------------------
The activation is defined as:

    NMErf(x) = tanh(x) * exp(1 - (x^3 - m)^2)

It is the product of two conceptually distinct components:

1.  **Sign carrier — ``tanh(x)``:** bounded in ``(-1, 1)``, this term sets the
    sign and gentle saturation of the output, behaving like an identity-ish
    response near the origin and flattening for large ``|x|``.
2.  **Gaussian bump gate — ``exp(1 - (x^3 - m)^2)``:** a downward parabola in
    ``x^3`` exponentiated into a bell curve. The gate peaks at value ``e``
    (~2.718) exactly when ``x^3 = m`` (i.e. ``x = m^(1/3)``) and decays toward
    ``0`` as the input moves away from that preferred value in either
    direction. This makes the neuron "tuned": it responds strongly only to
    inputs near a preferred magnitude ``m^(1/3)`` and is quiet elsewhere.

With the default ``m = 1.5`` the gate peaks at ``x = 1.5^(1/3) ~= 1.1447``.

Numerical Properties
--------------------
The exponent argument ``1 - (x^3 - m)^2`` is a downward parabola in ``x^3`` and
is therefore ``<= 1`` for **all** real ``x`` and ``m``. Consequently
``exp(1 - (x^3 - m)^2) <= e ~= 2.718`` always, and the forward pass **cannot
overflow** float32 (max ~3.4e38) regardless of how large ``|x|`` becomes. As
``|x| -> inf`` the squared term diverges, the gate vanishes, and the product
collapses cleanly to ``0`` — no NaN or Inf is produced for any finite input.

References:
    - Novel / personal activation function devised by Nikolas Markou (2026).
      Not previously published in the literature.
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# Standalone activation function
# ---------------------------------------------------------------------


def nm_erf(inputs: keras.KerasTensor, m: float = 1.5) -> keras.KerasTensor:
    """Compute the NMErf activation: ``tanh(x) * exp(1 - (x^3 - m)^2)``.

    The output is the product of a ``tanh`` sign carrier and a Gaussian bump
    gate that peaks (value ``e``) at ``x^3 = m`` and decays to ``0`` for inputs
    far from the preferred value ``m^(1/3)``. The exponent is ``<= 1`` for all
    real inputs, so the forward pass cannot overflow.

    :param inputs: Input tensor of any shape.
    :type inputs: keras.KerasTensor
    :param m: Center of the Gaussian bump in ``x^3`` space; the gate peaks at
        ``x = m^(1/3)``. Any finite float is valid. Defaults to ``1.5``.
    :type m: float
    :return: Output tensor with the same shape as ``inputs``.
    :rtype: keras.KerasTensor
    """
    # Gaussian bump gate in x^3 space: exp(1 - (x^3 - m)^2), always <= e.
    cube = ops.power(inputs, 3)
    gate = ops.exp(1.0 - ops.square(cube - m))
    # tanh sign carrier modulated by the localized gate.
    return ops.tanh(inputs) * gate


# ---------------------------------------------------------------------
# Keras layer implementation
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NMErf(keras.layers.Layer):
    """NMErf activation function layer.

    NMErf is an element-wise, localized "tuning-curve" activation defined as
    ``NMErf(x) = tanh(x) * exp(1 - (x^3 - m)^2)``. A ``tanh`` sign carrier is
    gated by a Gaussian bump in ``x^3`` space that peaks (value ``e``) at
    ``x = m^(1/3)`` and decays to ``0`` for inputs far from that preferred
    value, yielding a neuron-tuning-curve-like response. The single parameter
    ``m`` (default ``1.5``) is a plain configuration float, not a trainable
    weight, so the layer has no ``build`` step and adds no variables.

    The exponent ``1 - (x^3 - m)^2`` is ``<= 1`` for all real inputs, so the
    forward pass is provably overflow-free for any finite input.

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., features)
                │
                ├──────────────────────────┐
                │                          │
                ▼                          ▼
        ┌───────────────┐   ┌──────────────────────────────┐
        │   tanh(x)     │   │  gate =                      │
        │ (sign carrier)│   │   exp(1 - (x^3 - m)^2)       │
        └───────┬───────┘   └──────────────┬───────────────┘
                │                          │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │   tanh(x) * gate         │
                └────────────┬─────────────┘
                             ▼
        Output: (batch, ..., features)

    :param m: Center of the Gaussian bump in ``x^3`` space; the gate peaks at
        ``x = m^(1/3)``. Any finite float is valid. Defaults to ``1.5``.
    :type m: float
    :param kwargs: Additional keyword arguments passed to the ``Layer`` base
        class, such as ``name``, ``dtype``, ``trainable``, etc.

    References:
        - Novel / personal activation function devised by Nikolas Markou
          (2026). Not previously published in the literature.
    """

    def __init__(self, m: float = 1.5, **kwargs: Any) -> None:
        """Initialize the NMErf activation layer and store its configuration.

        :param m: Center of the Gaussian bump in ``x^3`` space; the gate peaks
            at ``x = m^(1/3)``. Any finite float is valid. Defaults to ``1.5``.
        :type m: float
        :param kwargs: Additional keyword arguments for the ``Layer`` base class.
        """
        super().__init__(**kwargs)

        # Store configuration parameter (plain float, not a trainable weight).
        # Cast to float for consistent serialization (mirrors SaturatedMish).
        self.m = float(m)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply NMErf: ``tanh(x) * exp(1 - (x^3 - m)^2)``.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training or inference mode. Not used
            in this activation layer but kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor with the same shape as ``inputs`` after applying NMErf.
        :rtype: keras.KerasTensor
        """
        return nm_erf(inputs, self.m)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to ``input_shape``.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration including the
            ``m`` parameter along with the parent class configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({"m": self.m})
        return config

    def __repr__(self) -> str:
        """Return string representation of the layer.

        :return: String representation including the ``m`` parameter and name.
        :rtype: str
        """
        return f"NMErf(m={self.m}, name='{self.name}')"

# ---------------------------------------------------------------------
