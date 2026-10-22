"""
Exact GELU and LeakyReLU: registered forms whose constants follow the input dtype.

Motivation
----------
TensorFlow 2.18 builds two activation constants in float32 even when the input
is float64 (measured, D-218 and plan-2026-10-14T062616-81938e81 F-11):

- ``keras.ops.gelu(x, approximate=False)``, ``keras.activations.gelu`` and the
  string ``"gelu"`` all reach ``tf.nn.gelu``, which divides by
  ``math_ops.cast(1.4142135623730951, features.dtype)``. ``math_ops.cast``
  turns the Python float into a float32 tensor first and then widens it, so a
  float64 GELU is off by up to ``5.02e-9`` against a float64 erf reference.
- ``keras.layers.LeakyReLU``, ``keras.ops.leaky_relu`` and
  ``keras.activations.leaky_relu`` reach the ``LeakyRelu`` op, whose slope is a
  float32 op attribute: a float64 slope of ``0.01`` applies as
  ``0.009999999776482582`` (``2.2e-8`` relative).

The forms here write the same expressions out with Python-float constants,
which the backend converts in the input's dtype:

- :func:`exact_gelu` is ``0.5 * x * (1 + erf(x / 1.4142135623730951))``. At
  float64 it is within ``8.9e-16`` of the reference; at float32 it is the same
  op sequence as ``tf.nn.gelu`` and bit-identical to it (300,000 points,
  asserted).
- :func:`exact_leaky_relu` is ``where(x > 0, x, negative_slope * x)``. At
  float64 it is exact; at float32 it is bit-identical to the ``LeakyRelu`` op
  for the slopes 0.01, 0.2 and 0.3, and its gradient at 0 is the slope, as
  stock's is.
- :class:`ExactLeakyReLU` is ``keras.layers.LeakyReLU`` with ``call`` routed
  through :func:`exact_leaky_relu`.

:data:`EXACT_ACTIVATIONS` maps the two stock names to these forms. The shared
resolvers (``common.resolve_activation`` and ``gelu_tanh.resolve_activation``)
consult it before ``keras.activations.get``, so a layer configured with the
string ``"gelu"`` or ``"leaky_relu"`` gets the exact form and keeps storing the
string.

Why registered, and why new names
---------------------------------
``keras.activations.serialize`` returns the bare function name for any function
whose ``__name__`` is one of its own built-ins. An unregistered function named
``gelu`` therefore serializes to ``"gelu"`` and reloads as STOCK gelu, a silent
revert. Registered under names that differ from stock (``exact_gelu``,
``exact_leaky_relu``, ``ExactLeakyReLU``), they serialize to a dict that
reloads the repo object. A stock layer given one of these callables stores that
dict in its own ``activation`` config entry.
"""

import keras
from typing import Any, Callable, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

_PACKAGE = "dl_techniques.layers.activations.exact_activations"


@register_dl_technique(_PACKAGE)
def exact_gelu(x: keras.KerasTensor) -> keras.KerasTensor:
    """Exact (error-function) GELU, computed in the input's dtype.

    ``0.5 * x * (1 + erf(x / sqrt(2)))``, with ``sqrt(2)`` written as the
    Python float ``1.4142135623730951`` and divided, not multiplied by its
    reciprocal: ``x * 0.7071067811865476`` is up to ``1.4e-6`` away from stock
    at float32, the division is bit-identical to it.

    :param x: Input tensor of any shape and floating dtype.
    :type x: keras.KerasTensor
    :return: Tensor of the same shape and dtype with GELU applied.
    :rtype: keras.KerasTensor
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-277
    # Do NOT delegate to keras.ops.gelu / keras.activations.gelu: tf.nn.gelu
    # builds sqrt(2) as a float32 constant cast up, so a float64 input comes
    # back off by 5.02e-9. Do NOT rename this function to `gelu`: an
    # unregistered or stock-named function serializes to the string "gelu"
    # and reloads as stock. See decisions.md D-277.
    return 0.5 * x * (1.0 + keras.ops.erf(x / 1.4142135623730951))


@register_dl_technique(_PACKAGE)
def exact_leaky_relu(
    x: keras.KerasTensor, negative_slope: float = 0.2
) -> keras.KerasTensor:
    """Leaky ReLU whose slope is applied in the input's dtype.

    ``where(x > 0, x, negative_slope * x)``. The default slope ``0.2`` is
    ``keras.activations.leaky_relu``'s. ``where`` keeps stock's gradient at
    ``x = 0`` (the slope); ``maximum(x, negative_slope * x)`` would give 1.

    :param x: Input tensor of any shape and floating dtype.
    :type x: keras.KerasTensor
    :param negative_slope: Slope for ``x <= 0``.
    :type negative_slope: float
    :return: Tensor of the same shape and dtype.
    :rtype: keras.KerasTensor
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-277
    # Do NOT call keras.ops.leaky_relu: the LeakyRelu op carries its slope as
    # a float32 attribute (0.01 applies as 0.009999999776482582 at float64).
    # Do NOT use maximum(x, a * x): its gradient at 0 is 1, stock's is a.
    return keras.ops.where(x > 0, x, negative_slope * x)


@register_dl_technique(_PACKAGE)
class ExactLeakyReLU(keras.layers.LeakyReLU):
    """``keras.layers.LeakyReLU`` computed by :func:`exact_leaky_relu`.

    Same constructor (default slope ``0.3``), config and masking as the stock
    layer; only ``call`` changes. A layer on the default float32 policy still
    casts a float64 input down to float32 before ``call``, so construct it with
    the owning layer's ``dtype``.

    :param negative_slope: Slope for ``x <= 0``. Defaults to ``0.3``.
    :type negative_slope: float
    :param kwargs: Base ``keras.layers.Layer`` arguments (``name``, ``dtype``).
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply :func:`exact_leaky_relu` with this layer's slope.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :return: Tensor of the same shape and dtype.
        :rtype: keras.KerasTensor
        """
        return exact_leaky_relu(inputs, negative_slope=self.negative_slope)


# ---------------------------------------------------------------------

#: The repo table both shared resolvers consult before ``keras.activations.get``.
#: Interface contract: keys are the stock Keras names whose TensorFlow kernels
#: round a float64 constant to float32; values are the registered exact forms,
#: each callable as ``f(tensor) -> tensor`` with stock's default arguments. Any
#: other name is absent, and a resolver falls through to Keras for it.
EXACT_ACTIVATIONS: Dict[str, Callable[..., Any]] = {
    "gelu": exact_gelu,
    "leaky_relu": exact_leaky_relu,
}

# ---------------------------------------------------------------------
