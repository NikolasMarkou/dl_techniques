"""
Tanh-approximate GELU as a serializable, registered activation *function*.

Motivation
----------
``keras.activations.gelu`` defaults to ``approximate=False`` — the exact
error-function form ``0.5 * x * (1 + erf(x / sqrt(2)))``
(``keras/src/activations/activations.py:339`` in the pinned Keras 3.8.0). Every
bare ``"gelu"`` string in this repository therefore resolves to the *exact*
form, because ``keras.activations.get`` looks the string up in
``ALL_OBJECTS_DICT`` and Keras registers **no** alias for the approximation
(no ``"gelu_approximate"``, no ``"gelu_pytorch_tanh"``).

Several reference implementations this repository ports specify the **tanh
approximation** instead:

- original BERT — ``google-research/bert`` ``modeling.py``::

      cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))

  https://github.com/google-research/bert/blob/master/modeling.py
- Gemma 3 — HuggingFace ``Gemma3TextConfig.hidden_activation`` defaults to
  ``"gelu_pytorch_tanh"``, i.e. ``partial(nn.functional.gelu, approximate="tanh")``.
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/configuration_gemma3.py

The two forms differ by up to ``max|exact - tanh| = 4.732e-04`` (attained at
``x ~= 2.699``, interior to the realistic post-LayerNorm activation range), so
the choice is **inference-changing**, not a training-only detail: it changes the
function every token passes through in every forward pass.

Why a registered function rather than a lambda
----------------------------------------------
A bare ``lambda`` or a ``functools.partial`` cannot be serialized by
``keras.activations.serialize``, which is what every FFN layer in this package
calls from ``get_config()``. A module-level function decorated with
``@keras.saving.register_keras_serializable`` serializes to its registered name
and deserializes back to the identical object, so ``.keras`` round-trips are
bit-exact.
"""

from typing import Any, Callable, Dict, Union

import keras
from keras import ops

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(
    package="dl_techniques.activations", name="gelu_tanh"
)
def gelu_tanh(x: keras.KerasTensor) -> keras.KerasTensor:
    """Tanh-approximate GELU.

    Computes ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))`` by
    delegating to ``keras.ops.gelu(x, approximate=True)``, which is the backend's
    own implementation of exactly that expression.

    Reference: https://github.com/google-research/bert/blob/master/modeling.py

    :param x: Input tensor of any shape.
    :type x: keras.KerasTensor
    :return: Tensor of the same shape with the tanh-approximate GELU applied.
    :rtype: keras.KerasTensor
    """
    return ops.gelu(x, approximate=True)


# ---------------------------------------------------------------------

#: Extra activation identifiers understood by :func:`resolve_activation` on top
#: of the stock Keras vocabulary. The keys are the spellings the upstream
#: references use, so a port can name its reference's activation verbatim.
_EXTENDED_ACTIVATIONS: Dict[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = {
    "gelu_tanh": gelu_tanh,
    "gelu_approximate": gelu_tanh,
    "gelu_pytorch_tanh": gelu_tanh,  # HuggingFace's spelling
}


def resolve_activation(
    identifier: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor], Dict[str, Any], None]
) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """Resolve an activation identifier, extending ``keras.activations.get``.

    Interface contract (this is a shared asset, called from model packages):

    - **Accepts** anything ``keras.activations.get`` accepts (``str``, callable,
      serialized ``dict``, ``None``), plus the extra string keys in
      :data:`_EXTENDED_ACTIVATIONS`.
    - **Returns** a callable ``f(tensor) -> tensor``. For the extended keys the
      returned callable is a *registered serializable* function, so a layer that
      stores it and emits ``keras.activations.serialize(...)`` from
      ``get_config()`` round-trips through ``.keras`` unchanged.
    - **Fails** exactly as ``keras.activations.get`` does — ``ValueError`` for an
      unknown identifier. It never silently falls back to a different function.

    :param identifier: Activation name, callable, or serialized config.
    :type identifier: Union[str, Callable, Dict[str, Any], None]
    :return: The resolved activation callable.
    :rtype: Callable[[keras.KerasTensor], keras.KerasTensor]
    :raises ValueError: If the identifier names no known activation.
    """
    if isinstance(identifier, str) and identifier in _EXTENDED_ACTIVATIONS:
        return _EXTENDED_ACTIVATIONS[identifier]
    return keras.activations.get(identifier)


# ---------------------------------------------------------------------
