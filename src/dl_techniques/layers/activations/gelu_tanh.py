"""
Tanh-approximate GELU as a serializable, registered activation *function*.

Motivation
----------
``keras.activations.gelu`` defaults to ``approximate=False``, the exact
error-function form ``0.5 * x * (1 + erf(x / sqrt(2)))``. In the pinned Keras
3.8.0, ``"gelu"`` is the only gelu key in ``ALL_OBJECTS_DICT`` — there is no
``"gelu_approximate"`` and no ``"gelu_pytorch_tanh"``. So every bare
``"gelu"`` string in this repository resolves to the exact form.

Several reference implementations this repository ports specify the **tanh
approximation** instead:

- original BERT — ``google-research/bert`` ``modeling.py``::

      cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))

  https://github.com/google-research/bert/blob/master/modeling.py
- Gemma 3 — HuggingFace ``Gemma3TextConfig.hidden_activation`` defaults to
  ``"gelu_pytorch_tanh"``, i.e. ``partial(nn.functional.gelu, approximate="tanh")``.
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/configuration_gemma3.py

The two forms are not interchangeable. They differ by up to
``max|exact - tanh| = 4.732e-04``, attained at ``x ~= 2.699`` — squarely
inside the range a post-LayerNorm activation lives in. Picking the wrong one
changes the function every token passes through on every forward pass, so it
is an inference-time difference, not a training-only detail.

Why a registered function rather than a lambda
----------------------------------------------
``keras.activations.serialize`` cannot serialize a ``lambda`` or a
``functools.partial``, and that is what every FFN layer in this package calls
from ``get_config()``. A module-level function decorated with
``@keras.saving.register_keras_serializable`` serializes to its registered
name and deserializes back to the identical object, so ``.keras`` round-trips
are bit-exact.
"""

import keras
from typing import Any, Callable, Dict, Union

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
def gelu_tanh(x: keras.KerasTensor) -> keras.KerasTensor:
    """Tanh-approximate GELU.

    Computes ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))``. It
    delegates to ``keras.ops.gelu(x, approximate=True)``, which is the
    backend's own implementation of that expression.

    The decorator registers this function under its own name, which is what
    makes it survive a ``.keras`` round-trip. Do not replace a use of it with
    ``lambda x: keras.ops.gelu(x, approximate=True)``: that lambda computes
    the same values but cannot be serialized.

    Reference: https://github.com/google-research/bert/blob/master/modeling.py

    :param x: Input tensor of any shape.
    :type x: keras.KerasTensor
    :return: Tensor of the same shape with the tanh-approximate GELU applied.
    :rtype: keras.KerasTensor
    """
    return keras.ops.gelu(x, approximate=True)


# ---------------------------------------------------------------------

#: Extra activation identifiers understood by :func:`resolve_activation` on top
#: of the stock Keras vocabulary. All three keys map to the same function. The
#: keys are the spellings the upstream references use, so a port can name its
#: reference's activation verbatim.
_EXTENDED_ACTIVATIONS: Dict[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = {
    "gelu_tanh": gelu_tanh,
    "gelu_approximate": gelu_tanh,
    # "gelu_pytorch_tanh" is HuggingFace's spelling.
    "gelu_pytorch_tanh": gelu_tanh,
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

    **Architecture Overview:**

    .. code-block:: text

                   identifier
                        │
                        ▼
        ┌───────────────────────────────┐
        │ str in _EXTENDED_ACTIVATIONS? │
        └───────────────┬───────────────┘
                        │
             ┌──────────┴──────────┐
             │ yes                 │ no
             ▼                     ▼
         gelu_tanh                keras.activations.get
         (registered,             (str, callable, dict
          serializes               or None; ValueError
          by name)                 on an unknown
                                   identifier)

    Only a ``str`` can take the left branch. Everything else goes right,
    including a callable that happens to be ``gelu_tanh`` itself.

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
