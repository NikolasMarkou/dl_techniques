"""Shared helpers for the step-7 DiTXA guards.

**Why an activation helper exists at all.** A freshly built ``DiTXA`` predicts
the EXACT zero tensor: every block's adaLN ``Dense`` is zero in kernel and bias
(so all three gates are 0 and every block is the identity on ``x``) and the
final layer's projection is zero in kernel and bias too. That is correct
adaLN-Zero behaviour and it is also a trap for any output-level guard -- an
assertion of the form "these two outputs differ" is VACUOUSLY satisfiable, and
an assertion of the form "these two outputs are identical" is vacuously TRUE, at
initialisation. Every output-level arm in this directory therefore calls
:func:`activate` first and asserts non-degeneracy before it asserts anything
else.

The name of the module starts with ``_`` so pytest does not collect it.
"""

from typing import Any, Dict, Optional

import keras
import numpy as np


def np_(x: Any) -> np.ndarray:
    """Convert any backend tensor to NumPy."""
    return keras.ops.convert_to_numpy(x)


def activate(model: keras.Model, seed: int = 0) -> keras.Model:
    """Replace every all-zero trainable weight with a random one, in place.

    Targets exactly the adaLN-Zero population: each block's
    ``Dense(12 * hidden)`` kernel and bias, the final layer's ``Dense(2 *
    hidden)`` and its output projection, plus the zero-initialised biases of the
    attention and MLP projections. Non-trainable weights (the fixed positional
    table, the frequency ladders) are never touched.

    :param model: A BUILT model.
    :type model: keras.Model
    :param seed: Seed for the NumPy generator that draws the replacements.
    :type seed: int
    :return: The same model, mutated.
    :rtype: keras.Model
    """
    rng = np.random.default_rng(seed)
    for weight in model.weights:
        if not weight.trainable:
            continue
        value = np_(weight)
        if np.any(value != 0.0):
            continue
        weight.assign(
            rng.normal(scale=0.3, size=value.shape).astype(value.dtype)
        )
    return model


def batch(
    model: keras.Model,
    batch_size: int = 4,
    seed: int = 1234,
    direction: Optional[Any] = None,
    cond_mask: Optional[Any] = None,
) -> Dict[str, Any]:
    """A deterministic input dict for ``model``'s geometry.

    :param model: A ``DiTXA``; its ``input_size``/``in_channels`` set the shapes.
    :type model: keras.Model
    :param batch_size: Number of samples.
    :type batch_size: int
    :param seed: Seed for the NumPy generator.
    :type seed: int
    :param direction: Optional explicit ``(B,)`` direction flags; defaults to
        all-forward zeros.
    :param cond_mask: Optional ``(B,)`` mask; omitted from the dict when None.
    :return: The input dictionary.
    :rtype: Dict[str, Any]
    """
    rng = np.random.default_rng(seed)
    shape = (batch_size, model.input_size, model.input_size, model.in_channels)
    inputs = {
        "x_t": rng.normal(size=shape).astype("float32"),
        "t": rng.uniform(0.05, 0.95, size=(batch_size,)).astype("float32"),
        "y": rng.integers(0, model.num_classes, size=(batch_size,)).astype("int32"),
        "x_cond": rng.normal(size=shape).astype("float32"),
        "direction": (
            np.zeros((batch_size,), dtype="float32")
            if direction is None
            else np.asarray(direction, dtype="float32")
        ),
    }
    if cond_mask is not None:
        inputs["cond_mask"] = np.asarray(cond_mask, dtype="float32")
    return inputs
