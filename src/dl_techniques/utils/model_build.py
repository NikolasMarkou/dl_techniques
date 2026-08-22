"""Materialize a subclassed model's sub-layers from a ``build()`` call.

Why this exists
---------------
A ``keras.Model`` subclass that creates its sub-layers in ``__init__`` and never
implements ``build()`` inherits ``keras.layers.Layer.build``, which does exactly
two things: it emits

    ``build()`` was called on layer 'X', however the layer does not have a
    ``build()`` method implemented and it looks like it has unbuilt state. This
    will cause the layer to be marked as built, despite not being actually
    built, which may cause failures down the line.

and then sets ``self.built = True``. The model is now *marked* built while
holding zero materialized state. Keras' own auto-build path
(``Layer._maybe_build``) does the right thing -- it traces ``call()`` on
symbolic inputs -- but that path is only taken when ``build()`` is *not*
overridden AND the model is being ``__call__``-ed. An explicit
``model.build(shape)``, and the ``build_from_config`` step of ``.keras``
deserialization, both land on the defaulted ``build`` instead.

:func:`materialize_sublayers` is that same symbolic trace, exposed so a model
can spend four lines to get the behaviour Keras' auto-build already has:

.. code-block:: python

    def build(self, input_shape):
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

Why a trace rather than a hand-written shape walk
-------------------------------------------------
The alternative is a per-model chain of ``sublayer.build(shape)`` /
``shape = sublayer.compute_output_shape(shape)``. That is a second, hand-
maintained encoding of the forward topology, and it drifts from ``call()``
silently -- the failure mode is a sub-layer that stops being built when the
architecture changes, which is the defect this function exists to remove.
Tracing ``call()`` cannot drift from ``call()``.

What it deliberately does NOT do
--------------------------------
It never falls back to an EAGER forward pass on concrete tensors. An eager
trace succeeds on strictly more models (measured: it materializes
``latent_gmm_registration``, ``lewm`` and ``video_jepa``, all of which the
symbolic trace cannot), but it also executes ``add_loss()`` calls and
``BatchNormalization`` updates for real, leaving accumulated losses and moved
moving statistics on a model that has merely been built. A model whose
``call()`` cannot be traced symbolically raises here rather than being built
with side effects; the caller is expected NOT to implement ``build()`` at all in
that case, so Keras' own eager auto-build stays in charge.
"""

from typing import Any

import keras

# ---------------------------------------------------------------------

__all__ = ["materialize_sublayers"]

# ---------------------------------------------------------------------


def _rebatch(input_shape: Any, batch_size: int) -> Any:
    """Replace the leading axis of every shape in ``input_shape``."""
    return keras.tree.map_shape_structure(
        lambda shape: (batch_size,) + tuple(shape[1:]), input_shape
    )


def materialize_sublayers(
        model: keras.Model,
        input_shape: Any,
        batch_size: int = 1,
) -> None:
    """Build every sub-layer of ``model`` by tracing ``call()`` symbolically.

    Interface contract (call sites: the ``build()`` method of every
    ``keras.Model`` subclass that creates its sub-layers in ``__init__``):
    ``model.call`` is invoked ONCE on :class:`keras.KerasTensor` placeholders
    derived from ``input_shape``. It is ``model.call``, never ``model(...)`` --
    the latter re-enters ``__call__`` -> ``_maybe_build`` -> ``build`` and
    recurses forever.

    Two shapes are tried, in order: ``input_shape`` exactly as given (normally
    carrying ``None`` in the batch axis), and then the same shapes with the
    batch axis replaced by ``batch_size``. The retry exists because some
    ``call()`` implementations do integer arithmetic on the batch dimension and
    raise ``TypeError`` on ``None``; a concrete batch changes no weight shape,
    so the retry cannot change what is materialized.

    :param model: The model to materialize. Must not already be built -- callers
        guard on ``self.built``.
    :param input_shape: A shape tuple, or any nest (dict / list / tuple) of shape
        tuples matching ``call()``'s first argument.
    :param batch_size: Batch size for the retry. Defaults to 1.
    :raises Exception: If neither trace succeeds, **the first attempt's own
        exception is re-raised unchanged** -- type, message and all. This is
        deliberately loud (a silent failure here reproduces exactly the "marked
        built while unbuilt" state the function exists to prevent) and
        deliberately NOT wrapped: a model's ``call()`` may raise a contract
        error of its own on a bad input, and callers -- including
        ``pytest.raises(ValueError, match=...)`` in the model's own test suite
        -- are entitled to see it. Wrapping was tried first and turned FNet's
        "Dictionary input must contain 'input_ids' key" and GPT-2's "exceeds
        max_seq_len" into an opaque ``RuntimeError``, breaking three tests whose
        subject is exactly that message.
    """
    first_error = None
    for shapes in (input_shape, _rebatch(input_shape, batch_size)):
        try:
            model.call(
                keras.tree.map_shape_structure(keras.KerasTensor, shapes)
            )
            return
        except Exception as exc:  # noqa: BLE001 -- re-raised below
            if first_error is None:
                first_error = exc

    raise first_error

# ---------------------------------------------------------------------
