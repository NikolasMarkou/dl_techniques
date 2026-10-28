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
with side effects.

What such a model should do instead
-----------------------------------
Not "nothing". Leaving ``build()`` unimplemented leaves Keras' defaulted
``Layer.build`` in charge, and that is the marked-built-while-unbuilt state
above -- ``model.build(shape)`` and ``.keras`` ``build_from_config`` both land
on it. Three models here (``latent_gmm_registration``, ``lewm``,
``video_jepa``) reach a raw-backend op or an ``add_loss()`` inside ``call()``
and CANNOT use this helper; each writes a ``build()`` that walks its own
weight-bearing sub-layers, reusing its own forward helpers wherever it can. The
hand-walk hazard named above is real and is answered with an assertion rather
than a promise: ``tests/test_models/test_the_explicit_build_materializes_the_
model.py`` pins that each such ``build()`` materializes exactly the population a
real forward call does, so drift fails loudly. Prefer this helper; hand-walk
only when the trace provably cannot run, and pin the result.

Two more (``qwen``, ``scunet``) DO use this helper, after coercing a ``None``
axis to a concrete probe extent with :func:`concretize_axes` -- their blocker is
the trace mechanism, not their design.

Functional subclasses and unknown keywords
------------------------------------------
A ``keras.Model`` subclass that builds its graph in ``__init__`` and hands it to
``super().__init__(inputs, outputs, **kwargs)`` reaches Keras 3.8's
``Functional.__init__(self, inputs, outputs, name=None, **kwargs)``, which pops
``trainable`` and DISCARDS every other keyword without a word (read in the
installed source; measured at plan-2026-10-14T062616-81938e81 step 4:
``SwinTransformer(..., head_config_preset='default')`` built without raising,
while a subclassed ``keras.Model`` refuses the same keyword with
``ValueError: Unrecognized keyword arguments``). A misspelled or deleted
argument is then silently ignored. :func:`refuse_unknown_model_kwargs` restores
the subclassed models' refusal for such a class, called on its ``**kwargs``
before ``super().__init__``.

Head overrides that ``get_config`` can write
--------------------------------------------
A model keeps the caller's ``head_config_overrides`` and returns them from
``get_config``. A live Keras object in them (``{"kernel_regularizer": L2(1e-4)}``)
saves in a ``.keras`` archive, because Keras serializes nested objects, but
``json.dumps(model.get_config())`` then raises ``TypeError`` long after the
mistake. :func:`plain_head_config_overrides` refuses it at construction, naming
the key.
"""

import json
from typing import Any, Dict, Mapping, Optional

import keras
import numpy as np

# ---------------------------------------------------------------------

__all__ = [
    "coerce_include_top",
    "concretize_axes",
    "materialize_sublayers",
    "plain_head_config_overrides",
    "refuse_input_shape",
    "refuse_unknown_model_kwargs",
    "require_channel_count",
    "validate_rates",
]

#: The keywords a functional ``keras.Model`` subclass may forward to
#: ``super().__init__``: ``name`` and ``trainable``, which Keras applies, and
#: ``dtype``, which every ``get_config`` emits (Keras' ``Functional`` does not
#: apply it), so ``from_config`` must be able to pass it back.
FUNCTIONAL_MODEL_KWARGS = frozenset({"name", "trainable", "dtype"})

# ---------------------------------------------------------------------


def _rebatch(input_shape: Any, batch_size: int) -> Any:
    """Replace the leading axis of every shape in ``input_shape``."""
    return keras.tree.map_shape_structure(
        lambda shape: (batch_size,) + tuple(shape[1:]), input_shape
    )


def concretize_axes(
        input_shape: Any,
        replacements: Mapping[int, int],
) -> Any:
    """Substitute a concrete extent for named ``None`` axes of a shape nest.

    Interface contract (call sites: ``Qwen3Next.build``, ``SCUNet.build``).
    Returns a NEW nest of the same structure; the input is not mutated. For
    every ``axis: value`` pair, an axis that is ``None`` becomes ``value``; an
    axis that already carries a concrete extent is left ALONE, and an axis a
    given shape is too short to have is ignored. Negative axes are not
    supported -- pass a non-negative index.

    This is a shape transform only. It deliberately does NOT live inside
    :func:`materialize_sublayers`: which axis may be substituted, and with what
    probe value, is a per-model fact (``Qwen3Next`` may use any ``seq_len``
    because no weight shape depends on it; ``SCUNet`` must use its own
    ``input_resolution``), and a blanket "retry with concrete axes" inside the
    shared helper would apply that reasoning to the 20 models for which it has
    not been checked. See ``decisions.md`` D-420.

    :param input_shape: A shape tuple, or any nest (dict / list / tuple) of
        shape tuples.
    :param replacements: Mapping of axis index to the extent to substitute when
        that axis is ``None``.
    :return: The nest with the requested ``None`` axes made concrete.
    :raises ValueError: If any axis index is negative.
    """
    # DECISION plan-2026-08-23T091307-9a110062/D-420
    # A SHAPE TRANSFORM, not a build strategy. Do NOT "simplify" this by moving
    # the substitution into `materialize_sublayers` as a third retry attempt:
    # the retry would then apply to all 25 models on the helper, and for a model
    # whose weight shapes DO depend on the substituted axis (a positional table
    # sized from the traced sequence length is the obvious case) it would build
    # the wrong tensor silently -- the failure this whole module exists to
    # prevent. Two call sites justify the shared helper; the JUDGEMENT stays at
    # each call site. See decisions.md D-420.
    for axis in replacements:
        if axis < 0:
            raise ValueError(
                f"concretize_axes needs non-negative axis indices, got {axis}"
            )

    def one(shape: Any) -> Any:
        out = list(shape)
        for axis, value in replacements.items():
            if axis < len(out) and out[axis] is None:
                out[axis] = value
        return tuple(out)

    return keras.tree.map_shape_structure(one, input_shape)

# ---------------------------------------------------------------------


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


def refuse_unknown_model_kwargs(kwargs: Mapping[str, Any], class_name: str) -> None:
    """Raise for any keyword a functional ``keras.Model`` subclass would drop.

    Interface contract (call sites: the ``__init__`` of each functional
    ``keras.Model`` subclass, on its own ``**kwargs``, before
    ``super().__init__(inputs=..., outputs=..., **kwargs)``): ``kwargs`` is
    inspected, never mutated. Keys in :data:`FUNCTIONAL_MODEL_KWARGS` pass.
    Any other key raises ``ValueError`` with the message a subclassed
    ``keras.Model`` gives, ``"Unrecognized keyword arguments passed to
    <class_name>: {...}"``, so the two kinds of model refuse alike.

    :param kwargs: The constructor's leftover keyword arguments.
    :param class_name: The model class name, quoted in the message.
    :raises ValueError: If ``kwargs`` holds a key outside
        :data:`FUNCTIONAL_MODEL_KWARGS`.
    """
    unknown = {k: kwargs[k] for k in kwargs if k not in FUNCTIONAL_MODEL_KWARGS}
    if unknown:
        raise ValueError(
            f"Unrecognized keyword arguments passed to {class_name}: {unknown}"
        )

# ---------------------------------------------------------------------


def plain_head_config_overrides(
    head_config_overrides: Optional[Mapping[str, Any]], class_name: str
) -> Optional[Dict[str, Any]]:
    """Return a copy of a model's ``head_config_overrides`` that ``get_config`` can write.

    Interface contract (call sites: the ``__init__`` of the vision classifiers that
    take ``head_config_overrides``, which store the result and return it from
    ``get_config``): the argument is inspected, never mutated. ``None`` or an empty
    mapping returns ``None``. Otherwise every value must be JSON data (a string, a
    number, a bool, ``None``, a list or dict of those, a tuple counting as a list):
    a regularizer or initializer is passed as its name or as the dict
    ``keras.regularizers.serialize(...)`` returns, never as the live object.

    :param head_config_overrides: The constructor argument.
    :param class_name: The model class name, quoted in the message.
    :return: ``dict(head_config_overrides)``, or ``None`` when it is empty.
    :raises ValueError: If a value is not JSON data; the message names the key, the
        value's type and the serialized form to pass instead.
    """
    if not head_config_overrides:
        return None
    for key, value in head_config_overrides.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{class_name}: head_config_overrides[{key!r}] is a "
                f"{type(value).__name__}, which get_config cannot write as JSON. "
                "Pass a name or a serialized dict instead, for example "
                "keras.regularizers.serialize(...)."
            ) from error
    return dict(head_config_overrides)

# ---------------------------------------------------------------------


def refuse_input_shape(kwargs: Mapping[str, Any], class_name: str, reason: str) -> None:
    """Raise for an ``input_shape`` or ``input_dim`` keyword a layer or model does not take.

    Interface contract (call sites, from ``grep -rn refuse_input_shape src``: ``RBFProtoNet``,
    ``CliffordRBFProtoNet``, ``UcaEncoder``, ``MambaLCT``, ``ConvUNextStem``, ``CliffordNet``
    and ``HierarchicalCliffordNet``, each in its constructor, on its own keyword arguments and
    before the base class constructor runs, in every layer or subclassed model that has no
    ``input_shape`` parameter): ``kwargs`` is inspected, never mutated. ``keras.layers.Layer.__init__`` pops both keywords for the
    sake of ``Sequential`` and only warns ("Do not pass an `input_shape`/`input_dim`
    argument to a layer"), so a caller written against an older signature that took an
    ``input_shape`` keeps constructing a model that ignores it. This refuses it, naming the
    keyword, the class and why the class takes none. Every other unknown keyword is already
    refused by Keras with ``ValueError: Unrecognized keyword arguments passed to <class>``.

    :param kwargs: The constructor's leftover keyword arguments.
    :param class_name: The class name, quoted in the message.
    :param reason: Why the class takes no shape, as a clause (``"it runs at any spatial size"``).
    :raises ValueError: If ``kwargs`` holds ``input_shape`` or ``input_dim``; the message names
        the keyword, the class and the value.
    """
    for key in ("input_shape", "input_dim"):
        if key in kwargs:
            raise ValueError(
                f"{class_name} takes no {key} ({reason}); got {key}={kwargs[key]!r}"
            )


def coerce_include_top(include_top: Any) -> bool:
    """Return ``include_top`` as a Python ``bool``, refusing anything else.

    Interface contract (call sites: the ``__init__`` of every vision classifier that
    takes ``include_top``, as its first statement): ``bool`` and ``numpy.bool_`` (the
    type every comparison on an array yields) are accepted and coerced, so
    ``get_config`` stays JSON-serializable. Everything else is refused, including
    ``0``/``1``, a 0-d array, ``None`` and a string: ``bool("no")`` is ``True`` and
    would silently build the head, and ``include_top`` was once the second
    positional parameter of CliffordNet, where ``bool(...)`` would have let a
    misbound ``CliffordNet(10, 64)`` through. The message prints the repr and the
    module-qualified type, because ``type(np.False_).__name__`` is ``"bool"`` under
    numpy 2 and a bare name would contradict the refusal.

    :param include_top: The value passed as ``include_top``.
    :return: ``bool(include_top)``.
    :raises TypeError: If ``include_top`` is not a ``bool`` or ``numpy.bool_``.
    """
    if not isinstance(include_top, (bool, np.bool_)):
        kind = type(include_top)
        raise TypeError(
            f"include_top must be a bool or numpy.bool_, got {include_top!r} "
            f"of type {kind.__module__}.{kind.__qualname__}"
        )
    return bool(include_top)


def validate_rates(**rates: float) -> None:
    """Refuse a dropout or drop-path rate outside ``[0, 1)``, naming the argument.

    Interface contract (call sites, from ``grep -rn validate_rates src``: the ``__init__``
    of ``CliffordNet``, ``HierarchicalCliffordNet``, ``RADConvNet``, ``RBFProtoNet``,
    ``CliffordRBFProtoNet``, ``SqueezeNetV1``, ``SqueezeNoduleNetV2`` and ``_ConvNeXtBase``,
    before any sub-layer is built): pass each rate under the name of the
    constructor argument it came from, ``validate_rates(dropout_rate=dropout_rate, drop_path_rate=...)``.
    Rates are checked in call order and the first bad one raises. Without this check
    the sub-layer that first meets the value raises with its own argument name and
    value (a ramped ``drop_path_rate`` of 2.0 reads "got 1.333333" from
    ``StochasticDepth``), and a negative ``drop_path_rate`` disables the layer
    silently, because every ramped rate is then negative.

    :param rates: ``argument name -> value`` for each rate to check.
    :raises ValueError: If a rate is outside ``[0, 1)`` (``nan`` included); the
        message names the argument and the value.
    """
    for name, value in rates.items():
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {value}")


def require_channel_count(input_shape: Any) -> None:
    """Refuse an ``input_shape`` whose last axis is unknown, naming the argument and the value.

    Interface contract (call sites, from ``grep -rn require_channel_count src``:
    ``load_pretrained_checkpoint``, before it builds an unbuilt model; the
    ``from_variant`` of ``CliffordNet`` and ``HierarchicalCliffordNet``, which builds
    whenever it is given a shape; and the constructors of ``SqueezeNetV1`` and
    ``SqueezeNoduleNetV2``, which build in ``__init__``): ``input_shape`` is the
    unbatched shape a channels-last model is built at. The spatial axes may be ``None``; the last axis may not, because the first
    convolution derives its kernel from it (``None % groups``, reported by Keras as
    "unsupported operand type(s) for %" with no mention of the shape). An empty shape is
    refused too. Raises nothing for a shape with a known channel count.

    :param input_shape: The shape without the batch axis.
    :raises ValueError: If the shape is empty or its last axis is ``None``; the message
        carries ``input_shape`` and the tuple.
    """
    shape = tuple(input_shape)
    if not shape or shape[-1] is None:
        raise ValueError(
            f"input_shape must give the channel count, its last axis, to build the model; "
            f"got input_shape={shape} (for example (None, None, 3))"
        )


# ---------------------------------------------------------------------
