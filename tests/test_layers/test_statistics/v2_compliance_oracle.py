"""Shared §16.3 compliance instruments for ``dl_techniques.layers.statistics``.

This module deliberately carries **no** ``test_`` prefix, so pytest does not collect it
(guide §13.7.1: "Shared instrument — no ``test_`` prefix ... RED proofs live in a mirrored
``test_<name>.py``"). Its mirrored RED proof is
``test_the_v2_oracle_can_fail.py``: every helper here has been seen to reject a broken
subject, because an instrument that has never rejected anything is not an instrument.

What lives here, and which guide section each item implements:

============================== =========================================================
Helper                         Guide section
============================== =========================================================
``dtype_policy``               §13.2.6 precision arms, §13.7.4 session policy
``assert_value_round_trip``    §7.1 the round-trip test, on VALUES
``assert_weights_restored_``   §8.4 weight-value comparison, before the first call
``relative_weight_paths``      §8.3 enforcement (parity by relative ``w.path``)
``assert_build_parity``        §8.3 explicit build versus lazy build
``assert_gradients_reach_``    §13.2.2 gradient flow, per variable
``assert_forward_is_finite``   §10.2 / §16.3 ``ops.all(ops.isfinite(y))``
``assert_eager_matches_jit``   §13.2.7 graph and XLA equivalence
``weight_shape_signature``     §13.3.2 the value-knob instrument
``assert_value_knob_moves_``   §13.3.2 value knobs: outputs differ, signature identical
============================== =========================================================

Several subjects in this package are not plain single-input ``call`` layers: the
normalizing-flow classes take ``[data, context]``, ``ResidualACFLayer`` takes
``[predictions, targets]``, and ``AffineCouplingLayer`` has **no** ``call`` at all — its
public entry points are ``forward`` and ``inverse``. Every helper therefore accepts an
``apply`` callable, ``apply(layer, inputs) -> output``, defaulting to ``layer(inputs)``.
``MethodAdapter`` below turns a ``forward``-style entry point into a serializable layer so
that such a subject can still be put through a real ``.keras`` round trip.
"""

import os
import tempfile
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

# ---------------------------------------------------------------------
# types
# ---------------------------------------------------------------------

ArrayLike = Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]]
ApplyFn = Callable[[Any, Any], Any]

DTYPE_POLICIES: Tuple[str, ...] = ("float32", "mixed_float16", "float64")


# ---------------------------------------------------------------------
# a serializable adapter for entry points that are not `call`
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(
    package="dl_techniques.tests.layers.statistics.v2_compliance_oracle"
)
class MethodAdapter(keras.layers.Layer):
    """Expose a non-``call`` public method of ``inner`` as a normal layer ``call``.

    ``AffineCouplingLayer`` has no ``call``; its public entry points are ``forward`` and
    ``inverse``. Without an adapter such a layer cannot appear as a node in a functional
    model, so it can never be put through the §7.1 value round trip that §16.3 requires —
    and "the class has no ``call``" is not a reason to leave the round trip unmeasured.

    The adapter is registered and serializes ``inner`` through
    ``keras.saving.serialize_keras_object``, so a ``.keras`` round trip really does
    reconstruct the inner layer from its own ``get_config``. If the inner class had an
    incomplete ``get_config``, this adapter would surface it rather than hide it.

    :param inner: The layer whose method is being adapted.
    :param method_name: Name of the public method to route ``call`` to.
    :param output_index: If the method returns a tuple, take this element. ``None`` keeps
        the whole return value.
    """

    def __init__(
        self,
        inner: keras.layers.Layer,
        method_name: str,
        output_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.inner = inner
        self.method_name = method_name
        self.output_index = output_index

    def build(self, input_shape: Any) -> None:
        # Build exactly the tree `call` runs (§8.1): the inner layer, and nothing else.
        if not self.inner.built:
            self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        method = getattr(self.inner, self.method_name)
        args = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        out = method(*args, training=training)
        if self.output_index is not None:
            out = out[self.output_index]
        return out

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "inner": keras.saving.serialize_keras_object(self.inner),
                "method_name": self.method_name,
                "output_index": self.output_index,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "MethodAdapter":
        config = dict(config)
        config["inner"] = keras.saving.deserialize_keras_object(config["inner"])
        return cls(**config)


# ---------------------------------------------------------------------
# internal plumbing
# ---------------------------------------------------------------------


def _as_list(x: ArrayLike) -> List[np.ndarray]:
    """Normalize a single array or a sequence of arrays into a list."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _default_apply(layer: Any, inputs: Any) -> Any:
    return layer(inputs)


def _flatten_outputs(y: Any) -> List[np.ndarray]:
    """Flatten a tensor / tuple / list of tensors into a list of numpy arrays.

    ``NormalizingFlowLayer.call`` returns ``(z, log_det)``, so a helper that assumed one
    output would silently compare only the first half of that contract.
    """
    if isinstance(y, (list, tuple)):
        out: List[np.ndarray] = []
        for item in y:
            out.extend(_flatten_outputs(item))
        return out
    return [np.asarray(ops.convert_to_numpy(y))]


def _symbolic_inputs(sample_input: ArrayLike) -> Tuple[Any, List[Any]]:
    """Build ``keras.Input``s matching ``sample_input``.

    Returns ``(call_arg, input_list)``: ``call_arg`` is a bare tensor for a single-input
    subject and a list for a multi-input one, matching what the layer's ``call`` expects.
    """
    arrays = _as_list(sample_input)
    inputs = [
        keras.Input(shape=a.shape[1:], dtype=keras.backend.floatx()) for a in arrays
    ]
    call_arg = inputs if isinstance(sample_input, (list, tuple)) else inputs[0]
    return call_arg, inputs


def _tensor_args(sample_input: ArrayLike) -> Any:
    """Convert ``sample_input`` to the argument shape a layer's ``call`` expects."""
    arrays = [
        ops.convert_to_tensor(np.asarray(a, dtype=keras.backend.floatx()))
        for a in _as_list(sample_input)
    ]
    if isinstance(sample_input, (list, tuple)):
        return arrays
    return arrays[0]


def functional_model_from_layer(
    layer_factory: Callable[[], Any],
    sample_input: ArrayLike,
    apply: Optional[ApplyFn] = None,
) -> keras.Model:
    """Wrap one layer in a functional ``keras.Model`` driven by ``sample_input``'s shapes.

    A functional model, not a bare ``layer(x)`` call: §13.5 lists "direct ``layer.build``
    in a unit test" as an anti-pattern because it is structurally blind to the
    ``StatelessScope`` trap, and only a real model save exercises the serialization path.
    """
    apply = apply or _default_apply
    call_arg, inputs = _symbolic_inputs(sample_input)
    outputs = apply(layer_factory(), call_arg)
    # `inputs` must be a bare tensor for a single-input subject: a one-element LIST makes
    # the model expect a list at call time and Keras rejects a bare tensor with
    # "the structure of `inputs` doesn't match the expected structure".
    return keras.Model(inputs if len(inputs) > 1 else inputs[0], outputs)


# ---------------------------------------------------------------------
# 1. precision arms  (§13.2.6, §13.7.4)
# ---------------------------------------------------------------------


class DtypePolicyScope:
    """Set the Keras global dtype policy (and ``floatx``), then ALWAYS restore both.

    Implements guide §13.2.6 and the §13.7.4 rule that "any process-global setting (dtype
    policy, TF32, ``floatx``) is owned by one fixture that restores in ``finally`` and
    **asserts** the restoration".

    Two details are load-bearing:

    * The restore happens in ``finally``, so a failing test cannot leak ``mixed_float16``
      into every test that runs after it in the same process — a leaked policy turns a
      whole file's numbers into a different measurement without any test failing.
    * The ``float64`` arm also sets ``keras.backend.set_floatx("float64")``. §13.2.6:
      "a ``float64`` arm needs more than the policy: ``keras.Input`` still uses
      ``backend.floatx()``, so the graph rounds to float32 at the boundary" — the arm
      would be "a fake reading that agrees with float32 to eight digits".
    """

    def __init__(self, policy: str) -> None:
        self.policy = policy
        self._previous_policy: Optional[str] = None
        self._previous_floatx: Optional[str] = None

    def __enter__(self) -> str:
        self._previous_policy = keras.mixed_precision.global_policy().name
        self._previous_floatx = keras.backend.floatx()
        keras.mixed_precision.set_global_policy(self.policy)
        if self.policy == "float64":
            keras.backend.set_floatx("float64")
        return self.policy

    def __exit__(self, *exc: Any) -> None:
        try:
            keras.mixed_precision.set_global_policy(self._previous_policy)
            keras.backend.set_floatx(self._previous_floatx)
        finally:
            # Assert the restoration, per §13.7.4. A restore that silently failed is
            # exactly as damaging as no restore at all.
            assert keras.mixed_precision.global_policy().name == self._previous_policy, (
                f"dtype policy was not restored: "
                f"{keras.mixed_precision.global_policy().name!r} != "
                f"{self._previous_policy!r}"
            )
            assert keras.backend.floatx() == self._previous_floatx, (
                f"floatx was not restored: "
                f"{keras.backend.floatx()!r} != {self._previous_floatx!r}"
            )


@pytest.fixture(params=DTYPE_POLICIES)
def dtype_policy(request):
    """Set the Keras GLOBAL dtype policy for one test, then ALWAYS restore it.

    The §13.2.6 precision-arm fixture. Import it into a test module to get the three arms
    ``float32`` / ``mixed_float16`` / ``float64``; ``float32`` is the control that stops
    "fp16 is noisy" masquerading as a detected defect (§13.2.6 part 3).

    Restoration happens in ``finally`` and is ASSERTED, in ``DtypePolicyScope``. The
    ``float64`` arm also moves ``keras.backend.floatx``, because the policy alone leaves
    ``keras.Input`` on float32 and the arm becomes "a fake reading that agrees with
    float32 to eight digits".
    """
    with DtypePolicyScope(request.param) as policy:
        yield policy


def compute_dtype_atol(default: float = 1e-6) -> float:
    """A tolerance matched to the active compute dtype.

    §13.5 lists "tolerances below the dtype noise floor" as an anti-pattern: three such
    assertions "were never once green for their whole lifetime". float16 carries ~3
    decimal digits, so a 1e-6 bound under ``mixed_float16`` measures nothing but noise.
    """
    compute = keras.mixed_precision.global_policy().compute_dtype
    if compute == "float16":
        return 1e-2
    if compute == "bfloat16":
        return 5e-2
    return default


# ---------------------------------------------------------------------
# 2. the round trip, on VALUES  (§7.1)
# ---------------------------------------------------------------------


def assert_value_round_trip(
    layer_factory: Callable[[], Any],
    sample_input: ArrayLike,
    atol: float = 1e-6,
    apply: Optional[ApplyFn] = None,
) -> None:
    """A ``.keras`` round trip compared on VALUES.

    Implements guide §7.1. Three details are load-bearing and each is here on purpose:

    * ``rtol=0`` — ``np.testing.assert_allclose``'s default ``rtol=1e-7`` silently
      contributes to a nominally-``atol`` bound. §7.1 measured it contributing
      ``1.24e-05`` of a ``1.53e-05`` failure, which made the stated ``atol`` decorative.
    * ``training=False``, explicitly — "a bare ``model(x)`` is not inference; stochastic
      depth layers short-circuit only on ``training is False``".
    * Values, never shapes — "a shape-only round trip is satisfied by a model that
      restored **zero** weights".
    """
    model = functional_model_from_layer(layer_factory, sample_input, apply)
    args = _tensor_args(sample_input)

    original = _flatten_outputs(model(args, training=False))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "layer.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        restored = _flatten_outputs(loaded(args, training=False))

    assert len(original) == len(restored), (
        f"the round trip changed the number of outputs: "
        f"{len(original)} before, {len(restored)} after"
    )
    for index, (before, after) in enumerate(zip(original, restored)):
        np.testing.assert_allclose(
            before,
            after,
            atol=atol,
            rtol=0,
            err_msg=f"output {index} differs after a .keras round trip",
        )


# ---------------------------------------------------------------------
# 3. weight values, BEFORE the loaded model's first call  (§8.4)
# ---------------------------------------------------------------------


def assert_weights_restored_before_first_call(
    model_factory: Callable[[], keras.Model],
    sample_input: ArrayLike,
    tmp_path: Any,
) -> None:
    """Compare weight VALUES at ``atol=0.0`` before the loaded model's first forward pass.

    Implements guide §8.4. ``atol=0.0`` is correct because "restoration is a copy, not a
    computation". The comparison must happen **before** any forward pass on the loaded
    model: after one, "a ``build()``-only load path reads the same weight COUNT for the
    correct and the broken variant, because the gap has been filled with fresh random
    weights".

    The ``assert saved`` anti-vacuity check is mandatory (§13.4.8): with an empty weight
    list the ``zip`` below iterates zero times and the test passes on a model that
    restored nothing. Callers whose subject genuinely has no weights must ``xfail`` this
    item with the measured reason rather than let it pass vacuously.
    """
    model = model_factory()
    model(_tensor_args(sample_input), training=False)  # build the donor

    saved = [np.array(ops.convert_to_numpy(w)) for w in model.weights]
    assert saved, "the model has no weights to compare -- this check would be vacuous"
    saved_total = int(sum(int(np.prod(w.shape)) for w in model.weights))

    path = str(tmp_path / "restored.keras")
    model.save(path)
    loaded = keras.models.load_model(path)

    assert len(loaded.weights) == len(saved), (
        f"weight COUNT changed across the round trip: "
        f"{len(saved)} saved, {len(loaded.weights)} loaded"
    )
    # §8.4: "a weight COUNT invariant is blind to an internal-dimension change that
    # reshapes without adding or removing tensors. Assert the scalar parameter total too."
    loaded_total = int(sum(int(np.prod(w.shape)) for w in loaded.weights))
    assert loaded_total == saved_total, (
        f"scalar parameter total changed across the round trip: "
        f"{saved_total} saved, {loaded_total} loaded"
    )

    for before, weight in zip(saved, loaded.weights):
        np.testing.assert_allclose(
            before,
            np.array(ops.convert_to_numpy(weight)),
            atol=0.0,
            rtol=0,
            err_msg=f"weight {weight.path!r} was not restored",
        )


# ---------------------------------------------------------------------
# 4. build parity by relative `w.path`  (§8.3)
# ---------------------------------------------------------------------


def relative_weight_paths(model: Any) -> List[str]:
    """Sorted ``w.path`` values with the root segment stripped.

    Implements guide §8.3's ``_relative``. The root segment must go because Keras
    auto-increments generated names per instance, so two separately-constructed subjects
    read ``block/w`` versus ``block_1/w`` at the root and would never compare equal even
    when the trees are identical.
    """
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)


def assert_build_parity(
    build_fn: Callable[[], Any],
    input_shape: Any,
    sample_input: ArrayLike,
    apply: Optional[ApplyFn] = None,
) -> None:
    """An explicitly built subject and a lazily built one carry the same weight paths.

    Implements guide §8.3: ``build()`` must materialize precisely the sub-layer tree that
    ``call()`` runs. Under-build makes a reloaded model restore into nothing and fill the
    gap with fresh random weights, with nothing raising.

    Parity alone is blind to over-building — it passes if *both* paths build everything —
    so it is only half of the §8.3 pair; the sibling layout assertion belongs with the
    subject that has a ``None``/``False`` config.
    """
    apply = apply or _default_apply

    explicit = build_fn()
    explicit.build(input_shape)

    lazy = build_fn()
    apply(lazy, _tensor_args(sample_input))

    explicit_paths = relative_weight_paths(explicit)
    lazy_paths = relative_weight_paths(lazy)
    assert explicit_paths == lazy_paths, (
        "explicit build() and lazy build produced different weight trees\n"
        f"  only in explicit: {sorted(set(explicit_paths) - set(lazy_paths))}\n"
        f"  only in lazy    : {sorted(set(lazy_paths) - set(explicit_paths))}"
    )


# ---------------------------------------------------------------------
# 5. gradient flow, per variable  (§13.2.2)
# ---------------------------------------------------------------------


def assert_gradients_reach_every_trainable_weight(
    layer: Any,
    sample_input: ArrayLike,
    apply: Optional[ApplyFn] = None,
) -> None:
    """Every trainable variable receives a gradient that is non-``None`` AND non-zero.

    Implements guide §13.2.2. Both halves matter: a guard written as
    ``assert all(norm >= 0.0)`` reported green while **61 of 61** trainable weights had
    identically-zero gradients. Failures name ``var.path`` so the report says which weight
    is dead rather than "a gradient was None".

    The ``len(trainable_variables) > 0`` anti-vacuity assertion (§13.4.8) fires before the
    loop: with no trainable variables the loop body never runs and the check is a no-op
    that reports green forever.
    """
    apply = apply or _default_apply
    args = _tensor_args(sample_input)

    with tf.GradientTape() as tape:
        outputs = apply(layer, args)
        flat = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        loss = ops.sum(
            ops.stack([ops.mean(ops.square(ops.cast(o, "float32"))) for o in flat])
        )

    variables = list(layer.trainable_variables)
    assert len(variables) > 0, (
        f"{type(layer).__name__} exposed no trainable variables -- this gradient check "
        f"would be vacuous"
    )

    grads = tape.gradient(loss, variables)
    for var, grad in zip(variables, grads):
        assert grad is not None, f"no gradient for {var.path}"
        assert np.any(np.asarray(ops.convert_to_numpy(grad)) != 0.0), (
            f"all-zero gradient for {var.path}"
        )


# ---------------------------------------------------------------------
# 6. finiteness  (§10.2, §16.3)
# ---------------------------------------------------------------------


def assert_forward_is_finite(y: Any) -> None:
    """``ops.all(ops.isfinite(y))`` over every tensor in ``y``.

    Implements the §16.3 item "``ops.all(ops.isfinite(y))`` in every forward test".
    §10.2: a degenerate length that returns NaN instead of raising is invisible to a
    shape assertion, which is what most forward tests actually assert.
    """
    for index, array in enumerate(_flatten_outputs(y)):
        finite = bool(ops.convert_to_numpy(ops.all(ops.isfinite(array))))
        assert finite, (
            f"output {index} is not finite: "
            f"{int(np.sum(~np.isfinite(array)))} of {array.size} entries"
        )


# ---------------------------------------------------------------------
# 7. graph / XLA equivalence  (§13.2.7)
# ---------------------------------------------------------------------


class Tf32Scope:
    """Turn TF32 matmuls off for the body, then ALWAYS restore the previous setting.

    §13.6.1 calls TF32 "the default false model defect". On an Ampere-or-later GPU
    TensorFlow silently runs float32 matmuls through TF32, whose ~10-bit mantissa gives
    about 1e-3 relative accuracy. An eager-versus-XLA comparison then measures the TF32
    rounding rather than the layer.

    MEASURED in this worktree on the RTX 4090, ``assert_eager_matches_jit`` over the nine
    statistics subjects: with TF32 ON, four subjects disagree, max-abs
    ``5.42e-04`` / ``1.13e-03`` / ``1.03e-03`` / ``3.36e-03`` on outputs of order 1. With
    TF32 OFF, all nine agree inside ``1e-4``. The same run on CPU agrees inside ``1e-4``
    either way, which is why the failure appears only when the suite runs on a GPU.

    TF32 is process-global, so per §13.7.4 it is owned by one scope that restores in
    ``finally`` and ASSERTS the restoration -- leaving it off would silently change the
    numerics of every test that runs afterwards in the same process.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._previous: Optional[bool] = None

    def __enter__(self) -> bool:
        self._previous = tf.config.experimental.tensor_float_32_execution_enabled()
        tf.config.experimental.enable_tensor_float_32_execution(self.enabled)
        return self.enabled

    def __exit__(self, *exc: Any) -> None:
        try:
            tf.config.experimental.enable_tensor_float_32_execution(self._previous)
        finally:
            assert (
                tf.config.experimental.tensor_float_32_execution_enabled()
                == self._previous
            ), "TF32 execution was not restored"


@pytest.fixture
def tf32_disabled():
    """Fixture form of :class:`Tf32Scope`, for the XLA-equivalence arms."""
    with Tf32Scope(False) as state:
        yield state



def assert_eager_matches_jit(
    layer: Any,
    sample_input: ArrayLike,
    atol: float = 1e-5,
    apply: Optional[ApplyFn] = None,
) -> None:
    """An eager call and an XLA-compiled call agree, and both are finite.

    Implements guide §13.2.7: "an eager-only fix is not a fix". The compiled path is what
    ``model.fit`` uses by default (Keras compiles with ``jit_compile="auto"``), so a
    subject that only works eagerly is broken on the path users actually take.
    """
    apply = apply or _default_apply
    args = _tensor_args(sample_input)
    apply(layer, args)  # build once, outside the trace

    eager = _flatten_outputs(apply(layer, args))

    @tf.function(jit_compile=True)
    def traced(x: Any) -> Any:
        return apply(layer, x)

    compiled = _flatten_outputs(traced(args))

    assert len(eager) == len(compiled)
    for index, (a, b) in enumerate(zip(eager, compiled)):
        assert np.all(np.isfinite(b)), f"XLA output {index} is not finite"
        np.testing.assert_allclose(
            a.astype("float32"),
            b.astype("float32"),
            atol=atol,
            rtol=0,
            err_msg=f"eager and jit_compile=True disagree on output {index}",
        )


# ---------------------------------------------------------------------
# 8. the value-knob instrument  (§13.3.2)
# ---------------------------------------------------------------------


def weight_shape_signature(layer: Any) -> Tuple[Tuple[int, ...], ...]:
    """The layer's weight SHAPE signature, captured after a build.

    §13.3.2: capture after a forward pass; before one, an unbuilt subject has no weights
    and every config yields the same empty signature.
    """
    signature = tuple(tuple(w.shape) for w in layer.weights)
    return signature


def assert_value_knob_moves_output_not_shapes(
    builders: dict,
    sample_input: ArrayLike,
    apply: Optional[ApplyFn] = None,
    seed: int = 0,
) -> None:
    """A value knob must move the OUTPUT while leaving the weight-shape signature identical.

    Implements guide §13.3.2. This replaces the "shape-only knob sweep" anti-pattern
    listed in §13.5, which "is invariant under the knob being dead": sweeping a value knob
    and asserting the output shape is unchanged passes whether or not the knob is read.

    ``builders`` maps a knob value to a zero-argument factory. Bind loop variables as
    default arguments (``lambda v=v: ...``); a bare closure captures the LAST value for
    every entry and makes every builder identical.
    """
    apply = apply or _default_apply
    args = _tensor_args(sample_input)

    outputs: dict = {}
    signatures: dict = {}
    for key, build_fn in builders.items():
        keras.utils.set_random_seed(seed)
        layer = build_fn()
        outputs[key] = _flatten_outputs(apply(layer, args))
        signatures[key] = weight_shape_signature(layer)

    assert len(set(signatures.values())) == 1, (
        f"a value knob must not change the weight shapes, got {signatures}"
    )

    keys = list(outputs)
    for a, b in zip(keys, keys[1:]):
        differs = any(
            not np.allclose(x, y) for x, y in zip(outputs[a], outputs[b])
        )
        assert differs, f"value knob {a!r} vs {b!r} changed nothing in the output"


# ---------------------------------------------------------------------
# convenience for callers
# ---------------------------------------------------------------------


def forward_and_assert_finite(
    layer: Any,
    sample_input: ArrayLike,
    apply: Optional[ApplyFn] = None,
) -> Any:
    """Run one forward pass and assert every output tensor is finite."""
    apply = apply or _default_apply
    y = apply(layer, _tensor_args(sample_input))
    assert_forward_is_finite(y)
    return y
