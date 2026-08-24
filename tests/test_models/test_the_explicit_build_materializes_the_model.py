"""Guard: ``.build(shape)`` materializes the whole model, for the RD-1 class.

Plan ``plan-2026-08-22T035419-a11304c8``, step 4a-1. Root cause **RD-1** of the
R-038 warning sweep: **29 layer names / 98 node ids** emitting

    ``build()`` was called on layer 'X', however the layer does not have a
    ``build()`` method implemented and it looks like it has unbuilt state.

at ``keras/src/layers/layer.py:393``. Twenty of those names are
``keras.Model`` subclasses in ``src/`` and each gained a four-line ``build()``
delegating to ``dl_techniques.utils.model_build.materialize_sublayers``.

What was MEASURED, and what was not
-----------------------------------
The warning was a **contract violation without a present consequence**, and this
docstring says so rather than claiming a repaired weight loss. Before the fix,
every one of the 24 packages surveyed round-tripped through ``.keras`` **twice**
-- one round trip is blind, because ``load_model`` rebuilds from the saved
``input_shape`` and restores immediately -- with the weight count, the relative
weight-path set and the forward output all identical, ``max|delta|`` exactly
``0.000000e+00``, against a live relative perturbation. The only two non-zero
readings (``relgt`` 3.38e-02, ``sd3_mmdit`` 4.65e+00) were the INSTRUMENT: both
models sample at inference and their own call-to-call self-spread on a fixed
input, same weights, is 2.18e-02 and 4.07e+00 respectively. The after-fix table
is bit-identical to the before-fix table, including those two numbers.

What the fix DID change is the thing the warning names: ``.build(shape)`` alone
went from materializing **0** weights to materializing all of them (e.g.
``mobilenet`` 0 -> 267, ``pw_fnet`` 0 -> 100, ``power_mlp`` 0 -> 5).

The waiver is retired (plan ``plan-2026-08-23T091307-9a110062``)
-------------------------------------------------------------
``video_jepa``, ``latent_gmm_registration``, ``lewm``, ``qwen`` and ``scunet``
used to sit in an ``UNFIXED`` table here, pinned by name as un-materializable.
They are all five in :data:`FIXED` now (97 / 148 / 26 / 188 / 161 weights from
``.build(shape)`` alone, each equal to what a real forward call materializes),
and :data:`UNFIXED` is empty. The reasons the waiver gave were real failures but
none of them was a property of the model:

* ``qwen`` / ``scunet`` -- the blocker was the TRACE MECHANISM, not the design.
  ``materialize_sublayers`` calls ``.call`` directly on ``KerasTensor``
  placeholders, where ``ops.shape(x)[i]`` is the Python value ``None``; the
  functional path real callers use goes through ``Layer.__call__`` ->
  ``compute_output_spec``, where the same expression is a dynamic scalar tensor.
  Each model's ``build`` now substitutes a concrete probe extent for its own
  ``None`` axis and nothing else. The waiver's claim that this would break
  dynamic-extent callers is refuted by
  :func:`test_a_coerced_build_probe_does_not_pin_the_call_shape`, which builds
  at the fully-``None`` shape and then calls at two OTHER shapes.
* ``latent_gmm_registration`` / ``lewm`` / ``video_jepa`` -- ``call`` really
  cannot be traced (a raw ``tf.linalg.svd``; ``add_loss`` outside a tracking
  context), so each got a hand-written walk of its weight-bearing sub-layers.
  A hand walk is a second encoding of the forward topology and drifts silently,
  which is exactly what ``materialize_sublayers``' own docstring warns about;
  the mitigation is the second half of
  :func:`test_explicit_build_materializes_everything_a_call_does`, which asserts
  the built population equals the population after a real call, so drift fails
  loudly rather than quietly under-building.

One number the waiver carried did NOT reproduce: ``video_jepa`` "66/161" was
measured at **158/161** for the same failed-trace side effect on 2026-08-23.
It is recorded here as refuted rather than repeated.

See ``decisions.md`` D-013, D-420..D-425.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest
import keras

from .precision_arm_subjects import roundtrip_subject
from .roundtrip_instrument_oracle import _call, relative_path
from tests.optimizer_state import build_optimizer_state

#: Every package whose model class gained a ``build()`` in step 4a-1, with the
#: weight count ``.build(shape)`` alone MUST now materialize. Pinned exactly: a
#: package that loses its ``build()`` fires, and one whose weight population
#: changes fires too.
FIXED = {
    "cbam": 22,
    "distilbert": 16,
    "fnet": 15,
    "gemma": 15,
    "gpt2": 18,
    "ideogram4": 45,
    "latent_gmm_registration": 26,
    "lewm": 188,
    "mamba": 14,
    "masked_language_model": 23,
    "mobilenet": 267,
    "modern_bert": 13,
    "power_mlp": 5,
    "pw_fnet": 100,
    "qwen": 97,
    "relgt": 49,
    "resnet": 32,
    "scunet": 148,
    "sd3_mmdit": 124,
    "tabm": 5,
    "tree_transformer": 28,
    "video_jepa": 161,
    "vq_vae": 9,
    "vq_vae_rotation": 9,
}

#: RD-1 classes with a live waiver: ``build()`` deliberately NOT implemented.
#: **EMPTY, and that is the assertion** -- see
#: :func:`test_the_waiver_table_is_empty_and_every_retired_entry_has_a_build`.
#: A future class that genuinely cannot be materialized belongs here with its
#: measured reason; adding one re-arms the per-name arm automatically.
UNFIXED: dict = {}

#: The five names that used to live in :data:`UNFIXED`, mapped to the waiver
#: text that no longer holds. This is the anti-rot arm's subject list: with
#: :data:`UNFIXED` empty, "each unfixed class still inherits the base build" is
#: vacuously true, so the arm asserts the OPPOSITE for exactly these five --
#: each must now OVERRIDE ``build``. Deleting one of the five ``build()``
#: methods fails here by name, which is the property the old table provided.
RETIRED_WAIVER = {
    "latent_gmm_registration":
        "raw-tf op in call(); a KerasTensor cannot enter it",
    "lewm":
        "call() does integer arithmetic on the batch axis, then add_loss()",
    "qwen":
        "call() needs a concrete seq_len; real callers build at seq_len=None",
    "scunet":
        "call() does padding arithmetic on None spatial dims by design",
    "video_jepa":
        "same as lewm; also the only partially-built subject (66/161)",
}


def _input_shape(make_inputs):
    def one(value):
        return (None,) + np.asarray(value).shape[1:]
    sample = make_inputs()
    if isinstance(sample, dict):
        return {key: one(value) for key, value in sample.items()}
    if isinstance(sample, (list, tuple)):
        return [one(value) for value in sample]
    return one(sample)


def _paths(model):
    return sorted(relative_path(model, w) for w in model.weights)


def _shapes(model):
    return sorted(tuple(w.shape) for w in model.weights)


@pytest.mark.parametrize("name", sorted(FIXED))
def test_explicit_build_materializes_everything_a_call_does(name):
    """R-038 / RD-1, the arm that goes RED when a ``build()`` is reverted.

    Both halves are measured on ONE instance, so auto-name re-numbering between
    two instances (which ``RELOAD_PATH_DRIFT`` records for six of these
    packages) cannot make the comparison vacuous or flaky.
    """
    build, make_inputs, kwargs = roundtrip_subject(name)
    keras.utils.set_random_seed(0)
    model = build()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.build(_input_shape(make_inputs))
    unbuilt_state = [
        str(w.message) for w in caught
        if issubclass(w.category, UserWarning)
        and "does not have a `build()` method" in str(w.message)
    ]
    assert unbuilt_state == [], (
        f"{name}: Keras still reports unbuilt state after build(): "
        f"{unbuilt_state}")

    after_build = _paths(model)
    assert len(after_build) == FIXED[name], (
        f"{name}: build() materialized {len(after_build)} weights, "
        f"pinned at {FIXED[name]}")

    _call(model, make_inputs(), kwargs.get("call_fn"),
          bool(kwargs.get("training", False)))
    assert _paths(model) == after_build, (
        f"{name}: the forward call materialized weights build() did not: "
        f"{sorted(set(_paths(model)) - set(after_build))}")


@pytest.mark.parametrize("name", sorted(FIXED))
def test_two_keras_round_trips_preserve_the_weight_population(name):
    """TWO round trips, because one is blind.

    ``keras.models.load_model`` builds the fresh instance from the saved
    ``input_shape`` and restores into it immediately, which hides a lossy
    ``build()`` from any instrument that inspects only the first reload. The
    second cycle saves the RELOADED model, so a sub-layer that was never
    materialized on the way in cannot be written on the way out.

    Shapes rather than paths: six of these packages carry pinned reload path
    drift (``RELOAD_PATH_DRIFT`` in ``test_roundtrip_instrument_family.py``)
    from auto-named sub-layers, which is a naming fact, not a weight loss.
    """
    build, make_inputs, kwargs = roundtrip_subject(name)
    call_fn, training = kwargs.get("call_fn"), bool(kwargs.get("training", False))
    keras.utils.set_random_seed(0)
    model = build()
    inputs = make_inputs()
    _call(model, inputs, call_fn, training)
    donor = (len(model.weights), _shapes(model))

    with tempfile.TemporaryDirectory() as tmp:
        first = os.path.join(tmp, "a.keras")
        second = os.path.join(tmp, "b.keras")
        # R-038 / D-016: several of these builders COMPILE, and Keras allocates
        # the optimizer's slot variables lazily -- on the first gradient
        # application, which a round-trip test never performs. Saving first
        # would write an optimizer section holding only `iteration` and the
        # learning rate, and `load_model` would then warn and silently skip the
        # optimizer restore. `build_optimizer_state` allocates the slots the
        # archive is about to claim to contain; it moves no weight and consumes
        # no RNG draw, so the population assertions below are unaffected.
        build_optimizer_state(model)
        model.save(first)
        reloaded = keras.models.load_model(first)
        _call(reloaded, inputs, call_fn, training)
        assert (len(reloaded.weights), _shapes(reloaded)) == donor, (
            f"{name}: round trip 1 changed the weight population")

        build_optimizer_state(reloaded)
        reloaded.save(second)
        again = keras.models.load_model(second)
        _call(again, inputs, call_fn, training)
        assert (len(again.weights), _shapes(again)) == donor, (
            f"{name}: round trip 2 changed the weight population")


def test_the_cliffordnet_lm_build_materializes_all_nineteen():
    """``CliffordNetLM`` is not a ``roundtrip_subject``; it is charged here.

    MEASURED: 19 weights after one call, **1 after ``.build()`` alone** before
    the fix -- 95% of the model, the worst ratio the prior plan found (see
    ``tests/test_models/test_cliffordnet/test_lazy_build_contract.py``). Now 19.
    """
    from dl_techniques.models.vision.cliffordnet.lm import CliffordNetLM

    keras.utils.set_random_seed(0)
    model = CliffordNetLM(vocab_size=16, channels=8, depth=1,
                          max_seq_length=8, shifts=(1,))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.build((None, 8))
    assert [w for w in caught
            if issubclass(w.category, UserWarning)
            and "does not have a `build()` method" in str(w.message)] == []
    assert len(model.weights) == 19


@pytest.mark.parametrize("name", sorted(UNFIXED))
def test_a_waived_subject_is_pinned_by_name(name):
    """The waiver is not a blanket: each unfixed class is named with its reason.

    A class that GAINS a working ``build()`` fails here and must be moved into
    :data:`FIXED`, so the waiver cannot outlive its own justification.

    :data:`UNFIXED` is currently EMPTY, so this arm collects zero cases. It is
    kept live rather than deleted because it is the thing that makes adding a
    new waiver cost something. The emptiness itself is asserted by
    :func:`test_the_waiver_table_is_empty_and_every_retired_entry_has_a_build`,
    which is NOT parameterized over ``UNFIXED`` and therefore cannot vanish
    with it.
    """
    build, make_inputs, kwargs = roundtrip_subject(name)
    keras.utils.set_random_seed(0)
    model = build()
    assert type(model).build is keras.layers.Layer.build, (
        f"{name} now implements build(); move it from UNFIXED to FIXED "
        f"(waiver was: {UNFIXED[name]})")


@pytest.mark.parametrize("name", sorted(RETIRED_WAIVER))
def test_the_waiver_table_is_empty_and_every_retired_entry_has_a_build(name):
    """The retired waiver, asserted in the direction that can now fail.

    Two claims, both of which go RED on a regression:

    1. :data:`UNFIXED` is empty. Re-waiving one of these five by deleting its
       ``build()`` and re-listing it here fails on this line.
    2. Each of the five names an object whose class OVERRIDES ``build``.
       Deleting a ``build()`` without touching this file also fails, by name.

    Claim 2 is the exact inverse of the assertion the old
    ``test_the_three_unfixed_subjects_are_pinned_by_name`` made about these
    same five, so the file's total coverage of them did not drop when the
    waiver was retired.
    """
    assert UNFIXED == {}, (
        f"UNFIXED regrew: {sorted(UNFIXED)}. Every RD-1 class was measured "
        f"materializable on 2026-08-23; a new waiver needs its own measured "
        f"reason, not a restored one.")

    build, make_inputs, kwargs = roundtrip_subject(name)
    keras.utils.set_random_seed(0)
    model = build()
    assert type(model).build is not keras.layers.Layer.build, (
        f"{name} lost its build() and fell back to the defaulted "
        f"keras.layers.Layer.build, which marks the model built while it "
        f"holds zero materialized state. The retired waiver said: "
        f"{RETIRED_WAIVER[name]}")
    assert name in FIXED, (
        f"{name} implements build() but is missing from FIXED, so no arm "
        f"pins what that build() materializes")


@pytest.mark.parametrize(
    "name,build_shape,call_shapes,pinned",
    [
        # `input_ids` / `attention_mask`, sequence axis genuinely None -- the
        # shape `create_qwen3_next_generation` and `_classification` build the
        # backbone at. Called afterwards at TWO other sequence lengths.
        pytest.param(
            "qwen",
            {"input_ids": (None, None), "attention_mask": (None, None)},
            [(2, 20), (3, 7)],
            97,
            id="qwen",
        ),
        # H/W genuinely None -- the fully-convolutional build the SCUNet module
        # docstring advertises. Called at a larger square and at a non-square.
        pytest.param(
            "scunet",
            (None, None, None, 3),
            [(1, 96, 96, 3), (1, 64, 128, 3)],
            148,
            id="scunet",
        ),
    ],
)
def test_a_coerced_build_probe_does_not_pin_the_call_shape(
        name, build_shape, call_shapes, pinned):
    """The property the ``qwen``/``scunet`` waiver claimed was unobtainable.

    Both of these ``build()`` methods substitute a concrete extent for a
    ``None`` axis. That substitution is a BUILD-TIME probe: it must materialize
    the weights and then constrain nothing. The waiver's objection -- "a
    concrete seq_len would size the positional table wrongly", "SCUNet is
    DESIGNED for dynamic spatial extents" -- is precisely the claim measured
    here, so this test, not a comment, is what licenses the coercion.

    Built at the fully-``None`` shape, then called at two DIFFERENT concrete
    shapes: the forward must run, produce the shape the input asked for, and
    move the weight population not at all.
    """
    build, make_inputs, kwargs = roundtrip_subject(name)
    keras.utils.set_random_seed(0)
    model = build()

    model.build(build_shape)
    after_build = _paths(model)
    assert len(after_build) == pinned, (
        f"{name}: build() at the None-axis shape materialized "
        f"{len(after_build)} weights, pinned at {pinned}")

    for shape in call_shapes:
        if isinstance(build_shape, dict):
            # `attention_mask` is ONES, not zeros: a zero mask marks every
            # position as padding, and a fully-masked row is a softmax over
            # nothing -- a NaN forward would be read here as a shape defect.
            inputs = {
                key: (np.ones if key == "attention_mask" else np.zeros)(
                    shape, dtype="int32")
                for key in build_shape
            }
        else:
            inputs = np.zeros(shape, dtype="float32")
        outputs = model(inputs)
        # `qwen` appends a vocab axis, `scunet` returns the input rank, so the
        # comparison is over the axes the INPUT names.
        assert tuple(outputs.shape[:len(shape)]) == shape, (
            f"{name}: call at {shape} returned {tuple(outputs.shape)}; the "
            f"build-time probe leaked into the call shape")
        assert _paths(model) == after_build, (
            f"{name}: calling at {shape} changed the weight population that "
            f"build() at the None-axis shape produced: "
            f"{sorted(set(_paths(model)) ^ set(after_build))}")
