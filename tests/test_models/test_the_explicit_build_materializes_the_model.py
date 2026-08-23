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
``mobilenet`` 0 -> 267, ``pw_fnet`` 0 -> 100, ``power_mlp`` 0 -> 5), and
``video_jepa``/``latent_gmm_registration``/``lewm``/``qwen``/``scunet`` are
NOT here --
see :data:`UNFIXED` for the measured reason on each.

See ``decisions.md`` D-013.
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
    "mamba": 14,
    "masked_language_model": 23,
    "mobilenet": 267,
    "modern_bert": 13,
    "power_mlp": 5,
    "pw_fnet": 100,
    "relgt": 49,
    "resnet": 32,
    "sd3_mmdit": 124,
    "tabm": 5,
    "tree_transformer": 28,
    "vq_vae": 9,
    "vq_vae_rotation": 9,
}

#: The three RD-1 classes deliberately left unfixed, and why. ``build()`` is NOT
#: implemented for these: ``materialize_sublayers`` refuses to fall back to an
#: eager forward pass, which is the only trace that succeeds here and which
#: would execute their ``add_loss()`` calls and BatchNorm updates for real.
UNFIXED = {
    "latent_gmm_registration": "raw-tf op in call(); a KerasTensor cannot enter it",
    "lewm": "call() does integer arithmetic on the batch axis, then add_loss()",
    "video_jepa": "same as lewm; also the only partially-built subject (66/161)",
    # Not a trace limitation of the helper but of the SHAPE real callers use:
    # `create_qwen3_next_*` builds the backbone inside a functional graph whose
    # SEQUENCE axis is None, and `call()` needs a concrete `seq_len` for its
    # causal mask. The batch-axis retry deliberately does not substitute other
    # None axes -- a concrete seq_len would size the positional table wrongly.
    # Measured: a build() here turns 7 qwen node ids RED, all one cause,
    # `ValueError: seq_len required for causal mask`.
    "qwen": "call() needs a concrete seq_len; real callers build at seq_len=None",
    # SCUNet is DESIGNED for dynamic spatial extents -- it owns a whole
    # test file for it (`test_dynamic_spatial_dims.py`). Tracing from an
    # input_shape whose H/W are None reaches `-None` in the padding
    # arithmetic. Measured: 2 node ids RED, `TypeError: bad operand type
    # for unary -: 'NoneType'` at `models/scunet/model.py:521`.
    "scunet": "call() does padding arithmetic on None spatial dims by design",
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
    from dl_techniques.models.cliffordnet.lm import CliffordNetLM

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
def test_the_three_unfixed_subjects_are_pinned_by_name(name):
    """The waiver is not a blanket: each unfixed class is named with its reason.

    A class that GAINS a working ``build()`` fails here and must be moved into
    :data:`FIXED`, so the waiver cannot outlive its own justification.
    """
    build, make_inputs, kwargs = roundtrip_subject(name)
    keras.utils.set_random_seed(0)
    model = build()
    assert type(model).build is keras.layers.Layer.build, (
        f"{name} now implements build(); move it from UNFIXED to FIXED "
        f"(waiver was: {UNFIXED[name]})")
