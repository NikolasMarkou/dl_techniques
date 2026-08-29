"""The `.keras` round trip restores values, weights and layout.

Three §16.3 items, none of which existed in this directory before
2026-08-29 (`rtol=0` had zero grep hits, `atol=0.0` had zero, `w.path`
had zero):

* §7.1  round trip on VALUES, `rtol=0`, explicit `training=False`
* §8.4  weight-value comparison at `atol=0.0` BEFORE the loaded
        model's first call, plus the scalar parameter total
* §8.3  build parity by relative `w.path`, plus its no-sub-layer
        sibling -- parity alone is blind to over-building because it
        passes when BOTH paths build everything

Every "nothing changed" assertion here is paired with a "something
changed" twin (§13.1 rule 3): an identity assertion alone is satisfied
by a completely dead component.

Tolerances. Every comparison in this module is at `atol=0.0, rtol=0`.
That is not a guess: restoration is a copy, not a computation, and the
reloaded model runs the same graph on the same device, so the measured
difference is exactly 0.0 for all four classes (2026-08-29, RTX 4070,
policy float32, batch 4 on an 8x16 grid). The defect signal it sits
below is a reload that restores fresh random weights, which moves the
output by the full output magnitude (0.61 to 2.45 here).
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from .logic_subject_oracle import (
    SUBJECTS,
    SUBJECT_NAMES,
    relative_weight_paths,
)


def _randomize_weights(model, seed=99):
    """Overwrite every weight with a draw the initializers never make.

    Without this the whole module would be vacuous. Every default
    initializer in this package is `'zeros'` or a `Constant`, so a load
    path that silently re-initialized every weight instead of
    restoring it would produce EXACTLY the saved values, and both the
    value round trip and the `atol=0.0` weight comparison would pass on
    the defect they exist to catch (measured 2026-08-29 against an
    injected `CircuitDepthLayer.build` that never builds its children).
    """
    rng = np.random.default_rng(seed)
    for weight in model.weights:
        shape = keras.ops.convert_to_numpy(weight).shape
        weight.assign(rng.normal(0.0, 0.4, size=shape).astype("float32"))
    return model


def _saved_and_loaded(model, tmpdir):
    """Save ``model`` and return the reloaded copy, uncalled."""
    path = os.path.join(tmpdir, "subject.keras")
    model.save(path)
    return keras.models.load_model(path)


class TestKerasRoundTripPreservesValues:
    """§7.1. The saved model reproduces the donor's OUTPUT VALUES."""

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_the_reloaded_model_reproduces_every_output_value(self, name):
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        model = subject.model()
        sample = subject.inputs()
        model(sample, training=False)
        _randomize_weights(model)

        original = keras.ops.convert_to_numpy(
            model(sample, training=False)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = _saved_and_loaded(model, tmpdir)
            restored = keras.ops.convert_to_numpy(
                loaded(sample, training=False)
            )

        np.testing.assert_allclose(
            original, restored, atol=0.0, rtol=0,
            err_msg=f"{name} values differ after a .keras round trip",
        )

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_a_perturbed_weight_moves_the_output(self, name):
        """The twin. Without it, the identity above is also satisfied
        by a model whose output does not depend on its weights at all.

        The bump is applied to ONE scalar entry, not to the whole
        tensor. Adding the same constant to every selection logit is
        the exact no-op softmax is shift-invariant under, and a
        whole-tensor bump measured `moved == 0` for
        LearnableLogicOperator on 2026-08-29 -- a twin that proved
        nothing.
        """
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        model = subject.model()
        sample = subject.inputs()
        before = keras.ops.convert_to_numpy(model(sample, training=False))

        moved = []
        for weight in model.weights:
            value = keras.ops.convert_to_numpy(weight).copy()
            bumped = value.copy()
            bumped.reshape(-1)[0] += 0.75
            weight.assign(bumped)
            after = keras.ops.convert_to_numpy(
                model(sample, training=False)
            )
            weight.assign(value)
            if float(np.max(np.abs(after - before))) > 1e-4:
                moved.append(weight.path)

        assert model.weights, f"{name} built no weights to perturb"
        assert moved, (
            f"no single weight entry of {name} changed its output by "
            f"more than 1e-4; the round-trip identity above would pass "
            f"on a completely dead layer"
        )


class TestSavedWeightsAreRestoredBeforeTheFirstCall:
    """§8.4. Compare the WEIGHTS, before the reloaded model is called.

    After one forward pass a build()-only load path reads the same
    weight COUNT for the correct and the broken variant, because the
    gap has been filled with fresh random weights.
    """

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_every_weight_is_restored_bit_for_bit(self, name):
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        model = subject.model()
        model(subject.inputs(), training=False)
        _randomize_weights(model)

        saved = [
            keras.ops.convert_to_numpy(w).copy() for w in model.weights
        ]
        assert saved, f"{name} has no weights to compare"

        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = _saved_and_loaded(model, tmpdir)

            # Read `loaded.weights` before any forward pass on it.
            assert len(loaded.weights) == len(saved), (
                f"{name} reloaded with {len(loaded.weights)} weights, "
                f"donor had {len(saved)}"
            )
            for donor, restored in zip(saved, loaded.weights):
                np.testing.assert_allclose(
                    donor, keras.ops.convert_to_numpy(restored),
                    atol=0.0, rtol=0,
                    err_msg=f"{name}: {restored.path} was not restored",
                )

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_the_scalar_parameter_total_survives_the_round_trip(
            self, name
    ):
        """A weight COUNT is blind to an internal-dimension change that
        reshapes without adding or removing tensors (§8.4)."""
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        model = subject.model()
        model(subject.inputs(), training=False)
        _randomize_weights(model)
        expected = model.count_params()
        assert expected > 0, f"{name} has no parameters to count"

        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = _saved_and_loaded(model, tmpdir)
            assert loaded.count_params() == expected, (
                f"{name} reloaded with {loaded.count_params()} scalar "
                f"parameters, donor had {expected}"
            )


class TestBuildsExactlyWhatCallRuns:
    """§8.3. Build parity, and the sibling that sees over-building."""

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_explicit_build_matches_the_functional_build(self, name):
        """Parity by RELATIVE `w.path`. This only compares equal
        because every sub-layer in this package carries an explicit
        `name=`; without that, two instances read `block/w` versus
        `block_1/w` and a failure here is a naming problem before it is
        a build problem.
        """
        subject = SUBJECTS[name]

        keras.utils.set_random_seed(7)
        explicit = subject.make()
        shapes = subject.input_shapes()
        explicit.build(shapes if subject.arity > 1 else shapes[0])

        keras.utils.set_random_seed(7)
        lazy = subject.make()
        subject.model(lazy)

        assert relative_weight_paths(explicit), (
            f"{name} built no weights either way"
        )
        assert relative_weight_paths(explicit) == relative_weight_paths(
            lazy
        ), f"{name}: explicit build and functional build disagree"

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_a_disabled_sub_layer_creates_no_weights(self, name):
        """The anti-vacuity sibling. Parity is blind to over-building:
        it passes if BOTH paths build everything.
        """
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        absent = subject.make_absent()
        subject.model(absent)

        leaked = [
            w.path for w in absent.weights
            if subject.absent_scope in w.path
        ]
        assert not leaked, (
            f"{name} built {subject.absent_scope} weights while it was "
            f"configured off: {leaked}"
        )

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_the_default_configuration_does_create_that_sub_layer(
            self, name
    ):
        """The twin of the sibling above. Without it, an absent-weight
        assertion is satisfied by a class that never builds that
        sub-layer under any configuration, and the sibling proves
        nothing about the knob.
        """
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        present = subject.make_present()
        subject.model(present)

        found = [
            w.path for w in present.weights
            if subject.absent_scope in w.path
        ]
        assert found, (
            f"{name} never builds a {subject.absent_scope} weight, so "
            f"its absence proves nothing"
        )
