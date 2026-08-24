"""
RED proofs for ``roundtrip_instrument_oracle``.

Every assertion the round-trip family makes over 72 packages is proven here to
FAIL against a deliberately broken subject. A guard that has never been seen red
is not evidence, and this repo has measured the shape twice: a transposed-stride
injection that was a SYMMETRY of every subject it was meant to break (D-077),
and a round-trip test that passed 3/3 while 464 of 1,305 tensors were never
written.

The injections are REAL models wherever a real model can carry the defect --
a sub-layer that refuses to load its own variables, a sub-layer with no explicit
``name=``, a "disabled" head that gets built anyway, two sub-layers sharing a
name. Only the two report-shaped conditions (a loaded model that was called
before its weights were read; a lost weight) are injected at the report level,
and even those are paired with a liveness assertion on the real instrument.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from .roundtrip_instrument_oracle import (
    assert_build_parity,
    assert_disabled_component_has_no_weights,
    assert_roundtrip_output_values,
    assert_weights_restored_before_first_call,
    measure_build_parity,
    measure_roundtrip,
    n_colliding_paths,
    relative_path,
    weight_map,
)


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="roundtrip_red_proofs")
class LossyDense(keras.layers.Dense):
    """A ``Dense`` whose ``load_own_variables`` silently does nothing.

    This is not a hypothetical: it is the exact shape of D-029, where
    ``SHGCNLinkPredictor``'s archive was COMPLETE and its LOAD was lossy because
    an unbuilt sub-layer's ``load_own_variables`` was skipped and the decoder
    reverted to its defaults with no warning.
    """

    def load_own_variables(self, store):  # noqa: D102 -- see the class docstring
        return


def _named_model(dense_cls=keras.layers.Dense, name="subject"):
    inputs = keras.Input((4,), name="inp")
    hidden = keras.layers.Dense(6, name="hidden")(inputs)
    outputs = dense_cls(3, name="out")(hidden)
    return keras.Model(inputs, outputs, name=name)


@keras.saving.register_keras_serializable(package="roundtrip_red_proofs")
class SubclassedTwoLayer(keras.Model):
    """A SUBCLASSED model with two explicitly named sub-layers.

    Subclassed, deliberately: a functional model's variables never carry the
    model name at all, so the donor/loaded asymmetry this proof is about does
    not arise there -- it arises exactly where the audit found it, in the
    subclassed models that make up most of ``models/``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden = keras.layers.Dense(6, name="hidden")
        self.out = keras.layers.Dense(3, name="out")

    def call(self, inputs, training=None):
        return self.out(self.hidden(inputs, training=training),
                        training=training)


class UnnamedSublayerModel(keras.Model):
    """A model whose sub-layer is built with NO explicit ``name=``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.named = keras.layers.Dense(4, name="named_dense")
        self.unnamed = keras.layers.Dense(3)  # deliberately auto-named

    def call(self, inputs, training=None):
        return self.unnamed(self.named(inputs), training=training)


class CollidingNamesModel(keras.Model):
    """Two distinct sub-layers carrying the SAME name -- ``depth_anything``'s
    student/teacher shape, reduced to two dense layers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = keras.layers.Dense(4, name="twin")
        self.second = keras.layers.Dense(4, name="twin")

    def call(self, inputs, training=None):
        return self.first(inputs, training=training) + self.second(
            inputs, training=training)


class HeadModel(keras.Model):
    """A backbone plus an optional head, with an ``over_build`` mode that builds
    the head even when it is switched off."""

    def __init__(self, include_top=True, over_build=False, **kwargs):
        super().__init__(**kwargs)
        self.backbone = keras.layers.Dense(4, name="backbone")
        self.head = (keras.layers.Dense(2, name="classifier")
                     if include_top or over_build else None)
        self.include_top = include_top

    def call(self, inputs, training=None):
        features = self.backbone(inputs, training=training)
        if self.head is not None:
            # The head is CALLED (and therefore built) even when it is off:
            # this is the over-building that build parity alone cannot see.
            head_out = self.head(features, training=training)
            if self.include_top:
                return head_out
        return features


def _inputs():
    return np.random.RandomState(0).randn(2, 4).astype("float32")


# ---------------------------------------------------------------------------
# relative_path -- the instrument defect that was found first
# ---------------------------------------------------------------------------

class TestRelativePath:
    """The ``split("/", 1)[-1]`` spelling the rule sketch uses is asymmetric."""

    def test_the_naive_spelling_pairs_nothing_after_a_reload(self):
        model = SubclassedTwoLayer(name="subclassed_subject")
        model(_inputs())
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.keras")
            model.save(path)
            loaded = keras.models.load_model(path, compile=False)

            naive_donor = [w.path.split("/", 1)[-1] for w in model.weights]
            naive_loaded = [w.path.split("/", 1)[-1] for w in loaded.weights]
            # THE DEFECT: the naive spelling strips the model name on the donor
            # side and the first real LAYER name on the loaded side, so the two
            # key spaces are not the same space. Here it collapses four distinct
            # weights onto two keys.
            # THE DEFECT, measured: the donor keeps `hidden/kernel` while the
            # loaded model collapses to `kernel`, so the two key spaces do not
            # even intersect.
            assert not (set(naive_donor) & set(naive_loaded)), (
                f"donor {sorted(set(naive_donor))} vs loaded "
                f"{sorted(set(naive_loaded))}")

            proper_donor = {relative_path(model, w) for w in model.weights}
            proper_loaded = {relative_path(loaded, w) for w in loaded.weights}
            assert proper_donor == proper_loaded


# ---------------------------------------------------------------------------
# R-073 -- the weight arm
# ---------------------------------------------------------------------------

class TestWeightArm:

    def test_a_lossy_load_is_RED(self):
        """A sub-layer that refuses to load its own variables must fail."""
        report = measure_roundtrip(lambda: _named_model(LossyDense), _inputs)
        assert report["weight_max_delta"] > 0.0, (
            "the injected lossy load moved no weight; the injection is dead")
        with pytest.raises(AssertionError, match="came back different"):
            assert_weights_restored_before_first_call(report, atol=0.0)

    def test_the_healthy_subject_passes(self):
        report = measure_roundtrip(_named_model, _inputs)
        assert report["weight_max_delta"] == 0.0
        assert_weights_restored_before_first_call(report, atol=0.0)

    def test_the_call_counter_is_live(self):
        """The 'before first call' guard is only worth something if the counter
        it reads can move at all."""
        report = measure_roundtrip(_named_model, _inputs)
        assert report["call_count_before_weight_read"] == 0
        assert report["call_count_after_forward"] > 0

    def test_a_model_called_before_the_read_is_RED(self):
        report = dict(measure_roundtrip(_named_model, _inputs))
        report["call_count_before_weight_read"] = 1
        with pytest.raises(AssertionError, match="before its weights"):
            assert_weights_restored_before_first_call(report)

    def test_a_dead_call_counter_is_RED(self):
        report = dict(measure_roundtrip(_named_model, _inputs))
        report["call_count_after_forward"] = 0
        with pytest.raises(AssertionError, match="never incremented"):
            assert_weights_restored_before_first_call(report)

    def test_a_vacuous_comparison_is_RED(self):
        """Every compared weight reproduced by a FRESH instance -- the shape all
        four historical vacuity mechanisms take."""
        report = dict(measure_roundtrip(_named_model, _inputs))
        report["n_inert"] = report["n_compared"]
        report["n_effective"] = 0
        with pytest.raises(AssertionError, match="cannot fail"):
            assert_weights_restored_before_first_call(report)

    def test_undeclared_path_drift_is_RED(self):
        report = dict(measure_roundtrip(_named_model, _inputs))
        report["matched_by"] = "position"
        report["n_path_mismatch"] = 3
        with pytest.raises(AssertionError, match="fell back to weight ORDER"):
            assert_weights_restored_before_first_call(report)

    def test_an_undeclared_path_collision_is_RED(self):
        model = CollidingNamesModel(name="twins")
        model(_inputs())
        assert n_colliding_paths(weight_map(model)) == 2, (
            "the injected name collision did not collide")
        report = measure_roundtrip(lambda: CollidingNamesModel(name="twins"),
                                   _inputs)
        assert report["n_colliding_paths"] == 2
        with pytest.raises(AssertionError, match="share a relative path"):
            assert_weights_restored_before_first_call(report)
        # ...and passes once the measured collision is DECLARED.
        assert_weights_restored_before_first_call(report,
                                                  expect_path_collisions=2)

    def test_the_weight_snapshot_is_taken_after_the_forward(self):
        """The ``yolo12`` ordering defect, reproduced and pinned.

        A ``training=True`` forward updates BatchNorm moving statistics. A
        weight snapshot taken BEFORE it is stale by the time the model is
        saved, and reads a spurious delta against a true 0.0.
        """
        def build():
            inputs = keras.Input((4,), name="inp")
            outputs = keras.layers.BatchNormalization(name="bn")(inputs)
            return keras.Model(inputs, outputs, name="bn_subject")

        # The correct order, as `measure_roundtrip` does it.
        report = measure_roundtrip(build, _inputs, training=True)
        assert report["weight_max_delta"] == 0.0

        # The wrong order, by hand.
        keras.utils.set_random_seed(0)
        model = build()
        model(_inputs(), training=True)
        stale = dict(weight_map(model))
        model(_inputs(), training=True)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.keras")
            model.save(path)
            loaded = keras.models.load_model(path, compile=False)
            fresh = dict(weight_map(loaded))
        spurious = max(float(np.max(np.abs(stale[k].astype("float64")
                                           - fresh[k].astype("float64"))))
                       for k in stale)
        assert spurious > 0.0, (
            "the stale-snapshot ordering was expected to read a delta that is "
            "not there")


# ---------------------------------------------------------------------------
# R-063 -- the output arm
# ---------------------------------------------------------------------------

class TestOutputArm:

    def test_a_lossy_load_moves_the_output_and_is_RED(self):
        report = measure_roundtrip(lambda: _named_model(LossyDense), _inputs)
        assert report["output_max_delta"] > 0.0
        with pytest.raises(AssertionError):
            assert_roundtrip_output_values(report, atol=0.0)

    def test_the_healthy_subject_is_exact(self):
        report = measure_roundtrip(_named_model, _inputs)
        assert report["output_max_delta"] == 0.0
        assert_roundtrip_output_values(report, atol=0.0)

    def test_rtol_is_zero_not_the_numpy_default(self):
        """``assert_allclose``'s default ``rtol=1e-7`` would pass a relative
        difference this size; ``rtol=0`` must not."""
        report = dict(measure_roundtrip(_named_model, _inputs))
        # float64 on both sides: in float32 a 1e-8 relative nudge is below the
        # representable resolution and is an exact no-op, so the injection
        # would be dead and the proof would prove nothing.
        donor = report["donor_outputs"][0].astype("float64")
        scaled = donor * (1.0 + 1e-8)
        assert np.max(np.abs(donor - scaled)) > 0.0, "the nudge was a no-op"
        report["donor_outputs"] = [donor]
        report["loaded_outputs"] = [scaled]
        report["per_output_delta"] = [float(np.max(np.abs(donor - scaled)))]
        np.testing.assert_allclose(donor, scaled)  # the numpy default PASSES
        with pytest.raises(AssertionError):
            assert_roundtrip_output_values(report, atol=0.0)

    def test_calibrating_a_deterministic_model_is_RED(self):
        """``calibrate=True`` cannot be used to hide a real difference."""
        report = dict(measure_roundtrip(_named_model, _inputs))
        assert report["self_max_delta"] == 0.0
        with pytest.raises(AssertionError, match="it is deterministic"):
            assert_roundtrip_output_values(report, calibrate=True)


# ---------------------------------------------------------------------------
# R-072(a) -- build parity
# ---------------------------------------------------------------------------

class TestBuildParity:

    def test_an_unnamed_sublayer_is_RED(self):
        report = measure_build_parity(lambda: UnnamedSublayerModel(name="u"),
                                      _inputs, input_shape=(None, 4))
        assert report["drift"], "the unnamed sub-layer did not drift"
        with pytest.raises(AssertionError, match="covered by no declared"):
            assert_build_parity(report)
        # ...and passes once the auto-named stem is DECLARED.
        assert_build_parity(report, autoname_stems=("dense",))

    def test_a_stale_stem_is_RED(self):
        """A stem left behind by a repair must fire, or the waiver table rots."""
        report = measure_build_parity(_named_model, _inputs,
                                      input_shape=(None, 4))
        assert not report["drift"]
        with pytest.raises(AssertionError, match="matches no drifting path"):
            assert_build_parity(report, autoname_stems=("dense",))

    def test_a_healthy_builder_passes(self):
        report = measure_build_parity(_named_model, _inputs,
                                      input_shape=(None, 4))
        assert_build_parity(report)

    def test_an_undeclared_collision_is_RED(self):
        report = measure_build_parity(lambda: CollidingNamesModel(name="twins"),
                                      _inputs, input_shape=(None, 4))
        with pytest.raises(AssertionError, match="share a relative path"):
            assert_build_parity(report)
        assert_build_parity(report, expect_path_collisions=2)

    def test_a_phantom_weight_from_build_is_RED(self):
        report = dict(measure_build_parity(_named_model, _inputs,
                                           input_shape=(None, 4)))
        report["explicit"] = dict(report["explicit"])
        report["explicit"]["not_in_lazy"] = ["ghost/kernel"]
        with pytest.raises(AssertionError, match="a forward pass does not"):
            assert_build_parity(report)


# ---------------------------------------------------------------------------
# R-072(b) -- the no-sub-layer-config sibling
# ---------------------------------------------------------------------------

class TestDisabledComponent:

    def test_the_healthy_sibling_passes(self):
        report = assert_disabled_component_has_no_weights(
            lambda: HeadModel(include_top=True, name="h"),
            lambda: HeadModel(include_top=False, name="h"),
            _inputs, marker="classifier")
        assert report["n_marked_with"] == 2
        assert report["n_marked_without"] == 0

    def test_an_over_built_head_is_RED(self):
        """The failure the arm exists for: the head is built anyway, so build
        parity between the two paths would PASS."""
        with pytest.raises(AssertionError, match="still carries"):
            assert_disabled_component_has_no_weights(
                lambda: HeadModel(include_top=True, name="h"),
                lambda: HeadModel(include_top=False, over_build=True, name="h"),
                _inputs, marker="classifier")

    def test_a_marker_matching_nothing_is_RED(self):
        """A typo in the marker must not make the arm vacuously green."""
        with pytest.raises(AssertionError, match="matches no weight"):
            assert_disabled_component_has_no_weights(
                lambda: HeadModel(include_top=True, name="h"),
                lambda: HeadModel(include_top=False, name="h"),
                _inputs, marker="not_a_real_weight_name")
