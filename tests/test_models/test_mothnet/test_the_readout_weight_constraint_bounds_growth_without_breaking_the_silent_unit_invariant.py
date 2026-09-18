"""
Single-claim guard, ACTIVE-constraint half of plan-2026-09-18T110506-e42a44c7
Step 6.

`test_model.py::TestHebbianUpdateAndWinnerTakeAllAreLoadBearing` already pins
the D-018 silent-unit invariant (``delta[silent_MB_unit] == 0.0``) on the
off-by-default (``readout_weight_bound=None``) path. That is a regression
check, not a proof the invariant survives an ACTIVE constraint -- the two are
genuinely different code paths through ``hebbian_update()`` (one skips the
``kernel_constraint`` branch entirely, one runs it). This file supplies that
missing half, plus a comparative proof the bound is load-bearing (not merely
present and inert).

# DECISION plan-2026-09-18T110506-e42a44c7/D-006
This module is the measured proof behind D-006's reasoning ("a whole-tensor
clip cannot move a silent unit's row, because that row's `new_weights` never
moved away from `readout_weights` in the first place -- clipping an unmoved,
already-in-range value is idempotent"). See decisions.md D-006 and the
anchor at `src/dl_techniques/layers/mothnet_blocks.py:574`
(`HebbianReadoutLayer.hebbian_update`).
"""

import keras
import numpy as np

from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint
from dl_techniques.layers.mothnet_blocks import HebbianReadoutLayer
from dl_techniques.models.general_purpose.mothnet.model import MothNet

NUM_FEATURES = 64
NUM_CLASSES = 10

# decisions.md D-005: Step 1's empirically-derived bound (2x the epoch-7
# max-abs weight value from a faithful 15-epoch reproduction). Reused here,
# as plan.md's own Step 6 text permits, for test (b)'s comparative proof.
STEP1_DERIVED_BOUND = 2.7858


def _features(batch=8):
    return np.random.default_rng(0).random((batch, NUM_FEATURES)).astype("float32")


class TestActiveConstraintPreservesTheSilentUnitInvariant:
    """Test (a): with the constraint ACTIVE (not the off-by-default path),
    ``delta[silent_MB_unit] == 0.0`` still holds, and the clip is genuinely
    enforced (``max(|after|) <= bound``), not merely configured and skipped.

    RED-proof (mandatory per ``plans/LESSONS.md`` for any new guard,
    performed once by hand at authoring time against BOTH tests in this
    module, not re-run automatically by this file):
    ``src/dl_techniques/layers/mothnet_blocks.py``'s
    ``if self.kernel_constraint is not None:`` line was temporarily replaced
    with ``if False:`` (the constraint call became dead code). Both this
    test's magnitude assertion and
    ``TestTheConstraintBoundsMagnitudeAcrossSyntheticUpdates``'s below went
    RED (the unclipped ``new_weights`` exceeded their respective derived
    bounds); restoring the real ``if self.kernel_constraint is not None:``
    guard made both GREEN again, confirmed via ``git diff`` showing zero
    remaining change to the source file. See the executor's status report
    for the exact revert/restore transcript and captured failure output.
    """

    def test_active_constraint_bounds_magnitude_without_moving_silent_units(self):
        x = _features(batch=8)
        y = keras.utils.to_categorical(
            np.arange(8) % NUM_CLASSES, num_classes=NUM_CLASSES
        ).astype("float32")
        y_tensor = keras.ops.convert_to_tensor(y)

        # Probe: measure THIS layer's own unclipped growth from a known-zero
        # starting point, deriving the bound from measurement (never a
        # guessed constant). A first attempt at this test derived the bound
        # from the RAW glorot_uniform init range instead (its measured
        # tensor-wide max-abs) and found it does not compose safely with a
        # real MB sparse code: mb_units=2000 with k=200 firing means ~90% of
        # rows are silent for any one batch, so the tensor-wide max-abs
        # entry lands on a silent row with high probability -- growth then
        # provably CANNOT move it, and no bound can simultaneously sit above
        # the untouched max (required so no entry is clipped by
        # initialization alone) and below it (required so growth is what
        # engages the clip). Zeroing the starting weights removes that
        # confound: every entry's post-update value becomes its own growth
        # exactly (`learning_rate * weight_update`, both non-negative), so
        # growth strictly increases only the entries it touches and a silent
        # row's entries stay exactly zero -- trivially inside any positive
        # bound.
        probe = MothNet(num_classes=NUM_CLASSES)
        probe.build((None, NUM_FEATURES))
        mb_output = probe.mushroom_body(
            probe.antennal_lobe(x, training=False), training=False
        )
        probe.readout.readout_weights.assign(
            keras.ops.zeros_like(probe.readout.readout_weights)
        )
        probe.readout.hebbian_update(mb_output, y_tensor)
        unclipped_growth = keras.ops.convert_to_numpy(probe.readout.readout_weights)
        unclipped_max_abs = float(np.max(np.abs(unclipped_growth)))
        assert unclipped_max_abs > 0.0, (
            "precondition: hebbian_update must move at least one weight "
            "given this batch's real mb_output/target pair, or the "
            "derivation below is vacuous"
        )
        bound = unclipped_max_abs / 2.0

        # The real (constrained) run: identical zero starting point -- forced
        # via .assign(), not left to coincidence -- and the identical
        # mb_output/target update, with the derived bound active.
        model = MothNet(num_classes=NUM_CLASSES, readout_weight_bound=bound)
        model.build((None, NUM_FEATURES))
        model.readout.readout_weights.assign(
            keras.ops.zeros_like(model.readout.readout_weights)
        )
        before = np.array(
            keras.ops.convert_to_numpy(model.readout.readout_weights), copy=True
        )

        model.readout.hebbian_update(mb_output, y_tensor)

        after = keras.ops.convert_to_numpy(model.readout.readout_weights)
        delta = after - before

        # Same MB sparse-code precondition as the pinned D-018 test in
        # test_model.py: some mushroom-body units must be silent (k << 2000
        # mb_units), or the assertion below is vacuous.
        silent = np.all(keras.ops.convert_to_numpy(mb_output) == 0.0, axis=0)
        assert silent.any(), (
            "precondition: the sparse code must leave some units silent, or "
            "this assertion is vacuous"
        )
        assert float(np.max(np.abs(delta[silent]))) == 0.0, (
            "an ACTIVE constraint moved the readout row of a mushroom-body "
            "unit that never fired -- D-006's whole-tensor-clip reasoning "
            "does not hold in practice"
        )

        # The clip is actually enforced -- prove engagement, not just
        # configuration. bound was derived specifically so this could fail.
        max_abs_after = float(np.max(np.abs(after)))
        assert max_abs_after <= bound + 1e-6, (
            f"max(|readout_weights|)={max_abs_after} exceeds the active "
            f"bound {bound} -- the constraint did not engage"
        )


class TestTheConstraintBoundsMagnitudeAcrossSyntheticUpdates:
    """Test (b): a genuine COMPARATIVE proof over >= 20 synthetic
    ``hebbian_update()`` calls -- the constrained instance's magnitude never
    exceeds the bound, and the unconstrained instance's DOES exceed it, from
    the identical starting weights and the identical update sequence.
    """

    def test_constrained_stays_bounded_and_unconstrained_exceeds_it(self):
        bound = STEP1_DERIVED_BOUND
        input_dim = 5
        units = 3
        # Deliberately large learning_rate + batch layout (see below):
        # guarantees the unconstrained instance separates past `bound` well
        # inside 20 updates, without leaving that to chance.
        learning_rate = 1.0
        num_updates = 20
        batch_size = 6  # 6 % 3 == 0: every class gets an equal, FIXED share
        # of every batch (see `labels` below), so growth accumulates
        # deterministically instead of depending on an RNG draw.

        constrained = HebbianReadoutLayer(
            units=units,
            learning_rate=learning_rate,
            kernel_constraint=ValueRangeConstraint(min_value=-bound, max_value=bound),
        )
        constrained.build((None, input_dim))

        unconstrained = HebbianReadoutLayer(
            units=units,
            learning_rate=learning_rate,
            kernel_constraint=None,
        )
        unconstrained.build((None, input_dim))

        # "Identical seeds": force bit-identical starting weights rather than
        # relying on two independent random draws landing close together.
        unconstrained.set_weights(constrained.get_weights())
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(constrained.readout_weights),
            keras.ops.convert_to_numpy(unconstrained.readout_weights),
        )

        # Synthetic, non-negative pre_synaptic / one-hot post_synaptic --
        # no real MNIST needed. Every dim's pre-synaptic value is 1.0 and
        # every class gets exactly `batch_size // units` rows every update,
        # so each weight column's growth per update is an exact, deterministic
        # `(batch_size // units) / batch_size * learning_rate`.
        pre_synaptic = keras.ops.convert_to_tensor(
            np.ones((batch_size, input_dim), dtype="float32")
        )
        labels = np.arange(batch_size) % units
        post_synaptic = keras.ops.convert_to_tensor(
            keras.utils.to_categorical(labels, num_classes=units).astype("float32")
        )

        for step in range(num_updates):
            constrained.hebbian_update(pre_synaptic, post_synaptic)
            unconstrained.hebbian_update(pre_synaptic, post_synaptic)

            constrained_max = float(np.max(np.abs(
                keras.ops.convert_to_numpy(constrained.readout_weights)
            )))
            assert constrained_max <= bound + 1e-6, (
                f"constrained instance exceeded the bound at update {step}: "
                f"{constrained_max} > {bound}"
            )

        unconstrained_max = float(np.max(np.abs(
            keras.ops.convert_to_numpy(unconstrained.readout_weights)
        )))
        assert unconstrained_max > bound, (
            "unconstrained instance did not exceed the bound after "
            f"{num_updates} updates ({unconstrained_max} <= {bound}) -- the "
            "comparative proof requires a genuine separation, not a "
            "coincidental one"
        )
