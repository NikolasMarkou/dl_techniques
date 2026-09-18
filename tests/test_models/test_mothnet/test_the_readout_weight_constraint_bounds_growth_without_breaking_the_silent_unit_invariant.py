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

# DECISION plan-2026-09-18T110506-e42a44c7/D-008
This module originally proved D-006's whole-tensor-clip reasoning ("a
silent unit's row never moves, because its `new_weights` never moved away
from `readout_weights` in the first place -- clipping an unmoved,
already-in-range value is idempotent"). An adversarial review
(`findings/review-iter-1.md` Concern 1 and 2) found two problems: (1) D-006's
reasoning is false whenever a silent row is already OUTSIDE the constraint's
range before the update -- the whole-tensor clip moves it anyway; (2) this
module's own silent-unit assertion was structurally vacuous (it zeroed
`readout_weights` before testing, so `delta[silent] == clip(0.0) == 0.0`
trivially for any positive bound, and never exercised the violating
configuration). `HebbianReadoutLayer.hebbian_update` is now row-scoped (only
MB units that fired this batch are grown/clipped; every silent row is left
as `self.readout_weights` exactly, in-range or not) per D-008. This module
now proves that fix: a REALISTIC (nonzero, Glorot-initialized) silent-unit
case, and an explicit already-out-of-range silent-row case matching the
reviewer's own reproduction. See decisions.md D-008 and the anchor at
`src/dl_techniques/layers/mothnet_blocks.py:577`
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
    ``delta[silent_MB_unit] == 0.0`` still holds from REALISTIC (nonzero,
    Glorot-initialized) starting weights, and the clip is genuinely enforced
    (``max(|after|) <= bound``), not merely configured and skipped. Test (c)
    below pins the exact configuration `findings/review-iter-1.md` Concern 1
    measured broken: a silent row that is already OUTSIDE the constraint's
    range before the update.

    RED-proof for the D-008 row-scoping fix (performed once by hand at
    authoring time, not re-run automatically by this file):
    ``src/dl_techniques/layers/mothnet_blocks.py``'s row-scoped
    ``keras.ops.where(...)`` write was temporarily reverted to the prior
    unconditional ``self.readout_weights.assign(new_weights)`` (the
    whole-tensor clip D-006 originally shipped). Result:
    ``test_active_constraint_leaves_an_already_out_of_range_silent_row_untouched``
    went RED (the pre-existing out-of-range silent row was moved,
    ``max|delta[silent]| == 2.0`` instead of the required ``0.0``); this
    test (``test_active_constraint_bounds_magnitude_without_moving_silent_units``)
    stayed GREEN under the reverted code too -- correctly, since D-006's own
    "unmoved, already-in-range value clips to itself" reasoning IS true for
    this test's fixture (its derived ``bound`` is deliberately chosen above
    every silent row's realized Glorot magnitude, so no silent row starts
    out-of-range here). This test targets the REALISTIC (in-range-init)
    case; the out-of-range case above is what D-006's reasoning actually got
    wrong, and it is the one this module's predecessor version never
    exercised (findings/review-iter-1.md Concern 2). Restoring the real
    row-scoped fix left both tests GREEN, confirmed via ``git diff`` showing
    zero remaining change to the source file. See the executor's status
    report for the exact revert/restore transcript and captured failure
    output.
    """

    def test_active_constraint_bounds_magnitude_without_moving_silent_units(self):
        x = _features(batch=8)
        y = keras.utils.to_categorical(
            np.arange(8) % NUM_CLASSES, num_classes=NUM_CLASSES
        ).astype("float32")
        y_tensor = keras.ops.convert_to_tensor(y)

        # Real, nonzero Glorot-initialized starting weights -- NOT zeroed.
        # A prior version of this test zeroed `readout_weights` before
        # testing, which made `delta[silent] == clip(0.0) == 0.0` trivially
        # true for ANY bound (findings/review-iter-1.md Concern 2); real
        # Glorot values are nonzero, so the assertion below is now genuinely
        # falsifiable. A deliberately large `hebbian_learning_rate` (mirrors
        # the same choice `TestTheConstraintBoundsMagnitudeAcrossSyntheticUpdates`
        # below makes for its own comparative proof) guarantees growth on a
        # fired row dwarfs any silent row's random Glorot magnitude
        # regardless of the run's un-seeded random init draw, so the
        # precondition below is not a coin flip.
        probe = MothNet(num_classes=NUM_CLASSES, hebbian_learning_rate=100.0)
        probe.build((None, NUM_FEATURES))
        mb_output = probe.mushroom_body(
            probe.antennal_lobe(x, training=False), training=False
        )

        # Same MB sparse-code precondition as the pinned D-018 test in
        # test_model.py: some mushroom-body units must be silent (k << 2000
        # mb_units), or the derivation and assertion below are vacuous.
        silent = np.all(keras.ops.convert_to_numpy(mb_output) == 0.0, axis=0)
        assert silent.any(), (
            "precondition: the sparse code must leave some units silent, or "
            "this assertion is vacuous"
        )
        fired = ~silent

        initial_weights = np.array(
            keras.ops.convert_to_numpy(probe.readout.readout_weights), copy=True
        )
        initial_silent_max_abs = float(np.max(np.abs(initial_weights[silent])))

        # Probe: measure this layer's own unclipped growth on the FIRED rows
        # only, from the SAME real starting weights, deriving the bound from
        # measurement (never a guessed constant). The tensor-WIDE max-abs is
        # not a safe anchor here (and is what a first attempt at this
        # rewrite used, and it was itself vacuous): with mb_units=2000 and
        # k=200 firing, ~90% of rows are silent for any one batch, so the
        # tensor-wide max almost always sits on an UNTOUCHED silent row and
        # growth on the fired rows need not ever cross it.
        probe.readout.hebbian_update(mb_output, y_tensor)
        unclipped_after = keras.ops.convert_to_numpy(probe.readout.readout_weights)
        unclipped_fired_max_abs = float(np.max(np.abs(unclipped_after[fired])))
        assert unclipped_fired_max_abs > initial_silent_max_abs, (
            "precondition: hebbian_update must grow a fired row's weight "
            "past the largest silent row's magnitude given this batch's "
            "real mb_output/target pair, or the bound derivation below is "
            "vacuous"
        )
        # Sits strictly between the largest SILENT row's (untouched)
        # magnitude and the grown FIRED-row magnitude: no silent row starts
        # out-of-range (so a silent row's raw Glorot value can never itself
        # trip the clip -- that scenario is covered separately, by
        # `test_active_constraint_leaves_an_already_out_of_range_silent_row_untouched`
        # below), but growth on a fired row is expected to cross it (so the
        # clip is exercised, not merely configured).
        bound = (initial_silent_max_abs + unclipped_fired_max_abs) / 2.0

        # The real (constrained) run: identical real starting weights --
        # forced via .assign(), not left to coincidence -- and the identical
        # mb_output/target update, with the derived bound active.
        model = MothNet(
            num_classes=NUM_CLASSES,
            hebbian_learning_rate=100.0,
            readout_weight_bound=bound,
        )
        model.build((None, NUM_FEATURES))
        model.readout.readout_weights.assign(
            keras.ops.convert_to_tensor(initial_weights)
        )
        before = np.array(
            keras.ops.convert_to_numpy(model.readout.readout_weights), copy=True
        )

        model.readout.hebbian_update(mb_output, y_tensor)

        after = keras.ops.convert_to_numpy(model.readout.readout_weights)
        delta = after - before

        assert float(np.max(np.abs(delta[silent]))) == 0.0, (
            "an ACTIVE constraint moved the readout row of a mushroom-body "
            "unit that never fired -- the D-008 row-scoping fix does not "
            "hold in practice"
        )

        # The clip is actually enforced -- prove engagement, not just
        # configuration. bound was derived specifically so this could fail.
        max_abs_after = float(np.max(np.abs(after)))
        assert max_abs_after <= bound + 1e-6, (
            f"max(|readout_weights|)={max_abs_after} exceeds the active "
            f"bound {bound} -- the constraint did not engage"
        )

    def test_active_constraint_leaves_an_already_out_of_range_silent_row_untouched(self):
        """The exact configuration `findings/review-iter-1.md` Concern 1
        measured broken: a silent row whose PRE-UPDATE value already sits
        outside ``[min_value, max_value]`` -- reachable via a
        ``readout_weight_bound`` tighter than the realized
        ``kernel_initializer`` range, or an externally assigned/loaded
        out-of-range weight. Before the D-008 fix, the whole-tensor clip
        moved this row toward the rail (measured ``max|delta|=2.0``) even
        though it never fired this batch, breaking the pinned D-018
        invariant. `# DECISION plan-2026-09-18T110506-e42a44c7/D-008`.
        """
        input_dim = 5
        units = 3
        layer = HebbianReadoutLayer(
            units=units,
            kernel_constraint=ValueRangeConstraint(min_value=-1.0, max_value=1.0),
        )
        layer.build((None, input_dim))

        # Directly reproduces the reviewer's own counterexample: a silent
        # row already outside [-1.0, 1.0].
        out_of_range_row = np.array([3.0, -3.0, 0.05], dtype="float32")
        weights = keras.ops.convert_to_numpy(layer.readout_weights).copy()
        weights[0] = out_of_range_row
        layer.readout_weights.assign(keras.ops.convert_to_tensor(weights))
        before = np.array(
            keras.ops.convert_to_numpy(layer.readout_weights), copy=True
        )

        batch_size = 4
        pre_synaptic = np.ones((batch_size, input_dim), dtype="float32")
        # MB unit (row) 0 is the SILENT unit this test targets: zero its
        # column across the whole batch so it fires for no sample.
        pre_synaptic[:, 0] = 0.0
        labels = np.arange(batch_size) % units
        post_synaptic = keras.ops.convert_to_tensor(
            keras.utils.to_categorical(labels, num_classes=units).astype("float32")
        )

        layer.hebbian_update(
            keras.ops.convert_to_tensor(pre_synaptic), post_synaptic
        )

        after = keras.ops.convert_to_numpy(layer.readout_weights)
        delta = after - before

        np.testing.assert_array_equal(
            after[0], out_of_range_row,
            err_msg=(
                "an already out-of-range silent row was modified by "
                "hebbian_update -- D-018 violated exactly as measured by "
                "findings/review-iter-1.md Concern 1"
            ),
        )
        assert float(np.max(np.abs(delta[0]))) == 0.0, (
            "an already out-of-range silent row's delta was nonzero -- "
            "D-018 violated"
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
