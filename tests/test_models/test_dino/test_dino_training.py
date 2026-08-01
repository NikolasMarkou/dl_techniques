"""Tests for `src/dl_techniques/models/dino/dino_training.py` (DINOTrainingModel).

Written for plan-2026-08-01T105809-dc0c402e step 8. Three things here are load
bearing and must not be "simplified":

1. **Every weight is re-seeded NON-ZERO.** `DINOv1`'s biases initialize to
   zeros, and several probes below are structurally blind against a zeros
   weight: an EMA update `t <- d*t + (1-d)*s` moves nothing when `t == s == 0`,
   and a "teacher moved" assertion then passes with the mechanism dead.
2. **The EMA integration test asserts the teacher moved TOWARD the student**,
   not merely that it moved. A delta alone is satisfied by any perturbation.
3. **The student is frozen during the EMA integration test** (`SGD(0.0)`), with
   a runtime assertion that it really did not move, so every observed teacher
   movement is attributable to `update_teacher_ema` and nothing else.

The self-disable trap this file exists to catch:
`TeacherEMACallback.on_train_batch_end` does
`getattr(self.model, "update_teacher_ema", None)` and, when that is `None`,
logs ONE warning and sets `self._disabled = True`. The run then completes
normally with a teacher that was never updated.
"""

import inspect

import keras
import numpy as np
import pytest

from dl_techniques.losses.dino_loss import DINOLoss, pack_student_teacher
from dl_techniques.models.depth_anything.teacher_ema import (
    TeacherEMACallback,
    cosine_ema_schedule,
)
from dl_techniques.models.dino import DINOv1
from dl_techniques.models.dino.dino_training import (
    N_GLOBAL_VIEWS,
    DINOTrainingModel,
    create_dino_training_model,
)

IMAGE_SIZE = 32
PATCH_SIZE = 16
OUT_DIM = 16
N_LOCAL = 2
N_VIEWS = N_GLOBAL_VIEWS + N_LOCAL
# (teacher global view, student view) pairs, same-view pair removed.
N_PAIRS = N_GLOBAL_VIEWS * N_VIEWS - N_GLOBAL_VIEWS


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _backbone(name):
    """A deliberately tiny DINOv1 with a projection head."""
    return DINOv1(
        embed_dim=32,
        depth=2,
        num_heads=4,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_classes=0,
        include_projection_head=True,
        dino_out_dim=OUT_DIM,
        dino_hidden_dim=32,
        dino_bottleneck_dim=16,
        name=name,
    )


def _seed_nonzero(model, seed):
    """Overwrite EVERY weight with non-zero values.

    Not cosmetic: `DINOv1` initializes all biases to zeros, and a zeros weight
    makes an EMA probe blind (`d*0 + (1-d)*0 == 0` for every `d`).
    """
    rng = np.random.RandomState(seed)
    for weight in model.weights:
        values = rng.normal(loc=0.35, scale=0.25, size=weight.shape)
        # Push every entry off zero, keeping the sign structure.
        values = np.where(np.abs(values) < 0.05, 0.05, values)
        weight.assign(values.astype(np.asarray(weight).dtype))
    assert all(
        float(np.abs(np.asarray(w)).min()) > 0.0 for w in model.weights
    ), "seeding failed: some weight still contains an exact zero"


def _pair(seed_student=11, seed_teacher=22):
    """Two backbones seeded FAR APART, before `DINOTrainingModel` sees them.

    The gap here is deliberately NOT the source of any test's teacher/student
    divergence — `DINOTrainingModel.__init__` synchronizes the teacher to the
    student (D-034), so a model built from this pair comes out with the two
    EQUAL. The far-apart seeding survives only so that
    `test_construction_copies_the_student_into_the_teacher`'s non-vacuity
    control has something real to measure: it proves the pair WAS different
    before the copy.
    """
    student = _backbone("dino_student")
    teacher = _backbone("dino_teacher")
    _seed_nonzero(student, seed_student)
    _seed_nonzero(teacher, seed_teacher)
    return student, teacher


def _model(**kwargs):
    student, teacher = _pair()
    return DINOTrainingModel(
        student=student, teacher=teacher, n_local_views=N_LOCAL, **kwargs
    )


def _diverge_teacher(model, seed=33):
    """Push the teacher OFF the student, explicitly, after construction.

    Since D-034 the teacher starts EQUAL to the student, which is what DINO
    requires. Three kinds of test still need a gap: the EMA arithmetic (an
    `t <- d*t + (1-d)*s` update moves nothing when `t == s`), the EMA
    integration test (it asserts the teacher-student distance SHRINKS, which
    needs a non-zero distance to shrink), and the packed-pairing test (with
    `t == s` it could no longer tell a teacher forward pass from a student
    one). Those tests create the divergence HERE, visibly, instead of relying
    on construction leaving the pair apart — relying on that is precisely what
    made the missing-copy defect invisible to this suite.

    Returns the resulting teacher/student distance, asserted non-zero.
    """
    _seed_nonzero(model.teacher, seed)
    distance = _distance(model.teacher, model.student)
    assert distance > 0.0, (
        "the deliberate divergence did not take: teacher and student are still "
        "identical, so any 'the teacher moved toward the student' assertion "
        "below would be vacuous"
    )
    return distance


def _batch(batch_size=3, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(
        size=(batch_size, N_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 3)
    ).astype("float32")


def _weights(model):
    return [np.asarray(w).copy() for w in model.weights]


def _distance(model_a, model_b):
    """Total L1 distance between two structurally identical models."""
    return float(
        sum(
            np.abs(np.asarray(a) - np.asarray(b)).sum()
            for a, b in zip(model_a.weights, model_b.weights)
        )
    )


# ---------------------------------------------------------------------


class TestForwardContract:
    """The multi-crop input contract and the packed single-tensor output."""

    def test_forward_shape_at_the_documented_contract(self):
        model = _model()
        batch_size = 3
        out = model(_batch(batch_size))

        assert model.n_views == N_VIEWS
        assert model.n_pairs == N_PAIRS
        assert tuple(out.shape) == (batch_size * N_PAIRS, 2 * OUT_DIM)
        assert model.compute_output_shape(
            (batch_size, N_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 3)
        ) == (batch_size * N_PAIRS, 2 * OUT_DIM)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_packed_rows_are_the_expected_student_teacher_pairs(self):
        """Each output row really is (student view i, teacher global view j).

        This is the test that makes the pairing falsifiable. A model that
        returned the right SHAPE while pairing the wrong views -- or while
        running the teacher on local crops -- passes every shape assertion
        above and fails here.
        """
        model = _model()
        # Explicit divergence: since D-034 the teacher is built EQUAL to the
        # student, and an equal pair would make this test unable to tell a
        # teacher forward pass from a student one (the "teacher half" of every
        # row would match the student's output on the same view). See
        # `_diverge_teacher`.
        _diverge_teacher(model)
        x = _batch(2, seed=5)
        out = np.asarray(model(x)).reshape(2, N_PAIRS, 2 * OUT_DIM)

        # Independently recompute both networks' per-view outputs.
        student_by_view = np.stack(
            [np.asarray(model.student(x[:, v])) for v in range(N_VIEWS)],
            axis=1,
        )
        teacher_by_view = np.stack(
            [np.asarray(model.teacher(x[:, v], training=False))
             for v in range(N_GLOBAL_VIEWS)],
            axis=1,
        )
        # Non-vacuity: the views must actually differ from one another,
        # otherwise any pairing would satisfy the assertions below. BOTH
        # networks are checked -- the teacher's two global views separate by
        # only ~2.8e-04 here (a UnitNorm-constrained head over seeded weights
        # is nearly view-invariant), which is 28x the atol used below and is
        # exactly the margin the "wrong teacher view" RED arm fires on.
        student_separation = float(
            np.abs(student_by_view[:, 0] - student_by_view[:, 1]).max())
        teacher_separation = float(
            np.abs(teacher_by_view[:, 0] - teacher_by_view[:, 1]).max())
        assert student_separation > 1e-4, student_separation
        assert teacher_separation > 5e-5, (
            f"the teacher's two global views differ by only "
            f"{teacher_separation:.3e}, below the 1e-5 tolerance used below -- "
            f"this test can no longer tell global view 0 from global view 1"
        )

        expected_pairs = [
            (t, s)
            for t in range(N_GLOBAL_VIEWS)
            for s in range(N_VIEWS)
            if s != t
        ]
        assert len(expected_pairs) == N_PAIRS
        for index, (t_view, s_view) in enumerate(expected_pairs):
            np.testing.assert_allclose(
                out[:, index, :OUT_DIM], student_by_view[:, s_view],
                rtol=1e-5, atol=1e-5,
                err_msg=f"row {index} student half is not view {s_view}",
            )
            np.testing.assert_allclose(
                out[:, index, OUT_DIM:], teacher_by_view[:, t_view],
                rtol=1e-5, atol=1e-5,
                err_msg=f"row {index} teacher half is not global view {t_view}",
            )

    def test_layout_agrees_with_pack_student_teacher(self):
        """The model uses the loss module's packing helper, not its own order."""
        student = np.arange(6, dtype="float32").reshape(2, 3)
        teacher = np.arange(6, dtype="float32").reshape(2, 3) + 100.0
        packed = np.asarray(pack_student_teacher(student, teacher))
        np.testing.assert_array_equal(packed[:, :3], student)
        np.testing.assert_array_equal(packed[:, 3:], teacher)

    def test_wrong_view_count_is_refused(self):
        model = _model()
        wrong = np.zeros((2, N_VIEWS + 1, IMAGE_SIZE, IMAGE_SIZE, 3), "float32")
        with pytest.raises(ValueError, match=r"configured for n_views=4"):
            model(wrong)

    def test_wrong_per_view_resolution_is_refused(self):
        model = _model()
        wrong = np.zeros((2, N_VIEWS, IMAGE_SIZE * 2, IMAGE_SIZE, 3), "float32")
        with pytest.raises(ValueError, match=r"SAME pixel\s+resolution"):
            model(wrong)

    def test_non_rank_5_input_is_refused(self):
        model = _model()
        with pytest.raises(ValueError, match="rank-5 multi-crop input"):
            model(np.zeros((2, IMAGE_SIZE, IMAGE_SIZE, 3), "float32"))


# ---------------------------------------------------------------------


class TestTeacherIsGradientFree:
    """`stop_gradient` + `trainable=False`, each asserted separately."""

    def test_no_gradient_reaches_the_teacher_weights(self):
        import tensorflow as tf

        model = _model()
        x = _batch(2)
        # `tape.watch` needs the backing tf.Variable; a keras.Variable is
        # refused with `ValueError: Passed in object ... of type 'Variable'`.
        teacher_vars = [w.value for w in model.teacher.weights]

        with tf.GradientTape() as tape:
            tape.watch(teacher_vars)
            out = model(x, training=True)
            scalar = tf.reduce_sum(tf.square(out))
        grads = tape.gradient(scalar, teacher_vars)

        worst = max(
            0.0 if g is None else float(tf.reduce_max(tf.abs(g)))
            for g in grads
        )
        assert worst == 0.0, (
            f"gradient reached the teacher (max |dL/dw| = {worst:.6e}); the "
            f"teacher must be behind keras.ops.stop_gradient"
        )

    def test_the_same_probe_does_see_the_student_gradient(self):
        """Non-vacuity control for the test above."""
        import tensorflow as tf

        model = _model()
        x = _batch(2)
        student_vars = [w.value for w in model.student.trainable_weights]

        with tf.GradientTape() as tape:
            tape.watch(student_vars)
            scalar = tf.reduce_sum(tf.square(model(x, training=True)))
        grads = tape.gradient(scalar, student_vars)

        worst = max(
            0.0 if g is None else float(tf.reduce_max(tf.abs(g)))
            for g in grads
        )
        assert worst > 1e-8, (
            "the gradient probe sees nothing even for the STUDENT, so the "
            "teacher result above proves nothing"
        )

    def test_teacher_weights_are_excluded_from_trainable_weights(self):
        model = _model()
        model(_batch(2))
        assert model.teacher.trainable is False
        assert len(model.trainable_weights) == len(
            model.student.trainable_weights)
        teacher_ids = {id(w) for w in model.teacher.weights}
        assert not any(id(w) in teacher_ids for w in model.trainable_weights)


# ---------------------------------------------------------------------


class TestUpdateTeacherEMA:
    """The `TeacherEMACallback` contract and the EMA arithmetic."""

    def test_exposes_update_teacher_ema(self):
        model = _model()
        assert hasattr(model, "update_teacher_ema")

    def test_signature_matches_what_the_callback_calls(self):
        """The callback does `update_fn(decay=decay)` -- by keyword, by name."""
        model = _model()
        source = inspect.getsource(TeacherEMACallback.on_train_batch_end)
        assert 'getattr(self.model, "update_teacher_ema", None)' in source
        assert "update_fn(decay=decay)" in source

        parameters = inspect.signature(model.update_teacher_ema).parameters
        assert "decay" in parameters
        # Callable with the exact spelling the callback uses.
        model.update_teacher_ema(decay=0.9)

    def test_moves_every_weight_to_the_exact_ema_value(self):
        model = _model()
        _diverge_teacher(model)  # D-034: the pair is built EQUAL; see helper.
        decay = 0.5
        before = _weights(model.teacher)
        student = _weights(model.student)

        model.update_teacher_ema(decay)

        after = _weights(model.teacher)
        for index, (b, s, a) in enumerate(zip(before, student, after)):
            expected = decay * b + (1.0 - decay) * s
            # Non-vacuity: the update must actually be visible at this decay.
            assert np.abs(expected - b).max() > 1e-3, (
                f"weight {index} would not move even under a correct EMA -- "
                f"the fixture is not seeded far enough apart"
            )
            np.testing.assert_allclose(a, expected, rtol=1e-5, atol=1e-6)

    def test_moves_the_teacher_toward_the_student(self):
        model = _model()
        before = _diverge_teacher(model)  # D-034: built EQUAL; see helper.
        model.update_teacher_ema(0.5)
        after = _distance(model.teacher, model.student)
        assert before > 0.0
        assert after < before
        np.testing.assert_allclose(after, 0.5 * before, rtol=1e-4)

    @pytest.mark.parametrize("decay", [-0.1, 1.5])
    def test_out_of_range_decay_raises(self, decay):
        model = _model()
        with pytest.raises(ValueError, match=r"decay must be in \[0, 1\]"):
            model.update_teacher_ema(decay)


# ---------------------------------------------------------------------


class TestTeacherEMACallbackIntegration:
    """A real `fit()` with the real callback attached."""

    def test_callback_moves_the_teacher_toward_the_student(self):
        model = _model()
        # D-034: the pair is built EQUAL, so the gap this test closes has to be
        # created here, deliberately. See `_diverge_teacher`.
        _diverge_teacher(model)
        x = _batch(4)
        y = np.zeros((4, 1), "float32")

        # Freeze the student so every observed teacher movement is EMA.
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.0),
            loss=DINOLoss(out_dim=OUT_DIM),
        )
        callback = TeacherEMACallback(
            schedule=cosine_ema_schedule(0.5, 0.5, total_steps=10),
            warmup_steps=0,
        )

        # Settle the student's `UnitNorm` last-layer constraint BEFORE
        # snapshotting. MEASURED: a Keras constraint is applied inside
        # `optimizer.apply`, so even at learning_rate=0.0 the FIRST step
        # renormalizes the seeded kernel (256/256 entries moved, max 0.52) and
        # the "student is frozen" control below would fire on that, not on a
        # real weight update. Constraint projection is idempotent, so after
        # this warmup step the student is genuinely fixed.
        model.fit(x, y, epochs=1, batch_size=2, verbose=0)

        student_before = _weights(model.student)
        teacher_before = _weights(model.teacher)
        distance_before = _distance(model.teacher, model.student)

        model.fit(x, y, epochs=1, batch_size=2, verbose=0,
                  callbacks=[callback])

        # The callback ran and did NOT self-disable.
        assert callback._disabled is False, (
            "TeacherEMACallback SELF-DISABLED: the model does not expose "
            "update_teacher_ema(decay) under the name the callback looks up"
        )
        assert callback.step == 2, (
            f"expected 2 training batches, the callback counted "
            f"{callback.step} (it stops counting once disabled)"
        )

        # The student really is frozen, so the teacher's movement is the EMA's.
        # atol 1e-6, not 0: re-applying the (already satisfied) UnitNorm
        # constraint each step still perturbs the kernel by ~6e-08 in float32.
        # The teacher movement asserted below is ~1e4 x larger.
        for b, a in zip(student_before, _weights(model.student)):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

        moved = max(
            float(np.abs(a - b).max())
            for a, b in zip(_weights(model.teacher), teacher_before)
        )
        assert moved > 1e-4, (
            f"teacher weights did not move during fit() (max delta {moved:.3e})"
        )

        distance_after = _distance(model.teacher, model.student)
        assert distance_after < distance_before, (
            f"teacher moved but NOT toward the student "
            f"({distance_before:.4f} -> {distance_after:.4f})"
        )
        # Two batches at decay 0.5 => distance quartered.
        np.testing.assert_allclose(
            distance_after, 0.25 * distance_before, rtol=1e-3)


# ---------------------------------------------------------------------


class TestTeacherStartsFromTheStudent:
    """D-034. DINO's teacher is an EMA of the STUDENT'S OWN trajectory.

    Before this was fixed, `create_dino_teacher_student_pair` returned two
    independently random `DINOv1`s and nothing copied one into the other.
    MEASURED at the defect: 55 of 157 weight tensors differed and the two
    networks' outputs on the same input differed by max|delta| 0.3002, so the
    first several hundred steps distilled the student against an unrelated
    network (at `ema_decay_start=0.996`, `0.996**295 = 30.66%` of that
    unrelated teacher survives epoch 1).

    The suite was structurally blind to it because the fixtures seeded the pair
    FAR APART and every existing check was about distinct OBJECTS, never about
    equal VALUES.
    """

    def test_construction_copies_the_student_into_the_teacher(self):
        """Teacher == student at construction, with a NON-VACUITY control.

        The control matters: a test that only asserted equality would pass on a
        pair that happened to be identical already (e.g. two zero-initialized
        models), proving nothing about the copy. So the pair is measured BEFORE
        construction and required to differ.
        """
        student, teacher = _pair()

        before = max(
            float(np.abs(np.asarray(t) - np.asarray(s)).max())
            for t, s in zip(teacher.weights, student.weights)
        )
        differing_before = sum(
            1 for t, s in zip(teacher.weights, student.weights)
            if float(np.abs(np.asarray(t) - np.asarray(s)).max()) > 0.0
        )
        assert before > 0.1, (
            f"non-vacuity control failed: the pair handed to DINOTrainingModel "
            f"already agreed to {before:.3e}, so 'teacher == student after "
            f"construction' would be true whether or not the copy ran"
        )
        assert differing_before > 0

        model = DINOTrainingModel(
            student=student, teacher=teacher, n_local_views=N_LOCAL)

        after = max(
            float(np.abs(np.asarray(t) - np.asarray(s)).max())
            for t, s in zip(model.teacher.weights, model.student.weights)
        )
        assert after == 0.0, (
            f"the teacher was NOT initialized from the student: {after:.6e} "
            f"max|delta| remains across {len(model.teacher.weights)} weight "
            f"tensors (it was {before:.6e} across {differing_before} tensors "
            f"before construction). DINO's teacher must be an EMA of the "
            f"student's own trajectory from the student's own initialization."
        )

    def test_the_two_networks_agree_on_the_forward_pass_at_construction(self):
        """The value-level consequence, measured where it is observable."""
        model = _model()
        x = _batch(2, seed=9)[:, 0]
        student_out = np.asarray(model.student(x, training=False))
        teacher_out = np.asarray(model.teacher(x, training=False))
        # Non-vacuity: the outputs must not be trivially constant.
        assert float(np.abs(student_out).max()) > 1e-3
        np.testing.assert_allclose(
            teacher_out, student_out, rtol=0, atol=0,
            err_msg="teacher and student disagree on the same input at step 0",
        )

    def test_the_pair_factory_itself_synchronizes(self):
        """`create_dino_teacher_student_pair` is a public exported factory.

        It must return a synchronized pair on its own, not only when its output
        is handed to `DINOTrainingModel`.
        """
        from dl_techniques.models.dino.dino_v1 import (
            create_dino_teacher_student_pair,
        )

        teacher, student = create_dino_teacher_student_pair(
            variant="tiny", image_size=32, patch_size=16, dino_out_dim=64)
        worst = max(
            float(np.abs(np.asarray(t) - np.asarray(s)).max())
            for t, s in zip(teacher.weights, student.weights)
        )
        assert worst == 0.0, (
            f"create_dino_teacher_student_pair returned an UNSYNCHRONIZED "
            f"pair (max|delta| {worst:.6e})"
        )
        # ... while still being two distinct variable sets.
        assert not ({id(w) for w in teacher.weights}
                    & {id(w) for w in student.weights})

    def test_dino_backbones_are_built_at_construction(self):
        """WHERE the copy can live, decided by measurement rather than guess.

        A student->teacher copy placed before the weights exist copies nothing
        and no-ops silently — the same defect wearing a fix's clothes. MEASURED:
        `DINOv1.__init__` builds eagerly, so a freshly constructed backbone
        already carries its full weight list and `__init__` is a valid home for
        the copy. If this ever stops being true, the copy must move to `build()`
        and this test is the signal.
        """
        fresh = _backbone("freshly_constructed")
        assert fresh.built is True
        assert len(fresh.weights) > 0

    def test_the_sync_helper_refuses_an_unbuilt_model(self):
        """A copy that silently no-ops is the defect wearing a fix's clothes."""
        from dl_techniques.models.dino.common import sync_teacher_to_student

        unbuilt = keras.Sequential([keras.layers.Dense(4)])
        built = keras.Sequential([keras.layers.Dense(4)])
        built(np.zeros((1, 3), "float32"))
        assert not unbuilt.weights, "the fixture is not actually unbuilt"
        with pytest.raises(ValueError, match="BUILT models"):
            sync_teacher_to_student(unbuilt, built)


# ---------------------------------------------------------------------


class TestStudentTeacherIndependence:
    """A shared pair would make the whole EMA mechanism a silent no-op."""

    def test_weights_are_independent_objects(self):
        """Distinct VARIABLES. Their VALUES are equal at construction (D-034).

        The two properties are orthogonal and both matter: equal values are
        what DINO requires, distinct variables are what makes the EMA an EMA
        rather than an alias.
        """
        model = _model()
        student_ids = {id(w) for w in model.student.weights}
        assert not any(id(w) in student_ids for w in model.teacher.weights)

        model.student.weights[0].assign(
            np.asarray(model.student.weights[0]) + 1.0)
        assert not np.allclose(
            np.asarray(model.student.weights[0]),
            np.asarray(model.teacher.weights[0]),
        )

    def test_the_same_model_passed_twice_is_refused(self):
        student, _ = _pair()
        with pytest.raises(ValueError, match="two DISTINCT models"):
            DINOTrainingModel(student=student, teacher=student, n_local_views=1)

    def test_a_structurally_different_teacher_is_refused(self):
        student, _ = _pair()
        other = DINOv1(
            embed_dim=32, depth=3, num_heads=4, patch_size=PATCH_SIZE,
            image_size=IMAGE_SIZE, num_classes=0,
            include_projection_head=True, dino_out_dim=OUT_DIM,
            dino_hidden_dim=32, dino_bottleneck_dim=16, name="deeper",
        )
        with pytest.raises(ValueError, match="structurally identical"):
            DINOTrainingModel(student=student, teacher=other, n_local_views=1)


# ---------------------------------------------------------------------


class TestSerialization:
    """`.keras` round-trip, with VALUES asserted and `trainable` asserted."""

    def test_round_trip_reproduces_the_output_numerically(self, tmp_path):
        model = _model()
        x = _batch(2, seed=3)
        original = np.asarray(model(x))
        # Non-vacuity: a model whose output is ~0 everywhere would satisfy the
        # comparison below with every weight dropped.
        assert np.abs(original).max() > 1e-3

        path = tmp_path / "dino_training.keras"
        model.save(path)
        restored = keras.models.load_model(path)

        np.testing.assert_allclose(
            np.asarray(restored(x)), original, rtol=1e-6, atol=1e-6)

    def test_round_trip_keeps_the_teacher_frozen(self, tmp_path):
        """A frozen sub-model silently reloading UNFROZEN is a recorded gotcha.

        Its outputs are bit-identical, so only an explicit `trainable`
        assertion catches it -- the numeric round-trip test above cannot.
        """
        model = _model()
        model(_batch(2))
        path = tmp_path / "dino_training.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        restored(_batch(2))

        assert restored.teacher.trainable is False, (
            "the teacher reloaded TRAINABLE: it would receive gradients and "
            "the EMA would fight the optimizer, with identical outputs and no "
            "error anywhere"
        )
        assert len(restored.trainable_weights) == len(
            restored.student.trainable_weights)

    def test_round_trip_preserves_the_view_layout(self, tmp_path):
        model = _model()
        model(_batch(2))
        path = tmp_path / "dino_training.keras"
        model.save(path)
        restored = keras.models.load_model(path)

        assert restored.n_local_views == N_LOCAL
        assert restored.n_views == N_VIEWS
        assert restored.n_pairs == N_PAIRS
        assert restored.out_dim == OUT_DIM

    def test_reloaded_model_still_updates_its_teacher(self, tmp_path):
        model = _model()
        # D-034: a freshly built pair is EQUAL, and an EMA over an equal pair
        # moves nothing, so the divergence is created explicitly.
        _diverge_teacher(model)
        model(_batch(2))
        path = tmp_path / "dino_training.keras"
        model.save(path)
        restored = keras.models.load_model(path)

        before = _distance(restored.teacher, restored.student)
        assert before > 0.0
        restored.update_teacher_ema(0.5)
        np.testing.assert_allclose(
            _distance(restored.teacher, restored.student),
            0.5 * before, rtol=1e-4)

    def test_reload_keeps_a_trained_teacher(self, tmp_path):
        """The D-034 copy must NOT re-fire destructively on reload.

        `from_config` calls `cls(**config)`, so the student->teacher copy DOES
        run again when a saved model is loaded. That is safe only because
        `keras.models.load_model` restores the saved weights into the
        reconstructed object AFTERWARDS, and the restore is authoritative. If
        that ordering ever changed, a resumed pretraining run would silently
        throw away its trained teacher and restart the EMA from the student --
        no error, no warning, a plausible loss curve. This test is the only
        thing that would see it.
        """
        model = _model()
        # Stand in for a trained teacher: a value the student cannot produce.
        _diverge_teacher(model, seed=77)
        saved_teacher = _weights(model.teacher)
        saved_distance = _distance(model.teacher, model.student)
        assert saved_distance > 0.0

        model(_batch(2))
        path = tmp_path / "dino_training.keras"
        model.save(path)
        restored = keras.models.load_model(path)

        worst = max(
            float(np.abs(np.asarray(a) - b).max())
            for a, b in zip(restored.teacher.weights, saved_teacher)
        )
        assert worst == 0.0, (
            f"the reloaded teacher differs from the SAVED teacher by "
            f"{worst:.6e} -- the construction-time student->teacher copy "
            f"survived the weight restore and destroyed a trained teacher"
        )
        np.testing.assert_allclose(
            _distance(restored.teacher, restored.student),
            saved_distance, rtol=1e-5)


# ---------------------------------------------------------------------


class TestFactory:
    """`create_dino_training_model` follows the converged DINO factory scheme."""

    def test_builds_a_usable_model(self):
        model = create_dino_training_model(
            "tiny", image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
            n_local_views=1, dino_out_dim=8,
        )
        assert isinstance(model, DINOTrainingModel)
        assert model.n_views == N_GLOBAL_VIEWS + 1
        assert model.out_dim == 8
        out = model(np.zeros(
            (2, N_GLOBAL_VIEWS + 1, IMAGE_SIZE, IMAGE_SIZE, 3), "float32"))
        assert tuple(out.shape) == (2 * (N_GLOBAL_VIEWS * (N_GLOBAL_VIEWS + 1)
                                         - N_GLOBAL_VIEWS), 16)
        assert model.teacher.trainable is False

    def test_refuses_input_shape(self):
        with pytest.raises(TypeError, match="no longer accepts 'input_shape'"):
            create_dino_training_model(
                "tiny", image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            )

    def test_negative_local_views_raise(self):
        student, teacher = _pair()
        with pytest.raises(ValueError, match="n_local_views must be >= 0"):
            DINOTrainingModel(
                student=student, teacher=teacher, n_local_views=-1)
