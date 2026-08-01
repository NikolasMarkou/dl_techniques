"""
Tests for the DINO / iBOT / KoLeo self-supervised losses.

This module is the first test coverage `src/dl_techniques/losses/dino_loss.py`
has ever had. Before it, `DINOLoss(out_dim=8)` and `iBOTPatchLoss(out_dim=8)`
both raised `AttributeError: ... has no attribute 'add_weight'` at construction
and nothing under `tests/` referenced any of the three classes.

The load-bearing test here is `TestCenteringEMA::test_center_reaches_hand_computed_ema`:
it runs a REAL two-step `model.fit()` and asserts the centering variable reached
an independently recomputed float64 numpy EMA value. "The value changed" is not
an acceptable weakening of that assertion -- the whole point of the centering
mechanism is that it holds a specific statistic, and a `.assign()` that silently
no-ops under graph mode would still make the value change on the first step.
"""

import json
import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.losses.dino_loss import (
    DINOLoss,
    KoLeoLoss,
    iBOTPatchLoss,
)

# =============================================================================
# Numpy oracles -- deliberately independent reimplementations
# =============================================================================


def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax in float64.

    Args:
        x: Input array.
        axis: Axis to normalize over.

    Returns:
        Softmax of `x` along `axis`, in float64.
    """
    x = np.asarray(x, dtype=np.float64)
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def _np_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax in float64.

    Args:
        x: Input array.
        axis: Axis to normalize over.

    Returns:
        Log-softmax of `x` along `axis`, in float64.
    """
    x = np.asarray(x, dtype=np.float64)
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def _np_cross_entropy(
        teacher: np.ndarray,
        student: np.ndarray,
        center: np.ndarray,
        teacher_temp: float,
        student_temp: float,
) -> np.ndarray:
    """Per-element DINO cross-entropy, reducing only the last axis.

    Args:
        teacher: Teacher logits, `(..., out_dim)`.
        student: Student logits, `(..., out_dim)`.
        center: Centering vector broadcastable against `teacher`.
        teacher_temp: Teacher sharpening temperature.
        student_temp: Student sharpening temperature.

    Returns:
        Array of shape `teacher.shape[:-1]` holding `-sum(p_t * log p_s)`.
    """
    p_t = _np_softmax((np.asarray(teacher, np.float64) - center) / teacher_temp)
    log_p_s = _np_log_softmax(np.asarray(student, np.float64) / student_temp)
    return -(p_t * log_p_s).sum(axis=-1)


def _np_koleo(embeddings: np.ndarray, epsilon: float) -> float:
    """Reference KoLeo entropic regularizer value.

    Args:
        embeddings: `(batch, dim)` embeddings.
        epsilon: Stability constant added before the log.

    Returns:
        The scalar KoLeo loss.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    sim = x @ x.T
    sim = sim - 2.0 * np.eye(sim.shape[0])
    nn_sim = np.clip(sim.max(axis=1), -1.0, 1.0)
    distances = np.sqrt(np.maximum(2.0 - 2.0 * nn_sim, 0.0))
    return float(np.mean(-np.log(distances + epsilon)))


# -----------------------------------------------------------------------------
# TOLERANCE DERIVATION (used by every numeric assertion below)
#
# float32 eps = 2^-23 = 1.192e-07. The longest dependent chain in these tests is
# the centering EMA: a mean over a <=8-row batch (~3*eps), then per EMA step two
# multiplies and one add (~3*eps) applied twice (~6*eps), on top of a softmax /
# log-softmax pair over out_dim <= 8 (~4*eps). Total ~13*eps = 1.55e-06
# RELATIVE. Magnitudes here are bounded by |logits| <= ~3 and |loss| <= ~5, so
# the ABSOLUTE budget is ~7.8e-06. GPU cuBLAS reorders reduction summation,
# which changes the rounding but not the bound's magnitude; ~2.5x headroom gives
# the value below. This matches the step-1 DD-1 probe's independently derived
# ATOL, where the MEASURED error came in at 1.6e-08 (~1000x inside the bound).
ATOL = 2.0e-05
# -----------------------------------------------------------------------------

# Placeholder `y_true` for the structured-`y_pred` convention, where the loss
# ignores `y_true` entirely. It cannot be `None`: MEASURED on keras 3.8.0,
# `Loss.__call__` converts `y_true` to a tensor before dispatching to `call()`,
# so `None` raises `ValueError: Attempt to convert a value (None) with an
# unsupported type (<class 'NoneType'>) to a Tensor`. This mirrors
# `tests/test_losses/test_clip_contrastive_loss.py::get_dummy_labels`, the
# repo's established shape for a `y_true`-ignoring loss.
IGNORED_Y_TRUE = np.zeros((1,), dtype="float32")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cls_logits():
    """Teacher/student CLS-token logits for the `DINOLoss` tests.

    Returns:
        Tuple of `(teacher, student)` float32 arrays of shape `(4, 6)`.
    """
    rng = np.random.default_rng(20260801)
    teacher = rng.normal(size=(4, 6)).astype("float32")
    student = rng.normal(size=(4, 6)).astype("float32")
    return teacher, student


@pytest.fixture
def patch_logits():
    """Teacher/student patch logits and a partial mask for `iBOTPatchLoss`.

    Returns:
        Tuple of `(teacher, student, mask)` with shapes `(2, 5, 4)`,
        `(2, 5, 4)` and `(2, 5)`; the mask selects 4 of the 10 patches.
    """
    rng = np.random.default_rng(31337)
    teacher = rng.normal(size=(2, 5, 4)).astype("float32")
    student = rng.normal(size=(2, 5, 4)).astype("float32")
    mask = np.array(
        [[True, False, True, False, False],
         [False, True, False, False, True]],
        dtype=bool,
    )
    return teacher, student, mask


# =============================================================================
# Construction -- the F-01 regression guard
# =============================================================================


class TestConstruction:
    """Construction of all three losses.

    Every test in this class fails on the pre-fix code, where `DINOLoss` and
    `iBOTPatchLoss` called `self.add_weight(...)` in `__init__` and
    `keras.losses.Loss` (keras 3.8.0) has no such method.
    """

    def test_dino_loss_constructs(self):
        """`DINOLoss` instantiates and owns a zero-initialized center."""
        loss = DINOLoss(out_dim=8)
        assert isinstance(loss.center, keras.Variable)
        assert tuple(loss.center.shape) == (1, 8)
        assert loss.center.trainable is False
        np.testing.assert_array_equal(
            ops.convert_to_numpy(loss.center), np.zeros((1, 8), "float32")
        )

    def test_ibot_loss_constructs(self):
        """`iBOTPatchLoss` instantiates with a patch-broadcastable center."""
        loss = iBOTPatchLoss(out_dim=8)
        assert isinstance(loss.center, keras.Variable)
        assert tuple(loss.center.shape) == (1, 1, 8)
        assert loss.center.trainable is False

    def test_koleo_loss_constructs(self):
        """`KoLeoLoss` instantiates and is stateless."""
        loss = KoLeoLoss(epsilon=1e-8)
        assert loss.epsilon == 1e-8
        assert not hasattr(loss, "center")

    def test_center_is_float32_even_under_mixed_float16(self):
        """The center accumulator stays float32 under a float16 policy.

        A float16 centering accumulator would lose the small `(1 - momentum)`
        increments to rounding, which is exactly the silent-no-op failure the
        centering mechanism must not have.
        """
        previous = keras.mixed_precision.dtype_policy()
        try:
            keras.mixed_precision.set_dtype_policy("mixed_float16")
            assert DINOLoss(out_dim=4).center.dtype == "float32"
            assert iBOTPatchLoss(out_dim=4).center.dtype == "float32"
        finally:
            keras.mixed_precision.set_dtype_policy(previous)

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"out_dim": 0}, "out_dim must be positive"),
            ({"out_dim": 4, "student_temp": 0.0}, "student_temp must be positive"),
            ({"out_dim": 4, "teacher_temp": -1.0}, "teacher_temp must be positive"),
            ({"out_dim": 4, "center_momentum": 1.0}, "center_momentum must be in"),
        ],
    )
    @pytest.mark.parametrize("cls", [DINOLoss, iBOTPatchLoss])
    def test_invalid_arguments_raise(self, cls, kwargs, message):
        """Out-of-range constructor arguments raise a `ValueError` naming the field."""
        with pytest.raises(ValueError, match=message):
            cls(**kwargs)

    def test_koleo_rejects_non_positive_epsilon(self):
        """`KoLeoLoss` rejects a non-positive epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            KoLeoLoss(epsilon=0.0)


# =============================================================================
# Forward pass
# =============================================================================


class TestForward:
    """Forward-pass values and finiteness for all three losses."""

    def test_dino_forward_matches_numpy_oracle(self, cls_logits):
        """`DINOLoss` reproduces an independent numpy cross-entropy."""
        teacher, student = cls_logits
        loss = DINOLoss(out_dim=6, student_temp=0.1, teacher_temp=0.04)

        value = float(ops.convert_to_numpy(loss(teacher, student)))

        expected = float(
            _np_cross_entropy(
                teacher, student, np.zeros((1, 6)), 0.04, 0.1
            ).mean()
        )
        assert np.isfinite(value)
        np.testing.assert_allclose(value, expected, atol=ATOL, rtol=0.0)

    def test_dino_dict_and_tensor_conventions_agree(self, cls_logits):
        """The structured-dict and two-tensor conventions give the same loss.

        Neither of these is the stock-`fit()` convention -- that is the PACKED
        single tensor (see `TestPackedConvention`; D-009 measured that a dict
        `y_pred` is refused by `CompileLoss.build`). All three must agree, so
        that a test written against one convention is evidence about the
        others.
        """
        teacher, student = cls_logits
        tensor_value = float(ops.convert_to_numpy(DINOLoss(out_dim=6)(teacher, student)))
        dict_value = float(
            ops.convert_to_numpy(
                DINOLoss(out_dim=6)(
                    IGNORED_Y_TRUE,
                    {"teacher_logits": teacher, "student_logits": student},
                )
            )
        )
        np.testing.assert_allclose(dict_value, tensor_value, atol=ATOL, rtol=0.0)

    def test_dino_dict_missing_key_raises(self, cls_logits):
        """A dict `y_pred` missing `teacher_logits` raises a naming `KeyError`."""
        _, student = cls_logits
        with pytest.raises(KeyError, match="teacher_logits"):
            DINOLoss(out_dim=6)(IGNORED_Y_TRUE, {"student_logits": student})

    def test_dino_tensor_convention_without_teacher_raises(self, cls_logits):
        """A plain-tensor `y_pred` with `y_true=None` raises a `ValueError`."""
        _, student = cls_logits
        with pytest.raises(ValueError, match="teacher"):
            DINOLoss(out_dim=6).call(None, ops.convert_to_tensor(student))

    def test_ibot_forward_finite(self, patch_logits):
        """`iBOTPatchLoss` produces a finite scalar with a partial mask."""
        teacher, student, mask = patch_logits
        value = iBOTPatchLoss(out_dim=4)(
            IGNORED_Y_TRUE,
            {"teacher_logits": teacher, "student_logits": student, "mask": mask},
        )
        assert np.isfinite(float(ops.convert_to_numpy(value)))
        assert ops.convert_to_numpy(value).shape == ()

    def test_koleo_forward_finite(self):
        """`KoLeoLoss` produces a finite scalar."""
        rng = np.random.default_rng(5)
        value = KoLeoLoss()(IGNORED_Y_TRUE, rng.normal(size=(6, 5)).astype("float32"))
        assert np.isfinite(float(ops.convert_to_numpy(value)))

    def test_ibot_third_positional_mask_raises(self, patch_logits):
        """A third positional `mask` argument is refused, loudly.

        MEASURED on keras 3.8.0: `Loss.__call__` is
        `(y_true, y_pred, sample_weight=None)` and dispatches to a TWO-argument
        `call()`, so `loss(t, s, mask)` applies the mask as a `sample_weight`
        on the already-reduced scalar and returns a finite number while the
        masking logic never sees it. The `__call__` override turns that silent
        wrong answer into a `TypeError` naming the dict convention.
        """
        teacher, student, mask = patch_logits
        loss = iBOTPatchLoss(out_dim=4)
        with pytest.raises(TypeError, match="y_pred\\['mask'\\]"):
            loss(teacher, student, mask)


# =============================================================================
# The centering EMA under a real fit() -- the load-bearing test
# =============================================================================


class TestCenteringEMA:
    """The centering variable's behaviour under a real `model.fit()`."""

    @staticmethod
    def _build_two_batch_problem():
        """Build a frozen-weight 2-step training problem with a known oracle.

        The optimizer's learning rate is 0.0, so the student's outputs never
        influence the center; the EMA depends only on the teacher logits fed as
        `y`, which are known exactly. The oracle is therefore a pure float64
        recomputation of the EMA recurrence, not a re-run of the code path
        under test.

        Returns:
            Tuple `(model, x, teacher, batch_size, momentum, expected_center)`.
        """
        out_dim, momentum, batch_size = 3, 0.9, 2
        rng = np.random.default_rng(4242)
        x = rng.normal(size=(4, 5)).astype("float32")
        teacher = rng.normal(size=(4, out_dim)).astype("float32")

        keras.utils.set_random_seed(1234)
        model = keras.Sequential([keras.Input(shape=(5,)), keras.layers.Dense(out_dim)])

        center = np.zeros((1, out_dim), dtype=np.float64)
        for start in range(0, x.shape[0], batch_size):
            batch = np.asarray(teacher[start:start + batch_size], np.float64)
            center = (
                momentum * center
                + (1.0 - momentum) * batch.mean(axis=0, keepdims=True)
            )
        return model, x, teacher, batch_size, momentum, center

    @pytest.mark.parametrize("jit_compile", [False, True])
    def test_center_reaches_hand_computed_ema(self, jit_compile):
        """After a real 2-step `fit()` the center holds the exact EMA value.

        Run under both `jit_compile=False` and `jit_compile=True`: a stateful
        `.assign()` silently dropped by XLA is the single highest-severity
        failure mode of this design, and it would leave every other test in
        this file green.
        """
        (model, x, teacher, batch_size, momentum,
         expected) = self._build_two_batch_problem()
        loss = DINOLoss(out_dim=3, center_momentum=momentum)
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.0),
            loss=loss,
            jit_compile=jit_compile,
        )

        model.fit(x, teacher, epochs=1, batch_size=batch_size,
                  shuffle=False, verbose=0)

        measured = ops.convert_to_numpy(loss.center)

        # Non-vacuity: the correct value must be far from the initial zeros
        # relative to the tolerance, otherwise this assertion cannot tell a
        # live .assign() from a dead one.
        separation = float(np.max(np.abs(expected)))
        assert separation > 100 * ATOL, (
            f"vacuous test: expected center {expected} is only {separation} "
            f"from its zero initialization, against ATOL={ATOL}"
        )
        np.testing.assert_allclose(measured, expected, atol=ATOL, rtol=0.0)

    def test_center_does_not_move_without_a_call(self):
        """A constructed-but-never-called loss keeps a zero center.

        Guards the other direction: the center must move BECAUSE of `call()`,
        not because of construction or compilation.
        """
        loss = DINOLoss(out_dim=3)
        keras.Sequential(
            [keras.Input(shape=(5,)), keras.layers.Dense(3)]
        ).compile(optimizer="sgd", loss=loss)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(loss.center), np.zeros((1, 3), "float32")
        )

    def test_ibot_center_reaches_hand_computed_ema(self, patch_logits):
        """`iBOTPatchLoss`'s center EMAs over ALL patches, masked or not.

        The centering statistic is deliberately computed from every teacher
        patch, not only the participating ones -- a center estimated from the
        masked subset alone would drift with the mask ratio.
        """
        teacher, student, mask = patch_logits
        momentum = 0.9
        loss = iBOTPatchLoss(out_dim=4, center_momentum=momentum)

        loss(IGNORED_Y_TRUE, {"teacher_logits": teacher, "student_logits": student,
                    "mask": mask})

        expected = (1.0 - momentum) * np.asarray(
            teacher, np.float64
        ).mean(axis=(0, 1), keepdims=True)
        np.testing.assert_allclose(
            ops.convert_to_numpy(loss.center), expected, atol=ATOL, rtol=0.0
        )


# =============================================================================
# iBOT masking -- the F-12 boundary
# =============================================================================


class TestIBOTMasking:
    """Mask handling in `iBOTPatchLoss`, including both degenerate cases."""

    def test_partial_mask_matches_masked_subset_oracle(self, patch_logits):
        """The loss equals the mean over the PARTICIPATING patches only."""
        teacher, student, mask = patch_logits
        loss = iBOTPatchLoss(out_dim=4, student_temp=0.1, teacher_temp=0.04)

        value = float(ops.convert_to_numpy(loss(
            IGNORED_Y_TRUE,
            {"teacher_logits": teacher, "student_logits": student, "mask": mask},
        )))

        per_patch = _np_cross_entropy(
            teacher, student, np.zeros((1, 1, 4)), 0.04, 0.1
        )
        expected = float(per_patch[mask].mean())
        np.testing.assert_allclose(value, expected, atol=ATOL, rtol=0.0)

    def test_masked_value_differs_from_unmasked_value(self, patch_logits):
        """A partial mask gives a DIFFERENT loss than using every patch.

        This is the assertion that dies if the mask is ignored. Without it, a
        masked-loss implementation that silently averages over all patches
        passes every shape, finiteness and serialization test in this file.
        """
        teacher, student, mask = patch_logits
        masked = float(ops.convert_to_numpy(iBOTPatchLoss(out_dim=4)(
            IGNORED_Y_TRUE,
            {"teacher_logits": teacher, "student_logits": student, "mask": mask},
        )))
        unmasked = float(ops.convert_to_numpy(iBOTPatchLoss(out_dim=4)(
            IGNORED_Y_TRUE,
            {"teacher_logits": teacher, "student_logits": student},
        )))

        per_patch = _np_cross_entropy(
            teacher, student, np.zeros((1, 1, 4)), 0.04, 0.1
        )
        separation = abs(float(per_patch[mask].mean()) - float(per_patch.mean()))
        assert separation > 100 * ATOL, (
            f"vacuous fixture: the masked subset and the full set differ by "
            f"only {separation}; choose a mask that actually changes the mean"
        )
        assert abs(masked - unmasked) > 100 * ATOL

    def test_no_patch_masked_returns_exactly_zero(self, patch_logits):
        """An all-False mask yields exactly 0.0, with no branch and no NaN.

        This is the F-12 boundary. The pre-fix code tried to detect it with a
        Python `if` on the first dimension of an `ops.boolean_mask` output --
        a data-dependent shape. The branchless mask-weighted mean is correct
        here by construction: the numerator is 0 and the denominator is
        clamped away from 0.
        """
        teacher, student, _ = patch_logits
        value = float(ops.convert_to_numpy(iBOTPatchLoss(out_dim=4)(
            IGNORED_Y_TRUE,
            {
                "teacher_logits": teacher,
                "student_logits": student,
                "mask": np.zeros((2, 5), dtype=bool),
            },
        )))
        assert value == 0.0

    def test_all_patches_masked_equals_no_mask(self, patch_logits):
        """An all-True mask reproduces the unmasked mean exactly."""
        teacher, student, _ = patch_logits
        all_true = float(ops.convert_to_numpy(iBOTPatchLoss(out_dim=4)(
            IGNORED_Y_TRUE,
            {
                "teacher_logits": teacher,
                "student_logits": student,
                "mask": np.ones((2, 5), dtype=bool),
            },
        )))
        no_mask = float(ops.convert_to_numpy(iBOTPatchLoss(out_dim=4)(
            IGNORED_Y_TRUE,
            {"teacher_logits": teacher, "student_logits": student},
        )))
        np.testing.assert_allclose(all_true, no_mask, atol=ATOL, rtol=0.0)

    @pytest.mark.parametrize(
        "mask_fill", [None, False, True], ids=["no_mask", "empty", "full"]
    )
    def test_mask_boundary_under_a_graph_trace(self, patch_logits, mask_fill):
        """Every mask case survives a real graph trace, not just eager.

        This is the F-12 boundary proper. The pre-fix body could not be traced
        at all: its first statement called `keras.ops.boolean_mask`, which does
        not exist on keras 3.8.0, and the Python `if` on the masked tensor's
        first dimension behind it was never reached. `tf.function` here is a
        TEST instrument -- the shipped `call()` is pure `keras.ops`.
        """
        import tensorflow as tf

        teacher, student, _ = patch_logits
        loss = iBOTPatchLoss(out_dim=4)

        @tf.function
        def traced(t, s, m):
            payload = {"teacher_logits": t, "student_logits": s}
            if m is not None:
                payload["mask"] = m
            return loss.call(None, payload)

        mask = (
            None if mask_fill is None
            else tf.constant(np.full((2, 5), mask_fill, dtype=bool))
        )
        value = float(traced(tf.constant(teacher), tf.constant(student), mask))
        assert np.isfinite(value)
        if mask_fill is False:
            assert value == 0.0

    def test_ibot_trains_under_stock_fit(self):
        """`iBOTPatchLoss` runs as a `compile(loss=...)` argument.

        Uses the two-tensor convention -- teacher patch logits as `y_true`,
        student patch logits as the model's single-tensor output. A dict model
        output is NOT usable here: keras 3.8's `CompileLoss.build` broadcasts a
        single `Loss` across every leaf of a nested `y_pred`
        (`loss = tree.map_structure(lambda x: loss, y_pred)`), so each copy
        would see one leaf rather than the whole structure.
        """
        rng = np.random.default_rng(11)
        num_patches, out_dim = 5, 4
        x = rng.normal(size=(4, num_patches, 3)).astype("float32")
        teacher = rng.normal(size=(4, num_patches, out_dim)).astype("float32")

        model = keras.Sequential([keras.layers.Dense(out_dim)])
        model.compile(
            optimizer=keras.optimizers.SGD(0.0),
            loss=iBOTPatchLoss(out_dim=out_dim, center_momentum=0.9),
        )
        history = model.fit(x, teacher, epochs=1, batch_size=2,
                            shuffle=False, verbose=0)
        assert np.isfinite(history.history["loss"][0])


# =============================================================================
# Serialization
# =============================================================================


class TestSerialization:
    """`get_config()` / `from_config()` round-trips, including center VALUE."""

    def test_dino_config_round_trip_restores_center_value(self):
        """A non-zero center survives `from_config(get_config())`.

        Keras does NOT checkpoint loss-owned variables (MEASURED: a `.keras`
        round-trip returns the right subclass with a center reading back as
        zeros), so the value rides in the config. A round-trip test that only
        compared hyperparameters would pass while the state was dropped.
        """
        loss = DINOLoss(out_dim=5, student_temp=0.2, teacher_temp=0.05,
                        center_momentum=0.8)
        seeded = np.linspace(-1.0, 1.0, 5, dtype="float32").reshape(1, 5)
        loss.center.assign(seeded)

        restored = DINOLoss.from_config(loss.get_config())

        assert restored.out_dim == 5
        assert restored.student_temp == 0.2
        assert restored.teacher_temp == 0.05
        assert restored.center_momentum == 0.8
        np.testing.assert_allclose(
            ops.convert_to_numpy(restored.center), seeded, atol=0.0, rtol=0.0
        )

    def test_ibot_config_round_trip_restores_center_value(self):
        """The same guarantee for `iBOTPatchLoss`'s `(1, 1, out_dim)` center."""
        loss = iBOTPatchLoss(out_dim=5, center_momentum=0.7)
        seeded = np.linspace(2.0, 3.0, 5, dtype="float32").reshape(1, 1, 5)
        loss.center.assign(seeded)

        restored = iBOTPatchLoss.from_config(loss.get_config())

        assert restored.center_momentum == 0.7
        np.testing.assert_allclose(
            ops.convert_to_numpy(restored.center), seeded, atol=0.0, rtol=0.0
        )

    def test_koleo_config_round_trip(self):
        """`KoLeoLoss` round-trips its one hyperparameter."""
        restored = KoLeoLoss.from_config(KoLeoLoss(epsilon=1e-6).get_config())
        assert restored.epsilon == 1e-6

    @pytest.mark.parametrize(
        "loss",
        [DINOLoss(out_dim=4), iBOTPatchLoss(out_dim=4), KoLeoLoss()],
        ids=["dino", "ibot", "koleo"],
    )
    def test_config_is_json_serializable(self, loss):
        """Every config is plain JSON -- `.keras` saving requires it."""
        json.dumps(loss.get_config())

    def test_config_carries_every_constructor_argument(self):
        """No constructor argument is missing from `get_config()`."""
        for cls, expected in (
            (DINOLoss, {"out_dim", "student_temp", "teacher_temp",
                        "center_momentum"}),
            (iBOTPatchLoss, {"out_dim", "student_temp", "teacher_temp",
                             "center_momentum"}),
            (KoLeoLoss, {"epsilon"}),
        ):
            config = cls(out_dim=4) if cls is not KoLeoLoss else cls()
            missing = expected - set(config.get_config())
            assert not missing, f"{cls.__name__}.get_config() is missing {missing}"

    def test_from_config_tolerates_a_missing_center(self):
        """A config written before centers were serialized still loads."""
        config = DINOLoss(out_dim=4).get_config()
        config.pop("center")
        restored = DINOLoss.from_config(config)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(restored.center), np.zeros((1, 4), "float32")
        )

    def test_model_keras_round_trip_with_dino_loss(self, cls_logits):
        """A model compiled with `DINOLoss` saves and reloads as `.keras`.

        Also pins the fact that the reloaded loss is the registered subclass,
        which is what makes the config-carried center the effective restore
        path for the centering state.
        """
        teacher, _ = cls_logits
        model = keras.Sequential([keras.Input(shape=(5,)), keras.layers.Dense(6)])
        loss = DINOLoss(out_dim=6)
        loss.center.assign(np.full((1, 6), 0.25, dtype="float32"))
        model.compile(optimizer="sgd", loss=loss)
        model.predict(np.zeros((1, 5), "float32"), verbose=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)

        assert isinstance(reloaded.loss, DINOLoss)
        assert reloaded.loss.out_dim == 6


# =============================================================================
# KoLeo regression -- pinning the one class that already worked
# =============================================================================


class TestKoLeoRegression:
    """Value and directional regressions for `KoLeoLoss`."""

    def test_matches_numpy_oracle(self):
        """The loss matches an independent numpy reimplementation."""
        rng = np.random.default_rng(909)
        embeddings = rng.normal(size=(7, 4)).astype("float32")
        loss = KoLeoLoss(epsilon=1e-8)

        value = float(ops.convert_to_numpy(loss(IGNORED_Y_TRUE, embeddings)))

        np.testing.assert_allclose(
            value, _np_koleo(embeddings, 1e-8), atol=ATOL, rtol=0.0
        )

    def test_penalizes_collapsed_embeddings_more_than_spread_ones(self):
        """Near-identical embeddings score strictly worse than orthogonal ones.

        This is the property the regularizer exists for: it must fire on
        collapse. A value-only test would pass on a loss that ignores the
        geometry entirely.
        """
        spread = np.eye(4, dtype="float32")
        collapsed = np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"), (4, 1)
        ) + 1e-4 * np.eye(4, dtype="float32")

        loss = KoLeoLoss(epsilon=1e-8)
        spread_value = float(ops.convert_to_numpy(loss(IGNORED_Y_TRUE, spread)))
        collapsed_value = float(ops.convert_to_numpy(loss(IGNORED_Y_TRUE, collapsed)))

        assert collapsed_value > spread_value + 1.0, (
            f"KoLeo failed to penalize collapse: collapsed={collapsed_value}, "
            f"spread={spread_value}"
        )

    def test_gradient_flows(self):
        """The loss produces a non-zero, finite gradient w.r.t. its input."""
        import tensorflow as tf

        rng = np.random.default_rng(3)
        embeddings = tf.Variable(rng.normal(size=(5, 4)).astype("float32"))
        loss = KoLeoLoss()
        with tf.GradientTape() as tape:
            value = loss(IGNORED_Y_TRUE, embeddings)
        grad = tape.gradient(value, embeddings).numpy()
        assert np.all(np.isfinite(grad))
        assert np.abs(grad).max() > 0.0


# =============================================================================
# The PACKED single-tensor convention (D-009 / plan step 8)
# =============================================================================


class TestPackedConvention:
    """`y_pred` of last dimension `2 * out_dim`: student first, teacher second.

    This is the ONLY convention that works under stock `compile(loss=...)` /
    `fit()` when the teacher's logits come from the model rather than from the
    dataset. The dict form is refused by `CompileLoss.build` (D-009) and the
    two-tensor form would require the DATASET to supply the teacher's logits.
    """

    def test_packed_agrees_with_the_two_tensor_call(self, cls_logits):
        from dl_techniques.losses.dino_loss import pack_student_teacher

        teacher, student = cls_logits
        packed = ops.convert_to_numpy(pack_student_teacher(student, teacher))
        assert packed.shape == (teacher.shape[0], 2 * teacher.shape[1])

        direct = float(ops.convert_to_numpy(
            DINOLoss(out_dim=6)(teacher, student)))
        from_packed = float(ops.convert_to_numpy(
            DINOLoss(out_dim=6)(IGNORED_Y_TRUE, packed)))
        np.testing.assert_allclose(from_packed, direct, rtol=1e-6, atol=1e-6)

        # Non-vacuity: the two halves must not be interchangeable, or the
        # agreement above would hold for a swapped layout too.
        swapped = np.concatenate([teacher, student], axis=-1)
        swapped_value = float(ops.convert_to_numpy(
            DINOLoss(out_dim=6)(IGNORED_Y_TRUE, swapped)))
        assert abs(swapped_value - direct) > 1e-3, (
            "swapping the student and teacher halves changed nothing, so this "
            "test cannot see the layout at all"
        )

    def test_packed_ignores_y_true(self, cls_logits):
        from dl_techniques.losses.dino_loss import pack_student_teacher

        teacher, student = cls_logits
        packed = pack_student_teacher(student, teacher)
        loss = DINOLoss(out_dim=6)
        a = float(ops.convert_to_numpy(loss(IGNORED_Y_TRUE, packed)))
        loss_b = DINOLoss(out_dim=6)
        b = float(ops.convert_to_numpy(
            loss_b(np.full((4, 6), 99.0, "float32"), packed)))
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_a_wrong_width_plain_tensor_raises_naming_both_widths(self):
        loss = DINOLoss(out_dim=6)
        wrong = np.zeros((4, 7), "float32")
        with pytest.raises(ValueError, match=r"2 \* out_dim = 12"):
            loss.call(None, wrong)

    # -- The width check `pack_student_teacher`'s contract promises (D-038).
    #
    # Before it existed, BOTH cases below reached the arithmetic. MEASURED on a
    # correctly packed (6, 16) tensor plus a (6, 1) label:
    #   out_dim=9  -> InvalidArgumentError: required broadcastable shapes
    #                 [Op:Mul], naming neither width.
    #   out_dim=16 -> NO ERROR AT ALL, value 16.5552, computed from the LABEL
    #                 as the teacher's logits.
    # The second is why the check cannot be a y_pred-only check.

    @pytest.mark.parametrize("bad_out_dim", [9, 16])
    def test_a_packed_tensor_whose_width_misses_2x_out_dim_raises(
            self, bad_out_dim):
        """A width miss is a named `ValueError`, not a backend broadcast error.

        `out_dim=9` misses on `y_pred` (16 is neither 9 nor 18); `out_dim=16`
        misses on `y_true` (the label is 1 wide, not 16) and is the case that
        previously returned a number silently.
        """
        from dl_techniques.losses.dino_loss import pack_student_teacher

        rng = np.random.default_rng(0)
        student = rng.normal(size=(6, 8)).astype("float32")
        teacher = rng.normal(size=(6, 8)).astype("float32")
        packed = ops.convert_to_numpy(pack_student_teacher(student, teacher))
        label = np.zeros((6, 1), "float32")

        with pytest.raises(ValueError) as excinfo:
            DINOLoss(out_dim=bad_out_dim)(label, packed)

        message = str(excinfo.value)
        # The docstring's promise is specifically that BOTH widths are named.
        assert f"out_dim = {bad_out_dim}" in message, message
        assert f"2 * out_dim = {2 * bad_out_dim}" in message, message
        assert "16" in message or "1" in message, message

        # NON-VACUITY CONTROL: the same packed tensor at the CORRECT out_dim
        # must still score, or this test would pass on a loss that refuses
        # everything.
        value = float(ops.convert_to_numpy(DINOLoss(out_dim=8)(label, packed)))
        assert np.isfinite(value)

    def test_the_ibot_width_check_fires_on_a_rank_3_packed_tensor(self):
        """The same check reaches `iBOTPatchLoss`'s rank-3 packed form."""
        from dl_techniques.losses.dino_loss import pack_student_teacher

        rng = np.random.default_rng(1)
        student = rng.normal(size=(2, 5, 8)).astype("float32")
        teacher = rng.normal(size=(2, 5, 8)).astype("float32")
        packed = ops.convert_to_numpy(pack_student_teacher(student, teacher))
        label = np.zeros((2, 5, 1), "float32")

        with pytest.raises(ValueError, match=r"2 \* out_dim = 18"):
            iBOTPatchLoss(out_dim=9)(label, packed)

    def test_a_student_only_tensor_of_packed_width_is_NOT_caught(self):
        """CHARACTERIZATION of the residual ambiguity -- a known boundary.

        The conventions are told apart by width alone, so a tensor that is
        `2 * out_dim` wide for the WRONG reason is read as packed and scored.
        This test pins the measured behaviour so that a future reader meets the
        limit here rather than discovering it in a training run. It is NOT an
        endorsement: closing it needs a signal other than width, which is not
        available at trace time.
        """
        rng = np.random.default_rng(2)
        student_only = rng.normal(size=(6, 8)).astype("float32")
        label = np.zeros((6, 1), "float32")

        # 8 == 2 * 4, so this is accepted and scored as if it were packed.
        value = float(ops.convert_to_numpy(
            DINOLoss(out_dim=4)(label, student_only)))
        assert np.isfinite(value), (
            "if this now RAISES, the width check gained a signal beyond width "
            "and this characterization test should be replaced by a raise test"
        )

    def test_packed_trains_under_stock_fit(self):
        """A model emitting the packed tensor trains, and the center moves."""
        from dl_techniques.losses.dino_loss import pack_student_teacher

        out_dim = 5

        class _PackedModel(keras.Model):
            def __init__(self):
                super().__init__(name="packed_model")
                self.student = keras.layers.Dense(out_dim)
                self.teacher = keras.layers.Dense(out_dim)
                self.teacher.trainable = False

            def call(self, inputs, training=None):
                student = self.student(inputs)
                teacher = ops.stop_gradient(self.teacher(inputs))
                return pack_student_teacher(student, teacher)

        rng = np.random.default_rng(4242)
        x = rng.normal(size=(8, 3)).astype("float32")
        y = np.zeros((8, 1), "float32")

        model = _PackedModel()
        loss = DINOLoss(out_dim=out_dim)
        model.compile(optimizer=keras.optimizers.SGD(0.0), loss=loss)

        before = ops.convert_to_numpy(loss.center).copy()
        # shuffle=False: the hand-computed EMA below assumes the two
        # batches arrive in index order.
        history = model.fit(x, y, epochs=1, batch_size=4, verbose=0,
                            shuffle=False)

        assert np.isfinite(history.history["loss"][0])
        after = ops.convert_to_numpy(loss.center)
        assert not np.array_equal(before, after), (
            "the center did not move: the packed y_pred never reached the "
            "centering EMA"
        )
        # The center must equal the EMA of the TEACHER half specifically.
        packed = ops.convert_to_numpy(model(x))
        teacher_half = packed[:, out_dim:]
        expected = np.zeros((1, out_dim), np.float64)
        for start in (0, 4):
            batch_center = teacher_half[start:start + 4].astype(
                np.float64).mean(axis=0, keepdims=True)
            expected = 0.9 * expected + 0.1 * batch_center
        np.testing.assert_allclose(after, expected, rtol=1e-4, atol=1e-6)

    def test_ibot_also_accepts_the_packed_form(self, patch_logits):
        from dl_techniques.losses.dino_loss import pack_student_teacher

        teacher, student, _ = patch_logits
        packed = pack_student_teacher(student, teacher)
        direct = float(ops.convert_to_numpy(
            iBOTPatchLoss(out_dim=4)(teacher, student)))
        from_packed = float(ops.convert_to_numpy(
            iBOTPatchLoss(out_dim=4)(IGNORED_Y_TRUE, packed)))
        np.testing.assert_allclose(from_packed, direct, rtol=1e-6, atol=1e-6)


# =============================================================================
# A SCHEDULABLE teacher temperature (D-022 / plan step 8)
# =============================================================================


class TestSchedulableTeacherTemperature:
    """`teacher_temp` is a `keras.Variable` behind a read-only property.

    A Python-float temperature is constant-folded into the traced training
    step, so a warmup schedule that assigns it is a SILENT no-op. The tests
    here pin (a) that the property reads back exactly, (b) that a plain
    assignment RAISES rather than silently doing nothing, and (c) that
    `set_teacher_temp` really reaches a compiled `fit()`.
    """

    @pytest.mark.parametrize("cls", [DINOLoss, iBOTPatchLoss])
    def test_property_reads_back_the_exact_constructor_value(self, cls):
        loss = cls(out_dim=4, teacher_temp=0.07)
        assert isinstance(loss.teacher_temp, float)
        assert loss.teacher_temp == 0.07
        assert isinstance(loss._teacher_temp, keras.Variable)
        assert loss._teacher_temp.trainable is False

    @pytest.mark.parametrize("cls", [DINOLoss, iBOTPatchLoss])
    def test_plain_assignment_is_refused(self, cls):
        loss = cls(out_dim=4)
        with pytest.raises(AttributeError, match="set_teacher_temp"):
            loss.teacher_temp = 0.5

    @pytest.mark.parametrize("cls", [DINOLoss, iBOTPatchLoss])
    def test_set_teacher_temp_validates(self, cls):
        loss = cls(out_dim=4)
        with pytest.raises(ValueError, match="teacher_temp must be positive"):
            loss.set_teacher_temp(0.0)

    def test_set_teacher_temp_reaches_a_compiled_fit(self):
        """The whole point: a schedule must change a RUNNING training step.

        The centering EMA is frozen (`center_momentum` ~ 1.0) so the ONLY
        thing that can move the reported loss between the two epochs is the
        temperature.
        """
        out_dim = 6
        rng = np.random.default_rng(99)
        x = rng.normal(size=(8, 4)).astype("float32")
        teacher = (rng.normal(size=(8, out_dim)) * 3.0).astype("float32")

        loss = DINOLoss(out_dim=out_dim, center_momentum=0.999999)
        model = keras.Sequential([keras.layers.Dense(out_dim)])
        model.compile(optimizer=keras.optimizers.SGD(0.0), loss=loss)

        first = model.fit(x, teacher, epochs=1, batch_size=8,
                          verbose=0).history["loss"][0]
        loss.set_teacher_temp(4.0)
        second = model.fit(x, teacher, epochs=1, batch_size=8,
                           verbose=0).history["loss"][0]

        assert abs(second - first) > 0.5, (
            f"a 100x teacher_temp change moved the training loss by "
            f"{abs(second - first):.3e}; the temperature is being "
            f"constant-folded into the traced step (D-022)"
        )
        assert loss.teacher_temp == 4.0

    def test_config_round_trip_carries_the_current_temperature(self):
        loss = DINOLoss(out_dim=4, teacher_temp=0.04)
        loss.set_teacher_temp(0.07)
        restored = DINOLoss.from_config(loss.get_config())
        assert restored.teacher_temp == 0.07


# =============================================================================
# KoLeo's fp16 normalization overflow (D-023, the D-020 defect class)
# =============================================================================


class TestKoLeoFp16NormalizationOverflow:
    """`ops.normalize` in a 16-bit dtype makes KoLeo a DEAD regularizer.

    Not a toy-size test on purpose: at small widths / magnitudes the fp16 path
    is correct to ~1e-4 and every assertion here would be vacuous. The
    configuration is pinned at a width and magnitude where `sum(x**2)` really
    does exceed fp16's 65504, asserted at RUNTIME below.
    """

    WIDTH = 256
    SCALE = 300.0

    def _embeddings(self, seed=0):
        rng = np.random.RandomState(seed)
        return (rng.normal(size=(16, self.WIDTH)) * self.SCALE).astype("float32")

    def test_the_fixture_really_overflows_fp16(self):
        """Runtime non-vacuity floor for every test in this class."""
        embeddings = self._embeddings()
        sumsq = float(np.sum(embeddings[0].astype(np.float64) ** 2))
        assert sumsq > 65504.0, (
            f"sum(x**2) = {sumsq:.3e} does not exceed fp16's 65504, so these "
            f"tests would pass with the fix reverted"
        )

    def test_fp16_matches_float32(self):
        embeddings = self._embeddings()
        reference = float(ops.convert_to_numpy(
            KoLeoLoss()(IGNORED_Y_TRUE, embeddings)))
        fp16 = float(ops.convert_to_numpy(
            KoLeoLoss(dtype="float16")(IGNORED_Y_TRUE, embeddings)))
        np.testing.assert_allclose(fp16, reference, rtol=2e-2, atol=2e-3)

    def test_fp16_still_distinguishes_collapse_from_spread(self):
        """The defect's real damage: a CONSTANT loss with a zero gradient.

        With the normalization overflowing, every embedding becomes the zero
        vector, every pairwise cosine similarity becomes 0, and the loss
        collapses to `-log(sqrt(2) + eps) = -0.34657` for ANY input -- so the
        anti-collapse regularizer stops seeing collapse.
        """
        rng = np.random.RandomState(1)
        spread = self._embeddings()
        collapsed = np.tile(
            rng.normal(size=(1, self.WIDTH)) * self.SCALE, (16, 1)
        ).astype("float32") + rng.normal(size=(16, self.WIDTH)).astype("float32")

        loss = KoLeoLoss(dtype="float16")
        spread_value = float(ops.convert_to_numpy(
            loss(IGNORED_Y_TRUE, spread)))
        collapsed_value = float(ops.convert_to_numpy(
            loss(IGNORED_Y_TRUE, collapsed)))

        assert collapsed_value > spread_value + 1.0, (
            f"under fp16 the KoLeo loss cannot tell a collapsed batch "
            f"({collapsed_value:.6f}) from a spread one ({spread_value:.6f}). "
            f"A value pinned at -0.34657 for both is the D-023 overflow."
        )

    def test_fp16_gradient_is_not_dead(self):
        import tensorflow as tf

        embeddings = tf.Variable(self._embeddings())
        loss = KoLeoLoss(dtype="float16")
        with tf.GradientTape() as tape:
            value = loss(IGNORED_Y_TRUE, embeddings)
        grad = tape.gradient(value, embeddings).numpy()
        assert np.all(np.isfinite(grad))
        assert np.abs(grad).max() > 0.0, (
            "the fp16 KoLeo gradient is EXACTLY zero -- the regularizer "
            "contributes nothing to training (D-023)"
        )

    def test_float64_is_not_downcast(self):
        """The promotion must not pin float64 callers to float32."""
        embeddings = self._embeddings().astype("float64")
        value = KoLeoLoss(dtype="float64")(
            IGNORED_Y_TRUE.astype("float64"), embeddings)
        assert keras.backend.standardize_dtype(value.dtype) == "float64"
        assert np.isfinite(float(ops.convert_to_numpy(value)))
